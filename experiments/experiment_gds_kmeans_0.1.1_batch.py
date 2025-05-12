#!/usr/bin/env python3
import os, sys, logging, pandas as pd, numpy as np, ast
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

# -----------------------------------------------------------------------------  
# 0. LOG CONFIGURATION  
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s ▶ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger()

# -----------------------------------------------------------------------------  
# 1. CONNECT TO NEO4J & GDS  
# -----------------------------------------------------------------------------
URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7688")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD  = os.getenv("NEO4J_PASSWORD", "123412345")

log.info("▶ Connecting to Neo4j…")
try:
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    driver.verify_connectivity()
    log.info("  ✅ Neo4j connection OK")
except Exception as e:
    log.error("  ❌ Could not connect to Neo4j", exc_info=e)
    sys.exit(1)

log.info("▶ Initializing GDS client…")
gds = GraphDataScience(URI, auth=(USER, PWD))
log.info(f"  → GDS version: {gds.version()}")

# -----------------------------------------------------------------------------  
# 2. DATA‐LOADING HELPERS  
# -----------------------------------------------------------------------------
def load_tabular(fp):
    df = pd.read_csv(fp, index_col="Index")
    features = df.drop(columns=["target"])
    k = int(df["target"].nunique())
    return features, k

def load_graph(nodes_fp):
    df = pd.read_csv(nodes_fp)
    arrs = df["features"].apply(ast.literal_eval).apply(np.array)
    features = pd.DataFrame(arrs.tolist(), index=df["id"])
    k = int(df["label"].nunique())
    return features, k

# -----------------------------------------------------------------------------  
# 3. PER‐DATASET WORKFLOW  
# -----------------------------------------------------------------------------
def process(ds):
    name, typ = ds["name"], ds["type"]
    label, gname = ds["label"], ds["graph_name"]
    log.info(f"\n▶ Processing `{name}` ({typ})")

    # 3a) load features + infer k
    if typ == "tabular":
        feats, k = load_tabular(ds["file_path"])
    else:
        feats, k = load_graph(ds["nodes_path"])
    log.info(f"  • Loaded {len(feats)} rows × {feats.shape[1]} cols; k={k}")

    # 3b) ingest nodes
    with driver.session() as sess:
        sess.run(f"MATCH (n:{label}) DETACH DELETE n")
        rows = [{"id": int(i), "features": row.tolist()} 
                for i, row in feats.iterrows()]
        sess.run(
            f"UNWIND $rows AS r CREATE (n:{label} {{id:r.id,features:r.features}})",
            rows=rows
        )
    log.info("  ✓ Nodes ingested")

    # 3c) ingest edges if graph dataset
    if typ == "graph":
        with driver.session() as sess:
            sess.run(f"MATCH ()-[r:{ds['edge_type']}]->() DELETE r")
            edges_df = pd.read_csv(ds["edges_path"])
            edges_df = edges_df.drop_duplicates(subset=["source_id","target_id"])
            edges = edges_df.to_dict("records")
            sess.run(f"""
                UNWIND $es AS e
                WITH DISTINCT e.source_id AS src, e.target_id AS tgt
                MATCH (a:{label} {{id:src}}),(b:{label} {{id:tgt}})
                MERGE (a)-[:{ds['edge_type']}]->(b)
            """, es=edges)
        log.info("  ✓ Edges ingested (deduped)")

    # 3d) drop old projection
    exists = bool(gds.graph.exists(gname).squeeze()["exists"])
    if exists:
        gds.graph.drop(gname)
        log.info(f"  ↻ Dropped old graph `{gname}`")

    # 3e) cypher projection
    if typ == "tabular":
        rel_cypher = f"MATCH (n:{label}) RETURN id(n) AS source, id(n) AS target"
    else:
        rel_cypher = (
            f"MATCH (a:{label})-[r:{ds['edge_type']}]->(b:{label}) "
            "RETURN id(a) AS source, id(b) AS target"
        )

    log.info(f"  ▶ Projecting graph `{gname}` …")
    proj = gds.graph.project.cypher(
        gname,
        f"MATCH (n:{label}) RETURN id(n) AS id, n.features AS features",
        rel_cypher,
        validateRelationships=False
    )
    log.info(f"   ✓ Projected: {proj.graph.node_count} nodes, "
             f"{proj.graph.relationship_count} edges")

    # 3f) run K-means & write clusterId
    G = gds.graph.get(gname)
    log.info(f"  ▶ Running k-means (k={k}) …")
    summary = gds.kmeans.write(
        G,
        nodeProperty  = "features",
        writeProperty = "clusterId",
        k             = k,
        maxIterations = 10
    )
    log.info(f"   ✓ k-means done (avgDist={summary.averageDistanceToCentroid:.3f})")

    # 3g) create Centroid nodes + ASSIGNED_TO edges
    centroids = summary.centroids
    log.info("  ▶ Creating Centroid nodes and ASSIGNED_TO edges …")
    with driver.session() as sess:
        # remove old Centroid artifacts
        sess.run("MATCH (c:Centroid {graph:$g}) DETACH DELETE c", g=gname)
        # create new centroids
        centroid_rows = [
            {"cluster": int(i), "features": list(centroid)} 
            for i, centroid in enumerate(centroids)
        ]
        sess.run("""
            UNWIND $rows AS r
            CREATE (c:Centroid {
                graph: $g,
                cluster: r.cluster,
                features: r.features
            })
        """, rows=centroid_rows, g=gname)
        log.info(f"   ✓ Created {len(centroid_rows)} Centroid nodes")

        # connect data nodes to their centroid
        sess.run(f"""
            MATCH (n:{label}), (c:Centroid {{graph:$g}})
            WHERE n.clusterId = c.cluster
            MERGE (n)-[:ASSIGNED_TO]->(c)
        """, g=gname)
        log.info("   ✓ Created ASSIGNED_TO edges")

    # 3h) sample a few assignments
    with driver.session() as sess:
        recs = sess.run(f"""
            MATCH (n:{label})-[:ASSIGNED_TO]->(c:Centroid {{graph:$g}})
            RETURN n.id AS id, c.cluster AS c
            ORDER BY n.id LIMIT 5
        """, g=gname)
        sample = ", ".join(f"{r['id']}→{r['c']}" for r in recs)
    log.info(f"  • Sample connections: {sample}")

    # 3i) cleanup projection
    gds.graph.drop(gname)
    log.info(f"  ✓ Dropped graph `{gname}`")

# -----------------------------------------------------------------------------  
# 4. MAIN  
# -----------------------------------------------------------------------------
def main():
    datasets = [
        # feature
        {"name":"iris",         "type":"tabular", "file_path":"datasets/iris.csv",
         "label":"Iris",        "graph_name":"iris_graph"},
        {"name":"madelon",      "type":"tabular", "file_path":"datasets/madelon.csv",
         "label":"Madelon",     "graph_name":"madelon_graph"},
        {"name":"20newsgroups", "type":"tabular", "file_path":"datasets/20newsgroups.csv",
         "label":"NewsGroup",   "graph_name":"newsgroups_graph"},
        # graph
        {"name":"cora",     "type":"graph",
         "nodes_path":"datasets/cora_nodes.csv","edges_path":"datasets/cora_edges.csv",
         "edge_type":"CORA_EDGE","label":"CoraNode","graph_name":"cora_graph"},
        {"name":"citeseer", "type":"graph",
         "nodes_path":"datasets/citeseer_nodes.csv","edges_path":"datasets/citeseer_edges.csv",
         "edge_type":"CITESEER_EDGE","label":"CiteSeerNode","graph_name":"citeseer_graph"},
        {"name":"pubmed",   "type":"graph",
         "nodes_path":"datasets/pubmed_nodes.csv","edges_path":"datasets/pubmed_edges.csv",
         "edge_type":"PUBMED_EDGE","label":"PubMedNode","graph_name":"pubmed_graph"},
    ]

    for ds in datasets:
        try:
            process(ds)
        except Exception:
            log.exception(f"❌ Error processing {ds['name']} — skipping")

    log.info("✔ All done; closing connections.")
    driver.close()
    gds.close()

if __name__ == "__main__":
    main()
