# experiment_gds_kmeans.py
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. CONNECT & PRINT VERSIONS
# ------------------------------------------------------------------
URI = "bolt://localhost:7687"
USER, PWD = "neo4j", "123412345"

print("[INFO] ▶ Connecting to Neo4j…", end=" ")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))
driver.verify_connectivity()
print("OK")

# get_server_info() returns a ServerInfo mapping-like object
server_info = driver.get_server_info()
print(f"[INFO]  • Server info   : {server_info}")

# init GDS client
gds = GraphDataScience(URI, auth=(USER, PWD))
print(f"[INFO]  • GDS version   : {gds.version()}")

# ------------------------------------------------------------------
# 2. LOAD IRIS INTO PANDAS
# ------------------------------------------------------------------
iris = load_iris(as_frame=True).frame
iris_df = iris.drop(columns=["target"])
# tidy up column names
iris_df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in iris_df.columns]
print(f"[INFO] Loaded Iris into DataFrame: {iris_df.shape[0]}×{iris_df.shape[1]}")

# ------------------------------------------------------------------
# 3. INGEST NODES WITH ARRAY PROPERTY
# ------------------------------------------------------------------
LABEL = "Iris"
print(f"[INFO] Clearing out any existing :{LABEL} nodes…")
with driver.session() as sess:
    sess.run(f"MATCH (n:{LABEL}) DETACH DELETE n")

rows = [
    {"id": int(idx), "features": row.tolist()}
    for idx, row in iris_df.iterrows()
]
print(f"[INFO] Ingesting {len(rows)} nodes with `features` array…")
with driver.session() as sess:
    sess.run("""
        UNWIND $rows AS r
        CREATE (:Iris {id:r.id, features:r.features})
    """, rows=rows)
print("[INFO] ✓ Nodes ingested")

# ------------------------------------------------------------------
# 4. PROJECT A “SELF-LOOP” GRAPH
# ------------------------------------------------------------------
GRAPH_NAME = "iris_graph"
print(f"[INFO] Checking for existing graph ‘{GRAPH_NAME}’…")
exists_series = gds.graph.exists(GRAPH_NAME)
print("[DEBUG] gds.graph.exists →")
print(exists_series)
# pull out the boolean
exists_flag = bool(exists_series["exists"])
print(f"[INFO] Graph exists? → {exists_flag}")
if exists_flag:
    print(f"[INFO] Dropping old graph ‘{GRAPH_NAME}’…")
    gds.graph.drop(GRAPH_NAME)

print(f"[INFO] Projecting new graph ‘{GRAPH_NAME}’ with self-loops…")
proj_summary = gds.graph.project.cypher(
    GRAPH_NAME,
    # nodeQuery: pull id & features
    "MATCH (n:Iris) RETURN id(n) AS id, n.features AS features",
    # relationshipQuery: self-loop on each node
    "MATCH (n:Iris) RETURN id(n) AS source, id(n) AS target"
)
print("[INFO] Projection summary:")
print(proj_summary)

# ------------------------------------------------------------------
# 5. RUN K-MEANS & WRITE BACK clusterId
# ------------------------------------------------------------------
print("[INFO] Running K-means (k=3)…")
# need a Graph object, not just the name
G = gds.graph.get(GRAPH_NAME)
kmeans_summary = gds.kmeans.write(
    G,
    nodeProperty  = "features",    # array property
    writeProperty = "clusterId",
    k             = 3,
    maxIterations = 10
)
print("[INFO] K-means summary:")
print(kmeans_summary)

# ------------------------------------------------------------------
# 6. VERIFY FIRST 10 ASSIGNMENTS
# ------------------------------------------------------------------
print("[INFO] Verifying first 10 cluster assignments:")
with driver.session() as sess:
    for rec in sess.run("""
        MATCH (i:Iris)
        WHERE i.clusterId IS NOT NULL
        RETURN i.id AS id, i.clusterId AS c
        ORDER BY i.id
        LIMIT 10
    """):
        print(f"  Node {rec['id']:>3} → cluster {rec['c']}")

# ------------------------------------------------------------------
# 7. CLEAN UP
# ------------------------------------------------------------------
#print(f"[INFO] Dropping graph ‘{GRAPH_NAME}’…")
#gds.graph.drop(GRAPH_NAME)

driver.close()
gds.close()
print("[INFO] All done.")
