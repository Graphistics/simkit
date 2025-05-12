import os
import time
import pandas as pd
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import numpy as np # For eval if features are strings

# Neo4j and GDS connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123412345"

# Dataset configurations
DATASETS_CONFIG = [
    {
        "name": "iris", "type": "feature", "node_label": "IrisNodeGDS",
        "file_path": os.path.join("datasets", "iris.csv"),
        "remove_columns": "Index,target", "k": 3, "id_column": "Index"
    },
    {
        "name": "madelon", "type": "feature", "node_label": "MadelonNodeGDS",
        "file_path": os.path.join("datasets", "madelon.csv"),
        "remove_columns": "Index,target", "k": 2, "id_column": "Index"
    },
    {
        "name": "20newsgroups", "type": "feature", "node_label": "NewsGroupNodeGDS",
        "file_path": os.path.join("datasets", "20newsgroups.csv"),
        "remove_columns": "Index,target", "k": 3, "id_column": "Index"
    },
    {
        "name": "cora", "type": "graph", "node_label": "CoraNodeGDS",
        "node_file_path": os.path.join("datasets", "cora_nodes.csv"),
        "remove_columns": "id,label", "k": 7, "id_column": "id", "features_column": "features"
    },
    {
        "name": "citeseer", "type": "graph", "node_label": "CiteSeerNodeGDS",
        "node_file_path": os.path.join("datasets", "citeseer_nodes.csv"),
        "remove_columns": "id,label", "k": 6, "id_column": "id", "features_column": "features"
    },
    {
        "name": "pubmed", "type": "graph", "node_label": "PubMedNodeGDS",
        "node_file_path": os.path.join("datasets", "pubmed_nodes.csv"),
        "remove_columns": "id,label", "k": 3, "id_column": "id", "features_column": "features"
    }
]

def clear_neo4j_data(driver, label):
    print(f"[INFO] Clearing out any existing :{label} nodes…")
    with driver.session() as sess:
        sess.run(f"MATCH (n:{label}) DETACH DELETE n")
    print(f"[INFO] ✓ Cleared :{label} nodes.")

def ingest_nodes_with_features(driver, config):
    node_label = config["node_label"]
    id_column = config["id_column"]
    rows_to_ingest = []

    if config["type"] == "feature":
        file_path = config["file_path"]
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return False
        df = pd.read_csv(file_path)
        
        all_cols = df.columns.tolist()
        cols_to_remove_set = set([col.strip() for col in config["remove_columns"].split(',')])
        feature_columns = [col for col in all_cols if col not in cols_to_remove_set]

        for _, row in df.iterrows():
            node_id = row[id_column]
            features_raw = [row[fc] for fc in feature_columns]
            try:
                features = [float(f) for f in features_raw]
            except ValueError as e:
                print(f"[WARNING] Could not convert features to float for node {node_id} in {config['name']}: {features_raw}. Error: {e}. Skipping node.")
                continue
            rows_to_ingest.append({"csv_id": node_id, "features": features})

    elif config["type"] == "graph":
        node_file_path = config["node_file_path"]
        if not os.path.exists(node_file_path):
            print(f"[ERROR] File not found: {node_file_path}")
            return False
        features_column = config["features_column"]
        df = pd.read_csv(node_file_path)
        
        for _, row in df.iterrows():
            node_id = row[id_column]
            feat_val_str = row[features_column]
            try:
                # Ensure features are parsed as list of numbers (floats)
                features_raw = eval(feat_val_str) if isinstance(feat_val_str, str) else feat_val_str
                if not isinstance(features_raw, list): raise TypeError("Parsed features are not a list.")
                features = [float(f) for f in features_raw]
            except Exception as e:
                print(f"[WARNING] Could not parse/convert features for node {node_id} in {config['name']}: '{feat_val_str}'. Error: {e}. Skipping node.")
                continue
            rows_to_ingest.append({"csv_id": node_id, "features": features})
    else:
        raise ValueError(f"Unknown dataset type: {config['type']}")

    if not rows_to_ingest:
        print(f"[WARNING] No rows to ingest for {config['name']}.")
        return False

    print(f"[INFO] Ingesting {len(rows_to_ingest)} nodes for {config['name']} with label {node_label}...")
    
    # Batch ingestion
    batch_size = 10000  # Limit batch size to avoid memory issues
    for i in range(0, len(rows_to_ingest), batch_size):
        batch = rows_to_ingest[i:i + batch_size]
        with driver.session() as sess:
            sess.run(f"""
                UNWIND $rows AS r
                CREATE (n:{node_label} {{csv_id: r.csv_id, features: r.features}})
            """, rows=batch)

    print(f"[INFO] ✓ Nodes ingested for {config['name']}.")
    return True

def project_graph_gds(gds, graph_name, node_label, neo4j_driver): # Added neo4j_driver for debug
    print(f"[INFO] Checking for existing GDS graph ‘{graph_name}’…")
    if gds.graph.exists(graph_name)["exists"]:
        print(f"[INFO] Dropping old GDS graph ‘{graph_name}’…")
        gds.graph.drop(graph_name)

    print(f"[INFO] Projecting new GDS graph ‘{graph_name}’…")
    node_query = f"MATCH (n:{node_label}) RETURN id(n) AS id, n.features AS features"
    # Self-loops for KMeans, as relationship structure isn't used by this algorithm directly
    relationship_query = f"MATCH (n:{node_label}) RETURN id(n) AS source, id(n) AS target"
    
    try:
        gds_call_result = gds.graph.project.cypher(
            graph_name,
            node_query,
            relationship_query
        )
        
        # Assuming gds_call_result might be a tuple containing the Series as its first element
        proj_summary_series = gds_call_result
        if isinstance(gds_call_result, tuple) and len(gds_call_result) > 0:
            proj_summary_series = gds_call_result[0]
            print(f"[DEBUG] gds.graph.project.cypher returned a tuple, using its first element. Type: {type(proj_summary_series)}")
        elif not hasattr(proj_summary_series, 'get') and not isinstance(proj_summary_series, pd.Series):
             # Fallback if it's not a series or a tuple containing a series, print type and object for diagnosis
            print(f"[DEBUG] proj_summary_series is of unexpected type: {type(proj_summary_series)}")
            print(f"[DEBUG] proj_summary_series content: {proj_summary_series}")
        
        print("[INFO] GDS Projection summary:")
        # Access fields using dictionary-style keys on the (potentially unwrapped) Series
        print(f"  Graph Name: {proj_summary_series['graphName']}")
        print(f"  Node Count: {proj_summary_series['nodeCount']}")
        print(f"  Relationship Count: {proj_summary_series['relationshipCount']}")
        print(f"  Project Millis: {proj_summary_series['projectMillis']}")
        return gds.graph.get(graph_name)
    except Exception as e:
        print(f"[ERROR] Failed to project GDS graph '{graph_name}': {e}")
        # Attempt to get Neo4j node count for debugging
        try:
            with neo4j_driver.session() as debug_session: # Use neo4j_driver
                count_result = debug_session.run(f"MATCH (n:{node_label}) RETURN count(n) AS count").single()
                print(f"[DEBUG] Neo4j node count for label {node_label}: {count_result['count'] if count_result else 'N/A'}")
        except Exception as debug_e:
            print(f"[DEBUG] Error fetching Neo4j node count: {debug_e}")
        raise

def run_gds_kmeans_and_time(gds, G_object, k_clusters, config_name, node_label, neo4j_driver): # Added neo4j_driver for debug
    print(f"[INFO] Running GDS K-means (k={k_clusters}) for {config_name} on graph '{G_object.name()}'…")
    start_time = time.time()
    
    try:
        kmeans_summary_df = gds.kmeans.write(
            G_object,
            nodeProperty='features',
            writeProperty='clusterIdGDS', # Property to write back to Neo4j nodes
            k=k_clusters,
            maxIterations=25, 
            randomSeed=42 # For reproducibility
        )
    except Exception as e:
        print(f"[ERROR] GDS KMeans failed for {config_name} on graph '{G_object.name()}': {e}")
        print(f"[DEBUG] Graph details: Name: {G_object.name()}, Nodes: {G_object.node_count()}, Properties: {G_object.node_properties()}")
        # Check sample node features from Neo4j
        sample_nodes_query = f"MATCH (n:{node_label}) RETURN n.features AS features, n.csv_id AS csv_id LIMIT 5"
        try:
            with neo4j_driver.session() as sess: # Use neo4j_driver
                res = sess.run(sample_nodes_query)
                print("[DEBUG] Sample node features from Neo4j:")
                for rec in res:
                    features_sample = rec['features']
                    print(f"  CSV_ID: {rec['csv_id']}, Features: {str(features_sample)[:100]}... (Type: {type(features_sample)}, Element type: {type(features_sample[0]) if features_sample else 'N/A'})")
        except Exception as db_e:
            print(f"[ERROR] Could not fetch sample nodes for debugging: {db_e}")
        raise

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"[INFO] GDS K-means summary for {config_name}:")
    # Relevant fields: nodeCount, ranIterations, didConverge, processingMillis, computeMillis (from GDS 1.8+)
    # For older GDS, computeMillis might not exist, use processingMillis
    compute_millis_val = kmeans_summary_df["computeMillis"].iloc[0] if "computeMillis" in kmeans_summary_df else kmeans_summary_df["processingMillis"].iloc[0]

    print(kmeans_summary_df[['nodeCount', 'ranIterations', 'didConverge', 'processingMillis']])
    if "computeMillis" in kmeans_summary_df:
        print(f"  computeMillis: {kmeans_summary_df['computeMillis'].iloc[0]}")
        
    return {
        "dataset_name": config_name,
        "k": k_clusters,
        "gds_kmeans_processing_millis": kmeans_summary_df["processingMillis"].iloc[0],
        "gds_kmeans_compute_millis": compute_millis_val,
        "wall_time_seconds": duration,
        "iterations": kmeans_summary_df["ranIterations"].iloc[0],
        "converged": kmeans_summary_df["didConverge"].iloc[0]
    }

if __name__ == "__main__":
    all_results = []

    print("[INFO] ▶ Connecting to Neo4j and GDS…")
    try:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        # For GDS client, ensure aura_ds is False if not using AuraDS
        gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), aura_ds=False) 
        print(f"[INFO]  • Neo4j Server info: {neo4j_driver.get_server_info()}")
        print(f"[INFO]  • GDS version      : {gds.version()}")
        print("[INFO] ✓ Connected.")
    except Exception as e:
        print(f"[FATAL] Could not connect to Neo4j/GDS: {e}")
        exit(1)

    for ds_config in DATASETS_CONFIG:
        dataset_name = ds_config["name"]
        node_label = ds_config["node_label"]
        k_clusters = ds_config["k"]
        gds_graph_name = f"{dataset_name.replace(' ', '_')}_gds_graph_kmeans" # Ensure valid graph name

        print(f"\n--- Processing dataset: {dataset_name} ---")
        
        try:
            clear_neo4j_data(neo4j_driver, node_label)
            ingestion_success = ingest_nodes_with_features(neo4j_driver, ds_config)
            if not ingestion_success:
                print(f"[ERROR] Ingestion failed for {dataset_name}. Skipping.")
                continue

            G = project_graph_gds(gds, gds_graph_name, node_label, neo4j_driver) # Pass neo4j_driver
            if not G:
                print(f"[ERROR] GDS graph projection failed for {dataset_name}. Skipping.")
                continue
            
            kmeans_result = run_gds_kmeans_and_time(gds, G, k_clusters, dataset_name, node_label, neo4j_driver) # Pass neo4j_driver
            all_results.append(kmeans_result)

            print(f"[INFO] Dropping GDS graph ‘{gds_graph_name}’ after use…")
            gds.graph.drop(G) # Drop using GDS graph object

        except Exception as e:
            print(f"[ERROR] Unhandled exception during processing of dataset {dataset_name}: {e}")
            # Attempt to drop graph if it exists
            try:
                if gds.graph.exists(gds_graph_name)["exists"]:
                    gds.graph.drop(gds_graph_name)
            except Exception as cleanup_e:
                print(f"[ERROR] Failed to cleanup GDS graph {gds_graph_name} after error: {cleanup_e}")
            continue

    if all_results:
        results_df = pd.DataFrame(all_results)
        output_filename = os.path.join("results", "gds_kmeans_batch_results.csv")
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(output_filename, index=False)
        print(f"\n[INFO] ✓ All experiments complete. Results saved to {output_filename}")
        print(results_df)
    else:
        print("\n[INFO] No results generated.")

    print("[INFO] Closing connections…")
    neo4j_driver.close()
    gds.close()
    print("[INFO] All done.")
