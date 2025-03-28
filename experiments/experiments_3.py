import os
import subprocess
import sys
import threading
import time
import psutil
import pandas as pd
from tqdm import tqdm
import math
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Ensure required packages are installed
required_packages = ["neo4j", "pandas", "psutil", "tqdm", "scikit-learn", "scipy"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from neo4j import GraphDatabase

# For scikit-learn experiments
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh

# ------------------ NEW HELPER & CLUSTERING FUNCTION ------------------

def check_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)

def spectral_clustering(dataframe, similarity_graph, laplacian, number_of_clusters, eps=None, k=None):
    # Pairwise distances
    dimension = dataframe.shape[0]
    dist_mat = squareform(pdist(dataframe))
    sample_size = len(dist_mat)
    
    # Set n based on proportional selection, but limit by log scaling for large datasets
    n = min(sample_size // 10, int(math.log(sample_size)))
    
    # Fallback values for epsilon and k
    epsilon = eps if eps else np.percentile(dist_mat, 90)
    k = k if k else int(np.sqrt(sample_size))
    
    if similarity_graph == "full":
        # calculate local sigma
        sigmas = np.zeros(dimension)
        for i in tqdm(range(len(dist_mat)), desc="Calculating sigmas"):
            sigmas[i] = sorted(dist_mat[i])[n]
        # Adjacency matrix with optimal sigma
        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building full affinity"):
            for j in range(i+1, dimension):
                d = np.exp(-1 * dist_mat[i, j]**2 / (sigmas[i] * sigmas[j]))
                adjacency_matrix[i, j] = d
                adjacency_matrix[j, i] = d

    elif similarity_graph == "eps":
        # Adjacency matrix with epsilon threshold
        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building eps affinity"):
            for j in range(i+1, dimension):
                d = 1 if dist_mat[i, j] < epsilon else 0
                adjacency_matrix[i, j] = d
                adjacency_matrix[j, i] = d

    elif similarity_graph == "knn":
        # Adjacency matrix with k-neighbours
        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building knn affinity"):
            sorted_indices = np.argsort(dist_mat[i])
            k_nearest_indices = sorted_indices[1:k+1]  # Exclude the node itself
            adjacency_matrix[i, k_nearest_indices] = 1

    else:
        # Adjacency matrix with mutual k-neighbours (for "mknn")
        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building mknn affinity"):
            sorted_indices = np.argsort(dist_mat[i])
            k_nearest_indices = sorted_indices[1:k+1]
            for neighbor in k_nearest_indices:
                neighbor_sorted_indices = np.argsort(dist_mat[neighbor])
                if i in neighbor_sorted_indices[1:k+1]:
                    adjacency_matrix[i, neighbor] = 1
                    adjacency_matrix[neighbor, i] = 1

    # Calculate degree matrix
    degrees = np.sum(adjacency_matrix, axis=1)
    degree_matrix = np.diag(degrees)

    if laplacian == "sym":
        # Normalized Symmetric Laplacian matrix
        d_inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        d_half = np.diag(d_inv_sqrt)
        laplacian_matrix_normalized = d_half @ adjacency_matrix @ d_half
    elif laplacian == "rw":
        # Normalized Random Walk Laplacian matrix
        d_inv = np.zeros_like(degrees)
        nonzero = degrees > 0
        d_inv[nonzero] = 1.0 / degrees[nonzero]
        d_inverse = np.diag(d_inv)
        laplacian_matrix_normalized = d_inverse @ adjacency_matrix
    else:
        raise ValueError("Unsupported laplacian type. Only 'sym' and 'rw' are allowed.")

    if check_symmetric(laplacian_matrix_normalized):
        e, v = np.linalg.eigh(laplacian_matrix_normalized)
    else:
        e, v = np.linalg.eig(laplacian_matrix_normalized)
        idx = np.argsort(np.real(e))
        e = np.real(e[idx])
        v = np.real(v[:, idx])
    
    # Calculate eigengap
    eigengap = np.diff(e)
    optimal_number_of_clusters = np.argmax(eigengap[:10]) + 1

    # Use k=2 for circles/moons, eigengap for others
    if number_of_clusters == "fixed2":
        current_k = 2
    elif number_of_clusters == "fixed3":
        current_k = 3
    else:
        current_k = max(optimal_number_of_clusters, 2)

    # Perform clustering on the eigenvectors
    X = v[:, -current_k:]
    clustering = KMeans(n_clusters=current_k, random_state=42, n_init=100)
    cluster_labels = clustering.fit_predict(X)

    sil_score = silhouette_score(dataframe, cluster_labels)
    return [(current_k, cluster_labels, sil_score)]

# ------------------ END NEW FUNCTION ------------------

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123412345"

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def ensure_indexes(driver, datasets):
    """Ensures indexes exist for each dataset automatically."""
    try:
        with driver.session() as session:
            existing_indexes = session.run("SHOW INDEXES")
            existing_index_names = {record["name"] for record in existing_indexes}
            for dataset, params in datasets.items():
                node_label = params.get("label") or params.get("node_label")
                index_id_name = f"{node_label}_id_index"
                index_label_name = f"{node_label}_label_index"
                if index_id_name not in existing_index_names:
                    session.run(f"CREATE INDEX {index_id_name} FOR (n:{node_label}) ON (n.id);")
                if index_label_name not in existing_index_names:
                    session.run(f"CREATE INDEX {index_label_name} FOR (n:{node_label}) ON (n.label);")
        print("✅ Indexes ensured for all datasets.")
    except Exception as e:
        print(f"⚠️ Error creating indexes: {e}")

def delete_all_nodes(driver, batch_size=1000):
    """Delete all nodes and relationships in the Neo4j database in batches."""
    try:
        with driver.session() as session:
            while True:
                result = session.run(f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(n) AS deleted_count")
                deleted_count = result.single()["deleted_count"]
                if deleted_count == 0:
                    break
    except Exception as e:
        print("Error during node deletion:", e)

def delete_all_indexes(driver, batch_size=5):
    """Delete all indexes in the Neo4j database in batches."""
    try:
        with driver.session() as session:
            while True:
                indexes = session.run("CALL db.indexes()")
                index_names = [index["name"] for index in indexes]
                if not index_names:
                    break
                for index_name in index_names[:batch_size]:
                    session.run(f"DROP INDEX {index_name}")
                if len(index_names) <= batch_size:
                    break
    except Exception as e:
        print("Error during index deletion:", e)

def create_feature_nodes(data, driver, label):
    """Create feature-based nodes in Neo4j from a CSV file."""
    try:
        with driver.session() as session:
            for _, row in data.iterrows():
                properties = { (f"feature_{key}" if str(key).isdigit() else key): value
                               for key, value in row.items() }
                query = f"CREATE (n:{label} {{" + ', '.join([f"{key}: ${key}" for key in properties.keys()]) + "})"
                session.run(query, **properties)
    except Exception as e:
        print("Error during node creation:", e)
    finally:
        pass

def create_graph_nodes(data, driver, label):
    """Create graph-based nodes in Neo4j from a CSV file."""
    try:
        node_data = pd.read_csv(data)
        with driver.session() as session:
            for _, row in node_data.iterrows():
                properties = row.to_dict()
                properties['features'] = eval(properties['features'])
                query = f"CREATE (n:{label} {{id: $id, features: $features, label: $label}})"
                session.run(query, **properties)
    except Exception as e:
        print("Error during node creation:", e)
    finally:
        pass

def create_edges(data, driver, node_label, edge_label):
    """Create undirected edges in Neo4j from a CSV file, ensuring only one edge is created between two nodes."""
    try:
        edge_data = pd.read_csv(data)
        with driver.session() as session:
            for _, row in edge_data.iterrows():
                source_id = min(row['source_id'], row['target_id'])
                target_id = max(row['source_id'], row['target_id'])
                query = f"""
                MATCH (source:{node_label} {{id: $source_id}})
                MATCH (target:{node_label} {{id: $target_id}})
                MERGE (source)-[:{edge_label} {{value: 1}}]->(target)
                """
                session.run(query, {"source_id": source_id, "target_id": target_id})
    except Exception as e:
        print("Error during edge creation:", e)

def run_query(driver, query, parameters):
    """Run a query and measure performance metrics."""
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_cpu_times = process.cpu_times()
    start_mem = process.memory_info().rss
    with driver.session() as session:
        result = session.run(query, parameters)
        record = result.single()
        data = record.data() if record else None
    end_time = time.time()
    end_cpu_times = process.cpu_times()
    end_mem = process.memory_info().rss
    duration = end_time - start_time
    cpu_used = (end_cpu_times.user + end_cpu_times.system) - (start_cpu_times.user + start_cpu_times.system)
    memory_used = (end_mem - start_mem) / (1024 ** 2)
    return data, duration, memory_used, cpu_used

def monitor_progress():
    """Continuously fetch progress updates while the main query runs."""
    local_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    while True:
        with local_driver.session() as session:
            query = "MATCH (p:Progress {id: 'current'}) RETURN p.step ORDER BY p.timestamp DESC LIMIT 1"
            result = session.run(query)
            record = result.single()
            data = record.data() if record else None
        if data:
            try:
                print(f"🔄 Current Step: {data['p.step']}", flush=True)
            except Exception as e:
                print(f"Error: {e}, Result: {data}")
        time.sleep(2)

# Start the monitoring thread
monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
monitor_thread.start()

def run_sklearn_experiment_feature(config, file_path):
    """Run scikit-learn spectral clustering on feature-based data."""
    df = pd.read_csv(file_path)
    cols_to_remove = [col.strip() for col in config["remove_columns"].split(',')]
    features = df.drop(columns=cols_to_remove, errors='ignore')
    true_labels = df[config["target_column"]].values

    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_cpu = process.cpu_times()
    start_mem = process.memory_info().rss

    # Determine eps/k parameters based on graph_type
    if config["graph_type"] == "eps":
        eps_val = float(config["parameter"])
        k_val = None
    elif config["graph_type"] in ["knn", "mknn"]:
        k_val = int(config["parameter"])
        eps_val = None
    else:
        eps_val = None
        k_val = None

    # Call the custom spectral clustering function
    clustering_result = spectral_clustering(features, config["graph_type"], config["laplacian_type"],
                                            config["number_of_eigenvectors"], eps=eps_val, k=k_val)
    current_k, cluster_labels, sil_score = clustering_result[0]

    clustering_time = time.time() - start_time

    skl_silhouette = sil_score
    skl_rand_index = adjusted_rand_score(true_labels, cluster_labels)
    total_time = clustering_time
    end_cpu = process.cpu_times()
    cpu_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
    end_mem = process.memory_info().rss
    memory_used = (end_mem - start_mem) / (1024 ** 2)
    return {
        "sklearn_silhouette_score": skl_silhouette,
        "sklearn_rand_index": skl_rand_index,
        "sklearn_total_time": total_time,
        "sklearn_affinity_time": 0,
        "sklearn_laplacian_time": 0,
        "sklearn_clustering_time": clustering_time,
        "sklearn_adjusted_rand_index_time": 0,
        "sklearn_memory_used": memory_used,
        "sklearn_cpu_used": cpu_used
    }

def run_sklearn_experiment_graph(config, node_file_path, edge_file_path):
    """Run scikit-learn spectral clustering on graph-based data."""
    nodes_df = pd.read_csv(node_file_path)
    features = nodes_df.drop(columns=[col.strip() for col in config["remove_columns"].split(',')], errors='ignore')
    true_labels = nodes_df[config["target_column"]].values

    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_cpu = process.cpu_times()
    start_mem = process.memory_info().rss

    # For graph experiments, we assume 'full' type (eps/k not needed)
    eps_val = None
    k_val = None
    clustering_result = spectral_clustering(features, config["graph_type"], config["laplacian_type"],
                                            config["number_of_eigenvectors"], eps=eps_val, k=k_val)
    current_k, cluster_labels, sil_score = clustering_result[0]

    clustering_time = time.time() - start_time

    skl_silhouette = sil_score
    skl_rand_index = adjusted_rand_score(true_labels, cluster_labels)
    total_time = clustering_time
    end_cpu = process.cpu_times()
    cpu_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
    end_mem = process.memory_info().rss
    memory_used = (end_mem - start_mem) / (1024 ** 2)
    return {
        "sklearn_silhouette_score": skl_silhouette,
        "sklearn_rand_index": skl_rand_index,
        "sklearn_total_time": total_time,
        "sklearn_affinity_time": 0,
        "sklearn_laplacian_time": 0,
        "sklearn_clustering_time": clustering_time,
        "sklearn_adjusted_rand_index_time": 0,
        "sklearn_memory_used": memory_used,
        "sklearn_cpu_used": cpu_used
    }

def run_experiments(driver, experiments):
    """Run experiments and measure performance metrics using both SimKit and scikit-learn."""
    print("Init")
    try:
        with driver.session() as session:
            session.run("""
            RETURN simkit.initSimKit('bolt://localhost:7687', 'neo4j', '123412345')
            """)
    except Exception as e:
        print(f"Error initializing SimKit: {e}")
        return
    print("Init Done")
    results = []
    total_experiments = len(experiments)
    for idx, config in enumerate(experiments, 1):
        query = """
        WITH simkit.experimental_spectralClustering({
            node_label: $node_label,
            is_feature_based: $is_feature_based,
            distance_measure: "euclidean",
            graph_type: $graph_type,
            parameter: $parameter,
            remove_columns: $remove_columns,
            laplacian_type: $laplacian_type,
            number_of_eigenvectors: $number_of_eigenvectors,
            number_of_iterations: "100",
            distance_measure_kmean: "euclidean",
            target_column: $target_column,
            use_kmean_for_silhouette: $use_kmean_for_silhouette,
            seed: 42
        }) AS result
        RETURN result.silhouette_score AS silhouette_score, 
               result.rand_index AS rand_index,
               result.total_time AS total_time,
               result.affinity_time AS affinity_time,
               result.laplacian_time AS laplacian_time,
               result.clustering_time AS clustering_time,
               result.adjusted_rand_index_time AS adjusted_rand_index_time
        """
        data, duration, memory_used, cpu_used = run_query(driver, query, config)
        silhouette_score = data['silhouette_score'] if data else None
        rand_index = data['rand_index'] if data else None
        total_time = data['total_time'] if data else duration
        affinity_time = data['affinity_time'] if data else None
        laplacian_time = data['laplacian_time'] if data else None
        clustering_time = data['clustering_time'] if data else None
        adjusted_rand_index_time = data['adjusted_rand_index_time'] if data else None
        simkit_result = {
            "silhouette_score": silhouette_score,
            "rand_index": rand_index,
            "total_time": total_time,
            "affinity_time": affinity_time,
            "laplacian_time": laplacian_time,
            "clustering_time": clustering_time,
            "adjusted_rand_index_time": adjusted_rand_index_time,
            "memory_used": memory_used,
            "cpu_used": cpu_used
        }
        
        if config.get("is_feature_based"):
            # Assume dataset filename for feature-based experiments is the lower-case dataset name (e.g., "iris.csv")
            sklearn_result = run_sklearn_experiment_feature(config, os.path.join("datasets", f"{config['node_label'].replace('Node','').lower()}.csv"))
        else:
            node_file_path = os.path.join("datasets", f"{config['node_label'].replace('Node','').lower()}_nodes.csv")
            edge_file_path = os.path.join("datasets", f"{config['node_label'].replace('Node','').lower()}_edges.csv")
            sklearn_result = run_sklearn_experiment_graph(config, node_file_path, edge_file_path)
        
        merged_result = {**config, **simkit_result, **sklearn_result}
        results.append(merged_result)
        print(f"Completed experiment {idx}/{total_experiments} with config: {config}")
    return results

def save_results(results, dataset):
    """Save experiment results to a CSV file."""
    df = pd.DataFrame(results)
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    df.to_csv(os.path.join(results_dir, f"{dataset}_results.csv"), index=False)
    print(f"Results saved to {dataset}_results.csv")

def run_feature_experiment(dataset, label, remove_columns, number_of_eigenvectors, target_column):
    """Runs feature-based experiments and ensures indexes exist."""
    delete_all_nodes(driver)
    delete_all_indexes(driver)
    ensure_indexes(driver, {dataset: {"label": label}})
    file_path = os.path.join("datasets", f"{dataset}.csv")
    data = pd.read_csv(file_path)
    create_feature_nodes(data, driver, label)
    print(f"Feature nodes created for {dataset}")
    experiments = []
    laplacian_types = ["sym", "rw"]
    graph_types = ["full", "eps", "knn", "mknn"]
    parameters = {"iris": {"full": "11", "eps": "1.111", "knn": "10", "mknn": "30"},
                  "madelon": {"full": "45", "eps": "4.669", "knn": "419", "mknn": "117"},
                  "20newsgroups": {"full": "35", "eps": "1946.74", "knn": "512", "mknn": "26"}}
    for graph_type in tqdm(graph_types, desc="Processing graph types"):
        for laplacian_type in tqdm(laplacian_types, desc=f"Processing Laplacian for {graph_type}", leave=False):
            experiments.append({
                "node_label": label,
                "is_feature_based": True,
                "graph_type": graph_type,
                "parameter": parameters[dataset][graph_type],
                "remove_columns": remove_columns,
                "laplacian_type": laplacian_type,
                "number_of_eigenvectors": number_of_eigenvectors,
                "target_column": target_column,
                "use_kmean_for_silhouette": False
            })
    results = run_experiments(driver, tqdm(experiments, desc="Running feature experiments"))
    save_results(results, dataset)
    print(f"Feature experiment completed for {dataset}")

def run_graph_experiment(dataset, node_label, edge_label, remove_columns, number_of_eigenvectors, target_column):
    """Runs graph-based experiments and ensures indexes exist."""
    delete_all_nodes(driver)
    delete_all_indexes(driver)
    ensure_indexes(driver, {dataset: {"node_label": node_label}})
    node_file_path = os.path.join("datasets", f"{dataset}_nodes.csv")
    edge_file_path = os.path.join("datasets", f"{dataset}_edges.csv")
    create_graph_nodes(node_file_path, driver, node_label)
    create_edges(edge_file_path, driver, node_label, edge_label)
    print(f"Graph nodes and edges created for {dataset}")
    experiments = []
    laplacian_types = ["sym", "rw"]
    for laplacian_type in tqdm(laplacian_types, desc="Processing Laplacian types"):
        experiments.append({
            "node_label": node_label,
            "is_feature_based": False,
            "graph_type": "full",
            "parameter": "3",
            "remove_columns": remove_columns,
            "laplacian_type": laplacian_type,
            "number_of_eigenvectors": number_of_eigenvectors,
            "target_column": target_column,
            "use_kmean_for_silhouette": True
        })
    results = run_experiments(driver, tqdm(experiments, desc="Running graph experiments"))
    save_results(results, dataset)
    print(f"Graph experiment completed for {dataset}")

# Define datasets and their parameters
feature_datasets = {
    "iris": {"label": "IrisNode", "remove_columns": "Index,target", "number_of_eigenvectors": 3, "target_column": "target"},
    "madelon": {"label": "MadelonNode", "remove_columns": "Index,target", "number_of_eigenvectors": 2, "target_column": "target"},
    "20newsgroups": {"label": "NewsGroupNode", "remove_columns": "Index,target", "number_of_eigenvectors": 3, "target_column": "target"}
}
graph_datasets = {
    "cora": {"node_label": "CoraNode", "edge_label": "CoraEdge", "remove_columns": "id,label", "number_of_eigenvectors": 7, "target_column": "label"},
    "pubmed": {"node_label": "PubMedNode", "edge_label": "PubMedEdge", "remove_columns": "id,label", "number_of_eigenvectors": 3, "target_column": "label"},
    "citeseer": {"node_label": "CiteSeerNode", "edge_label": "CiteSeerEdge", "remove_columns": "id,label", "number_of_eigenvectors": 6, "target_column": "label"}
}

if __name__ == "__main__":
    try:
        # Run experiments for feature-based datasets
        for dataset, params in feature_datasets.items():
            run_feature_experiment(dataset, **params)
        # Run experiments for graph-based datasets
        for dataset, params in graph_datasets.items():
            run_graph_experiment(dataset, **params)
    finally:
        driver.close()

    # ------------------ Analysis Section ------------------
    # This section reads all CSV result files from the "results" folder, computes average metrics,
    # and generates boxplots and scalability plots as specified.
    import glob
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read all CSV files in the results folder and combine into one DataFrame
    result_files = glob.glob(os.path.join("results", "*_results.csv"))
    df_list = []
    for file in result_files:
        temp = pd.read_csv(file)
        # Extract dataset name from filename (assumes filename like "iris_results.csv")
        dataset_name = os.path.basename(file).split("_results.csv")[0]
        temp['dataset'] = dataset_name.capitalize()
        df_list.append(temp)
    combined_df = pd.concat(df_list, ignore_index=True)

    # (a) Average metrics for SimKit and scikit-learn
    simkit_avg = combined_df[['total_time', 'cpu_used', 'silhouette_score', 'rand_index']].mean()
    sklearn_avg = combined_df[['sklearn_total_time', 'sklearn_cpu_used', 'sklearn_silhouette_score', 'sklearn_rand_index']].mean()
    print("\nAverage Metrics for SimKit:")
    print(simkit_avg)
    print("\nAverage Metrics for scikit-learn:")
    print(sklearn_avg)

    # (b) Boxplots for affinity_time, laplacian_time, clustering_time separately for SimKit and scikit-learn
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=combined_df[['affinity_time', 'laplacian_time', 'clustering_time']])
    plt.title("SimKit Times")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    sns.boxplot(data=combined_df[['sklearn_affinity_time', 'sklearn_laplacian_time', 'sklearn_clustering_time']])
    plt.title("scikit-learn Times")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig("boxplot_separate.pdf", bbox_inches="tight")
    plt.show()

    # (c) Combined boxplot for total time comparison between SimKit and scikit-learn
    combined_box = pd.DataFrame({
        'SimKit Total Time': combined_df['total_time'],
        'scikit-learn Total Time': combined_df['sklearn_total_time']
    })
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=combined_box)
    plt.title("Total Time Comparison: SimKit vs. scikit-learn")
    plt.ylabel("Time (s)")
    plt.savefig("boxplot_combined.pdf", bbox_inches="tight")
    plt.show()

    # (d) Scalability plots: Regression plots of nodes vs. total_time (SimKit and scikit-learn separately)
    # Define hard-coded dataset dimensions (number of nodes)
    dataset_dims = {
        'Iris': 150,
        'Madelon': 2600,
        '20newsgroups': 3387,
        'Cora': 2708,
        'Citeseer': 3312,
        'Pubmed': 19717
    }
    # Map dataset name to number of nodes (convert to lower-case keys for matching)
    def get_nodes(dataset):
        for key in dataset_dims:
            if key.lower() == dataset.lower():
                return dataset_dims[key]
        return np.nan
    combined_df['nodes'] = combined_df['dataset'].apply(get_nodes)

    # SimKit scalability plot
    plt.figure(figsize=(6, 6))
    sns.regplot(x='nodes', y='total_time', data=combined_df, scatter_kws={'s': 50})
    plt.title("SimKit Scalability: Total Time vs. Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Total Time (s)")
    plt.savefig("scalability_simkit.pdf", bbox_inches="tight")
    plt.show()

    # scikit-learn scalability plot
    plt.figure(figsize=(6, 6))
    sns.regplot(x='nodes', y='sklearn_total_time', data=combined_df, scatter_kws={'s': 50})
    plt.title("scikit-learn Scalability: Total Time vs. Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Total Time (s)")
    plt.savefig("scalability_sklearn.pdf", bbox_inches="tight")
    plt.show()