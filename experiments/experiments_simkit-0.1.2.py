import os
import threading
import time
import math
import pandas as pd
from tqdm import tqdm
import psutil
from scipy.spatial.distance import pdist, squareform
import numpy as np
from neo4j import GraphDatabase

# For Python experiments
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import scipy.sparse as sp

def get_neo4j_usage():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline') or [])
            if ('neo4j' in cmdline.lower()) or ('org.neo4j' in cmdline.lower()):
                cpu = proc.cpu_percent(interval=1)
                mem = proc.memory_info().rss / (1024 * 1024)  # in MB
                return cpu, mem
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None, None

def check_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)


def spectral_clustering(dataframe, similarity_graph, laplacian, number_of_clusters, n=None, eps=None, k_knn=None,
                        k_mknn=None):  ## Check for inputs for all parameters
    # Pairwise distances

    dimension = dataframe.shape[0]
    dist_mat = squareform(pdist(dataframe))
    sample_size = len(dist_mat)

    # Set n based on proportional selection, but limit by log scaling for large datasets
    if (n is None):
        n = min(sample_size // 10, int(math.log(sample_size)))

    # Fallback values for epsilon and k
    epsilon = eps if eps else np.percentile(dist_mat, 90)
    k_knn = k_knn if k_knn else int(np.sqrt(sample_size))
    k_mknn = k_mknn if k_mknn else int(np.sqrt(sample_size))

    if similarity_graph == "full":
        # calculate local sigma
        sigmas = np.zeros(dimension)
        for i in tqdm(range(len(dist_mat)), desc="Calculating sigmas"):
            sigmas[i] = sorted(dist_mat[i])[n]

        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building full affinity"):
            for j in range(i + 1, dimension):
                d = np.exp(-1 * dist_mat[i, j] ** 2 / (sigmas[i] * sigmas[j]))
                adjacency_matrix[i, j] = d
                adjacency_matrix[j, i] = d

    elif similarity_graph == "eps":
        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building eps affinity"):
            for j in range(i + 1, dimension):
                d = 1 if dist_mat[i, j] < epsilon else 0
                adjacency_matrix[i, j] = d
                adjacency_matrix[j, i] = d
    elif similarity_graph == "knn":
        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building knn affinity"):
            sorted_indices = np.argsort(dist_mat[i])
            k_nearest_indices = sorted_indices[1:k_knn + 1]
            adjacency_matrix[i, k_nearest_indices] = 1
    else:
        adjacency_matrix = np.zeros([dimension, dimension])
        for i in tqdm(range(dimension), desc="Building mknn affinity"):
            sorted_indices = np.argsort(dist_mat[i])
            k_nearest_indices = sorted_indices[1:k_mknn + 1]
            for neighbor in k_nearest_indices:
                neighbor_sorted_indices = np.argsort(dist_mat[neighbor])
                if i in neighbor_sorted_indices[1:k_mknn + 1]:
                    adjacency_matrix[i, neighbor] = 1
                    adjacency_matrix[neighbor, i] = 1

    degrees = np.sum(adjacency_matrix, axis=1)
    degree_matrix = np.diag(degrees)

    if laplacian == "sym":
        d_inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        d_half = np.diag(d_inv_sqrt)
        laplacian_matrix_normalized = d_half @ adjacency_matrix @ d_half
    elif laplacian == "rw":
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

    eigengap = np.diff(e)
    optimal_number_of_clusters = np.argmax(eigengap[:10]) + 1

    if isinstance(number_of_clusters, int):
        current_k = number_of_clusters
    elif number_of_clusters == "fixed2":
        current_k = 2
    elif number_of_clusters == "fixed3":
        current_k = 3
    else:
        current_k = max(optimal_number_of_clusters, 2)

    X = v[:, -current_k:]
    clustering = KMeans(n_clusters=current_k, random_state=42, n_init=100)
    cluster_labels = clustering.fit_predict(X)

    sil_score = silhouette_score(dataframe, cluster_labels)

    print(
        f"🔍 Inside spectral_clustering -> n: {n}, epsilon: {eps}, k_knn: {k_knn}, k_mknn: {k_mknn}, clusters: {current_k}")

    return [(current_k, cluster_labels, sil_score)]


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123412345"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def ensure_indexes(driver, datasets):
    try:
        with driver.session() as session:
            existing_indexes = session.run("SHOW INDEXES")
            existing_index_names = {record["name"] for record in existing_indexes}
            for dataset, params in datasets.items():
                label = params.get("label") or params.get("label")
                index_id_name = f"{label}_id_index"
                index_label_name = f"{label}_label_index"
                if index_id_name not in existing_index_names:
                    session.run(f"CREATE INDEX {index_id_name} FOR (n:{label}) ON (n.id);")
                if index_label_name not in existing_index_names:
                    session.run(f"CREATE INDEX {index_label_name} FOR (n:{label}) ON (n.label);")
        print("✅ Indexes ensured for all datasets.")
    except Exception as e:
        print(f"⚠️ Error creating indexes: {e}")


def delete_all_nodes(driver, batch_size=1000):
    try:
        with driver.session() as session:
            while True:
                result = session.run(
                    f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(n) AS deleted_count")
                deleted_count = result.single()["deleted_count"]
                if deleted_count == 0:
                    break
    except Exception as e:
        print("Error during node deletion:", e)


def delete_all_indexes(driver, batch_size=5):
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
    try:
        with driver.session() as session:
            for _, row in data.iterrows():
                properties = {(f"feature_{key}" if str(key).isdigit() else key): value for key, value in row.items()}
                query = f"CREATE (n:{label} {{" + ', '.join([f"{key}: ${key}" for key in properties.keys()]) + "})"
                session.run(query, **properties)
    except Exception as e:
        print("Error during node creation:", e)
    finally:
        pass


def create_graph_nodes(data, driver, label):
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


def create_edges(data, driver, label, edge_label):
    try:
        edge_data = pd.read_csv(data)
        with driver.session() as session:
            for _, row in edge_data.iterrows():
                source_id = min(row['source_id'], row['target_id'])
                target_id = max(row['source_id'], row['target_id'])
                query = f"MATCH (source:{label} {{id: $source_id}}) MATCH (target:{label} {{id: $target_id}}) MERGE (source)-[:{edge_label} {{value: 1}}]->(target)"
                session.run(query, {"source_id": source_id, "target_id": target_id})
    except Exception as e:
        print("Error during edge creation:", e)


def run_query(driver, query, parameters):
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
    memory_used = abs(end_mem - start_mem) / (1024 ** 2)
    neo4j_cpu, neo4j_mem = get_neo4j_usage()
    return data, duration, memory_used, cpu_used, neo4j_mem, neo4j_cpu


def monitor_progress():
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


monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
monitor_thread.start()


def run_python_experiment_feature(config, file_path):
    # Run Python spectral clustering on feature-based data

    df = pd.read_csv(file_path)
    cols_to_remove = [col.strip() for col in config["remove_columns"].split(',')]
    features = df.drop(columns=cols_to_remove, errors='ignore')
    true_labels = df[config["target_column"]].values

    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_cpu = process.cpu_times()
    start_mem = process.memory_info().rss

    graph_type = config["graph_type"]
    parameter = float(config["parameter"])

    # Initialize and consider all possible parameters
    n_val = None
    eps_val = None
    knn_val = None
    mknn_val = None

    if graph_type == "full":
        n_val = int(parameter)
    elif graph_type == "eps":
        eps_val = parameter
    elif graph_type == "knn":
        knn_val = int(parameter)
    elif graph_type == "mknn":
        mknn_val = int(parameter)

    '''
    if config["graph_type"] == "eps":
        eps_val = float(config["parameter"])
        k_val = None
    elif config["graph_type"] in ["knn", "mknn"]:
        k_val = int(config["parameter"])
        eps_val = None
    else:
        eps_val = None
        k_val = None
    '''

    affinity_start = time.time()
    dist_mat = squareform(pdist(features))
    affinity_end = time.time()
    affinity_time = affinity_end - affinity_start

    laplacian_start = time.time()
    clustering_result = spectral_clustering(features, config["graph_type"], config["laplacian_type"],
                                            config["number_of_eigenvectors"], n=n_val, eps=eps_val, k_knn=knn_val,
                                            k_mknn=mknn_val)
    laplacian_end = time.time()
    laplacian_time = laplacian_end - laplacian_start

    current_k, cluster_labels, sil_score_val = clustering_result[0]
    clustering_end = time.time()
    clustering_time = clustering_end - laplacian_end
    ari_start = time.time()
    python_rand_index = adjusted_rand_score(true_labels, cluster_labels)
    ari_end = time.time()
    ari_time = ari_end - ari_start

    python_silhouette = sil_score_val
    total_time = ari_end - start_time
    end_cpu = process.cpu_times()
    cpu_used = (end_cpu.user + end_cpu.system) - (start_cpu.user + start_cpu.system)
    end_mem = process.memory_info().rss
    memory_used = (end_mem - start_mem) / (1024 ** 2)
    return {
        "python_silhouette_score": python_silhouette,

        "python_rand_index": python_rand_index,
        "python_total_time": total_time,
        "python_affinity_time": affinity_time,
        "python_laplacian_time": laplacian_time,
        "python_clustering_time": clustering_time,
        "python_adjusted_rand_index_time": ari_time,
        "python_memory_used": memory_used,
        "python_cpu_used": cpu_used

    }


def run_python_experiment_graph(config, node_file_path, edge_file_path):
    # Load node data
    nodes_df = pd.read_csv(node_file_path)
    true_labels = nodes_df[config["target_column"]].values
    features = nodes_df.drop(columns=[col.strip() for col in config["remove_columns"].split(',')], errors='ignore')

    if "features" in features.columns:
        features = np.array(features["features"].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist())
    else:
        features = features.values.astype(float)

    # Load edge data and create adjacency matrix
    edge_df = pd.read_csv(edge_file_path)
    node_ids = nodes_df["id"].tolist()
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    dim = len(node_ids)
    adjacency_matrix = np.zeros((dim, dim))

    for _, row in edge_df.iterrows():
        src = id_to_index.get(row["source_id"])
        tgt = id_to_index.get(row["target_id"])
        if src is not None and tgt is not None:
            adjacency_matrix[src, tgt] = 1
            adjacency_matrix[tgt, src] = 1  # assuming undirected graph

    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_cpu = process.cpu_times()
    start_mem = process.memory_info().rss

    # Compute Laplacian
    laplacian_start = time.time()
    degrees = np.sum(adjacency_matrix, axis=1)
    degree_matrix = np.diag(degrees)

    if config["laplacian_type"] == "sym":
        d_inv_sqrt = np.zeros_like(degrees, dtype=float)
        nonzero = degrees > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        d_half = sp.diags(d_inv_sqrt)
        laplacian_matrix_normalized = d_half @ adjacency_matrix @ d_half
    elif config["laplacian_type"] == "rw":
        d_inv = np.zeros_like(degrees, dtype=float)
        nonzero = degrees > 0
        d_inv[nonzero] = 1.0 / degrees[nonzero]
        d_inverse = sp.diags(d_inv)
        laplacian_matrix_normalized = d_inverse @ adjacency_matrix
    else:
        raise ValueError("Unsupported laplacian type. Only 'sym' and 'rw' are allowed.")

    # Eigen-decomposition and clustering
    if check_symmetric(laplacian_matrix_normalized):
        e, v = np.linalg.eigh(laplacian_matrix_normalized)
    else:
        e, v = np.linalg.eig(laplacian_matrix_normalized)
        idx = np.argsort(np.real(e))
        e = np.real(e[idx])
        v = np.real(v[:, idx])

    eigengap = np.diff(e)
    optimal_number_of_clusters = np.argmax(eigengap[:10]) + 1

    number_of_clusters = config["number_of_eigenvectors"]
    if isinstance(number_of_clusters, int):
        current_k = number_of_clusters
    elif number_of_clusters == "fixed2":
        current_k = 2
    elif number_of_clusters == "fixed3":
        current_k = 3
    else:
        current_k = max(optimal_number_of_clusters, 2)

    X = v[:, -current_k:]  # skip the first trivial eigenvector - ignore
    clustering = KMeans(n_clusters=current_k, random_state=42, n_init=100)
    cluster_labels = clustering.fit_predict(X)
    python_silhouette = silhouette_score(X, cluster_labels)
    laplacian_end = time.time()
    clustering_end = time.time()

    print(f"🔍 Inside spectral_clustering -> clusters: {current_k}")

    # Evaluation
    ari_start = time.time()
    python_rand_index = adjusted_rand_score(true_labels, cluster_labels)
    ari_end = time.time()

    # python_silhouette = silhouette_score(adjacency_matrix, cluster_labels, metric='precomputed')
    total_time = ari_end - start_time
    cpu_used = (process.cpu_times().user + process.cpu_times().system) - (start_cpu.user + start_cpu.system)
    memory_used = (process.memory_info().rss - start_mem) / (1024 ** 2)

    return {
        "python_silhouette_score": python_silhouette,

        "python_rand_index": python_rand_index,
        "python_total_time": total_time,
        "python_affinity_time": 0.0,  # not used now
        "python_laplacian_time": laplacian_end - laplacian_start,
        "python_clustering_time": clustering_end - laplacian_end,
        "python_adjusted_rand_index_time": ari_end - ari_start,
        "python_memory_used": memory_used,
        "python_cpu_used": cpu_used

    }


def run_experiments(driver, experiments):
    print("Init")
    try:
        with driver.session() as session:
            session.run(f"RETURN simkit.initSimKit('{NEO4J_URI}', '{NEO4J_USER}', '{NEO4J_PASSWORD}')")
    except Exception as e:
        print(f"Error initializing SimKit: {e}")
        return
    print("Init Done")
    results = []
    total_experiments = len(experiments)
    for idx, config in enumerate(experiments, 1):
        print(config)
        query = "WITH simkit.experimentalSpectralClustering({ label: $label, duplicate_node_set: True ,is_feature_based: $is_feature_based, distance_measure: 'euclidean', graph_type: $graph_type, parameter: $parameter, remove_columns: $remove_columns, laplacian_type: $laplacian_type, number_of_eigenvectors: $number_of_eigenvectors, number_of_iterations: 100, distance_measure_kmean: 'euclidean', target_column: $target_column, silhouette: $use_kmean_for_silhouette, seed: 42 }) AS result RETURN result.silhouette_score AS silhouette_score, result.rand_index AS rand_index, result.total_time AS total_time, result.affinity_time AS affinity_time, result.laplacian_time AS laplacian_time, result.clustering_time AS clustering_time, result.adjusted_rand_index_time AS adjusted_rand_index_time"
        data, duration, memory_used, cpu_used, neo4j_mem, neo4j_cpu = run_query(driver, query, config)
        silhouette_val = data['silhouette_score'] if data else None
        rand_index_val = data['rand_index'] if data else None
        total_time = data['total_time'] if data else duration
        affinity_time = data['affinity_time'] if data else None
        laplacian_time = data['laplacian_time'] if data else None
        clustering_time = data['clustering_time'] if data else None
        adjusted_rand_index_time = data['adjusted_rand_index_time'] if data else None
        simkit_result = {
            "silhouette_score": silhouette_val,
            "rand_index": rand_index_val,
            "total_time": total_time,
            "affinity_time": affinity_time,
            "laplacian_time": laplacian_time,
            "clustering_time": clustering_time,
            "adjusted_rand_index_time": adjusted_rand_index_time,
            "memory_used": memory_used,
            "cpu_used": cpu_used,
            "neo4j_memory_used": neo4j_mem,
            "neo4j_cpu_used": neo4j_cpu
        }
        if config.get("is_feature_based"):
            filename = f"{config['label'].replace('Node', '').lower()}.csv"
            if "news" in filename:
                filename = "20newsgroups.csv"
            python_result = run_python_experiment_feature(config, os.path.join("datasets", filename))
        else:
            node_file_path = os.path.join("datasets", f"{config['label'].replace('Node', '').lower()}_nodes.csv")
            edge_file_path = os.path.join("datasets", f"{config['label'].replace('Node', '').lower()}_edges.csv")
            python_result = run_python_experiment_graph(config, node_file_path, edge_file_path)
        merged_result = {**config, **simkit_result, **python_result}
        results.append(merged_result)
        print(f"Completed experiment {idx}/{total_experiments} with config: {config}")
        print(simkit_result)
    return results


def save_results(results, dataset):
    df = pd.DataFrame(results)
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    df.to_csv(os.path.join(results_dir, f"{dataset}_results.csv"), index=False)
    print(f"Results saved to {dataset}_results.csv")


def run_feature_experiment(dataset, label, remove_columns, number_of_eigenvectors, target_column):
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
    parameters = {
        "iris": {"full": "11", "eps": "1.111", "knn": "10", "mknn": "30"},
        "madelon": {"full": "45", "eps": "4.669", "knn": "419", "mknn": "117"},
        "20newsgroups": {"full": "35", "eps": "1946.74", "knn": "512", "mknn": "26"}
    }
    for graph_type in tqdm(graph_types, desc="Processing graph types"):
        for laplacian_type in tqdm(laplacian_types, desc=f"Processing Laplacian for {graph_type}", leave=False):
            experiments.append({
                "label": label,
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


def run_graph_experiment(dataset, label, edge_label, remove_columns, number_of_eigenvectors, target_column):
    delete_all_nodes(driver)
    delete_all_indexes(driver)
    ensure_indexes(driver, {dataset: {"label": label}})
    node_file_path = os.path.join("datasets", f"{dataset}_nodes.csv")
    edge_file_path = os.path.join("datasets", f"{dataset}_edges.csv")
    create_graph_nodes(node_file_path, driver, label)
    create_edges(edge_file_path, driver, label, edge_label)
    print(f"Graph nodes and edges created for {dataset}")
    experiments = []
    laplacian_types = ["sym", "rw"]
    for laplacian_type in tqdm(laplacian_types, desc="Processing Laplacian types"):
        experiments.append({
            "label": label,
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


feature_datasets = {
    "iris": {"label": "IrisNode", "remove_columns": "Index,target", "number_of_eigenvectors": 3, "target_column": "target"},
    "madelon": {"label": "MadelonNode", "remove_columns": "Index,target", "number_of_eigenvectors": 2, "target_column": "target"},
    "20newsgroups": {"label": "NewsGroupNode", "remove_columns": "Index,target", "number_of_eigenvectors": 3, "target_column": "target"}
}

graph_datasets = {
    "cora": {"label": "CoraNode", "edge_label": "CoraEdge", "remove_columns": "id,label", "number_of_eigenvectors": 7, "target_column": "label"},
    "pubmed": {"label": "PubMedNode", "edge_label": "PubMedEdge", "remove_columns": "id,label", "number_of_eigenvectors": 3, "target_column": "label"},
    "citeseer": {"label": "CiteSeerNode", "edge_label": "CiteSeerEdge", "remove_columns": "id,label", "number_of_eigenvectors": 6, "target_column": "label"}
}

try:
    for dataset, params in feature_datasets.items():
        run_feature_experiment(dataset, **params)
    for dataset, params in graph_datasets.items():
        run_graph_experiment(dataset, **params)
finally:
    driver.close()

import glob
import matplotlib.pyplot as plt
import seaborn as sns

result_files = glob.glob(os.path.join("results", "*_results.csv"))
df_list = []
for file in result_files:
    temp = pd.read_csv(file)
    dataset_name = os.path.basename(file).split("_results.csv")[0]
    temp['dataset'] = dataset_name.capitalize()
    df_list.append(temp)
combined_df = pd.concat(df_list, ignore_index=True)

simkit_avg = combined_df[['total_time', 'cpu_used', 'silhouette_score', 'rand_index']].mean()
python_avg = combined_df[
    ['python_total_time', 'python_cpu_used', 'python_silhouette_score', 'python_rand_index']].mean()
print("\nAverage Metrics for SimKit:")
print(simkit_avg)
print("\nAverage Metrics for Python:")
print(python_avg)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=combined_df[['affinity_time', 'laplacian_time', 'clustering_time']])
plt.title("SimKit Times")
plt.ylabel("Time (s)")

plt.subplot(1, 2, 2)
sns.boxplot(data=combined_df[['python_affinity_time', 'python_laplacian_time', 'python_clustering_time']])
plt.title("Python Times")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.savefig("boxplot_separate.pdf", bbox_inches="tight")
plt.show()

combined_box = pd.DataFrame({
    'SimKit Total Time': combined_df['total_time'],
    'Python Total Time': combined_df['python_total_time']
})
plt.figure(figsize=(6, 6))
sns.boxplot(data=combined_box)
plt.title("Total Time Comparison: SimKit vs. Python")
plt.ylabel("Time (s)")
plt.savefig("boxplot_combined.pdf", bbox_inches="tight")
plt.show()

dataset_dims = {
    'Iris': 150,
    'Madelon': 2600,
    '20newsgroups': 3387,
    'Cora': 2708,
    'Citeseer': 3312,
    'Pubmed': 19717
}


def get_nodes(dataset):
    for key in dataset_dims:
        if key.lower() == dataset.lower():
            return dataset_dims[key]
    return np.nan


combined_df['nodes'] = combined_df['dataset'].apply(get_nodes)

plt.figure(figsize=(6, 6))
sns.regplot(x='nodes', y='total_time', data=combined_df, scatter_kws={'s': 50})
plt.title("SimKit Scalability: Total Time vs. Nodes")
plt.xlabel("Number of Nodes")
plt.ylabel("Total Time (s)")
plt.savefig("scalability_simkit.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 6))
sns.regplot(x='nodes', y='python_total_time', data=combined_df, scatter_kws={'s': 50})
plt.title("Python Scalability: Total Time vs. Nodes")
plt.xlabel("Number of Nodes")
plt.ylabel("Total Time (s)")
plt.savefig("scalability_python.pdf", bbox_inches="tight")
plt.show()
