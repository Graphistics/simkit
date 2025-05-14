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

# For Python experiments
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

def check_symmetric(matrix, tol=1e-8):
    """
    Checks if a matrix is symmetric. Handles both dense NumPy arrays and SciPy sparse matrices.

    Args:
        matrix (np.ndarray or sp.spmatrix): The matrix to check.
        tol (float): Tolerance for numerical comparison (mainly for dense matrices).

    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    if sp.issparse(matrix):
        # For sparse matrices, the most reliable check for symmetry needed by eigsh
        # is that (matrix - matrix.T) has no non-zero elements.
        # This checks for A_ij == A_ji for all i, j.
        diff = matrix - matrix.T
        return diff.nnz == 0
    else:
        # For dense matrices, use np.allclose
        return

def spectral_clustering(dataframe, similarity_graph, laplacian, number_of_clusters, n=None, eps=None, k_knn=None, k_mknn=None):
    """
    Performs spectral clustering and returns labels, score, and timings.

    Returns:
        tuple: Contains (cluster_labels, sil_score, affinity_time,
               laplacian_time, clustering_time_combined, silhouette_time)
    """
    print(f"🐍 Starting Python Spectral Clustering (inside function): graph={similarity_graph}, laplacian={laplacian}, k_param={number_of_clusters}")

    # --- Affinity Matrix Calculation ---
    start_affinity = time.time()
    dimension = dataframe.shape[0]

    # Handle empty input DataFrame gracefully
    if dimension == 0:
        print("⚠️ Empty dataframe provided to spectral_clustering.")
        return np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0 # Return empty/zero values

    # Calculate pairwise distances
    dist_mat = squareform(pdist(dataframe))
    sample_size = len(dist_mat)

    # --- Parameter Setup ---
    # Use provided parameters or calculate defaults
    # Default n for 'full' graph (local sigma tuning)
    if n is None and similarity_graph == "full":
        if sample_size <= 1:
            n = 0 # Cannot compute neighbors
        else:
            # Heuristic: Use a value related to log(N) or sqrt(N), capped
            log_val = math.log(sample_size) if sample_size > 1 else 0
            # Ensure n is at least 1 and not more than available neighbors
            n = min(max(7, int(log_val)), 15) # Example heuristic, adjust as needed
            n = min(n, dimension - 1) if dimension > 1 else 0
            print(f"   Auto-calculated n for 'full' graph: {n}")
    elif n is not None:
         n = int(n) # Ensure n is int if provided


    # Default epsilon for 'eps' graph
    if eps is None and similarity_graph == "eps":
        non_zero_distances = dist_mat[dist_mat > 1e-9]
        if non_zero_distances.size > 0:
            eps = np.percentile(non_zero_distances, 10) # e.g., 10th percentile
        else:
            eps = 0.1 # Small default if all points are coincident
        print(f"   Auto-calculated eps for 'eps' graph: {eps:.4f}")
    elif eps is not None:
        eps = float(eps) # Ensure eps is float

    # Default k for 'knn'/'mknn' graphs
    default_k = 1
    if sample_size > 1:
         default_k = max(1, int(np.sqrt(sample_size)))

    if k_knn is None and similarity_graph == "knn":
        k_knn = default_k
        print(f"   Auto-calculated k_knn for 'knn' graph: {k_knn}")
    elif k_knn is not None:
         k_knn = int(k_knn)

    if k_mknn is None and similarity_graph == "mknn":
        k_mknn = default_k
        print(f"   Auto-calculated k_mknn for 'mknn' graph: {k_mknn}")
    elif k_mknn is not None:
        k_mknn = int(k_mknn)

    # --- Build Adjacency Matrix ---
    # (Using original logic, ensure tqdm is imported)
    if similarity_graph == "full":
        sigmas = np.zeros(dimension)
        # Ensure n is valid before using it as index
        kth_neighbor_index = min(n, dimension - 1) if n is not None and n > 0 else 1
        if kth_neighbor_index <= 0 and dimension > 1: kth_neighbor_index = 1 # Need at least 1 neighbor

        for i in range(dimension):
             if dimension > 1:
                 sorted_dist_i = np.sort(dist_mat[i])
                 sigmas[i] = sorted_dist_i[kth_neighbor_index]
             else:
                 sigmas[i] = 1.0 # Handle single point case

        sigmas = np.maximum(sigmas, 1e-9) # Prevent division by zero
        adjacency_matrix = np.zeros([dimension, dimension])
        # Can be slow for large D, consider optimizations if needed
        for i in tqdm(range(dimension), desc="Building full affinity", leave=False, ncols=80):
            for j in range(i + 1, dimension):
                # Avoid division by zero or invalid ops if sigmas are near zero
                if sigmas[i] * sigmas[j] > 1e-12:
                     d = np.exp(-(dist_mat[i, j]**2) / (sigmas[i] * sigmas[j]))
                     adjacency_matrix[i, j] = d
                     adjacency_matrix[j, i] = d
                # else: adjacency remains 0

    elif similarity_graph == "eps":
        adjacency_matrix = (dist_mat < eps).astype(float)
        np.fill_diagonal(adjacency_matrix, 0)

    elif similarity_graph == "knn":
        adjacency_matrix = np.zeros([dimension, dimension])
        k_to_use = min(k_knn, dimension - 1) if k_knn is not None else 0
        if k_to_use > 0:
            for i in tqdm(range(dimension), desc="Building knn affinity", leave=False, ncols=80):
                nearest_indices = np.argsort(dist_mat[i])[1:k_to_use + 1]
                adjacency_matrix[i, nearest_indices] = 1
            # Symmetrize
            adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)

    elif similarity_graph == "mknn": # Original logic assumed undirected, let's keep it
        adjacency_matrix = np.zeros([dimension, dimension])
        k_to_use = min(k_mknn, dimension - 1) if k_mknn is not None else 0
        if k_to_use > 0:
            # Precompute kNN sets for efficiency
            knn_sets = {}
            for i in range(dimension):
                 knn_sets[i] = set(np.argsort(dist_mat[i])[1:k_to_use+1])

            for i in tqdm(range(dimension), desc="Building mknn affinity", leave=False, ncols=80):
                 for neighbor in knn_sets[i]:
                     # Check if i is in neighbor's kNN set
                     if i in knn_sets.get(neighbor, set()):
                          adjacency_matrix[i, neighbor] = 1
                          # No need for adjacency_matrix[neighbor, i] = 1 due to outer loop covering all i

    else:
        raise ValueError(f"Unsupported similarity_graph: {similarity_graph}")

    affinity_time = time.time() - start_affinity
    print(f"   Affinity matrix built. Time: {affinity_time:.2f}s")
    # --- End Affinity ---


    # --- Laplacian Calculation ---
    start_laplacian = time.time()
    degrees = np.sum(adjacency_matrix, axis=1)
    # Handle isolated nodes to prevent division by zero
    if np.any(degrees == 0):
        print(f"   ⚠️ Warning: {np.sum(degrees == 0)} nodes have degree 0.")
        degrees += 1e-12 # Add small epsilon

    laplacian_matrix_normalized = None # Initialize
    if laplacian == "sym":
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_half = np.diag(d_inv_sqrt)
        laplacian_matrix_normalized = d_half @ adjacency_matrix @ d_half
    elif laplacian == "rw":
        d_inv = np.power(degrees, -1.0)
        d_inv[np.isinf(d_inv)] = 0
        d_inverse = np.diag(d_inv)
        laplacian_matrix_normalized = d_inverse @ adjacency_matrix
    else:
        raise ValueError("Unsupported laplacian type. Only 'sym' and 'rw' are allowed.")

    laplacian_time = time.time() - start_laplacian
    print(f"   Laplacian calculated. Time: {laplacian_time:.2f}s")
    # --- End Laplacian ---


    # --- Eigen Decomposition & KMeans ---
    start_clustering = time.time() # Time combined eigen + kmeans

    # Determine number of clusters (k)
    current_k = 0 # Initialize
    try:
        if isinstance(number_of_clusters, int):
            current_k = number_of_clusters
        elif number_of_clusters == "fixed2":
            current_k = 2
        elif number_of_clusters == "fixed3":
            current_k = 3
        elif isinstance(number_of_clusters, str) and number_of_clusters.startswith("eigengap"):
            # Calculate eigenvalues for eigengap (simplified version)
            max_k_check = min(dimension - 1, 15) # Check up to 15 eigenvalues
            if max_k_check > 1:
                 # Use appropriate eigenvalue solver
                 if check_symmetric(laplacian_matrix_normalized):
                     # For Lsym/NormAdj, interested in largest eigenvalues (LM)
                     evals = np.linalg.eigvalsh(laplacian_matrix_normalized) # Get all eigenvalues
                     evals_sorted = np.sort(np.abs(evals))[::-1] # Sort by magnitude desc
                 else: # RW case, not symmetric
                     evals = np.linalg.eigvals(laplacian_matrix_normalized) # Get all eigenvalues
                     evals_sorted = np.sort(np.real(evals))[::-1] # Sort by real part desc

                 # Calculate gaps
                 eigengaps = np.abs(np.diff(evals_sorted))
                 relevant_gaps = eigengaps[:max_k_check - 1]
                 if len(relevant_gaps) > 0:
                     optimal_k_eigengap = np.argmax(relevant_gaps) + 1 # +1 because diff reduces length
                     current_k = max(2, optimal_k_eigengap) # Ensure k>=2
                     print(f"   Eigengap heuristic suggests k = {current_k}")
                 else:
                     print("   Not enough eigenvalues for eigengap, using default k=2")
                     current_k = 2
            else:
                 print("   Dimension too small for eigengap, using default k=2")
                 current_k = 2
        else: # Fallback default
            print(f"   Unrecognized 'number_of_clusters': {number_of_clusters}. Using default k=2.")
            current_k = 2

        # Ensure k is valid
        current_k = max(2, current_k) # Minimum k=2 for clustering
        current_k = min(current_k, dimension) # Max k=N
        print(f"   Final number of clusters (k): {current_k}")

        # Eigen decomposition
        if current_k >= dimension: # Avoid requesting N or more eigenvectors
            print("   Warning: k >= N, using all available components.")
            num_eig_needed = dimension -1
        else:
             num_eig_needed = current_k

        if num_eig_needed <= 0:
             print("   Error: Cannot compute eigenvectors for k <= 0.")
             v = np.zeros((dimension, 1)) # Dummy
        else:
             if check_symmetric(laplacian_matrix_normalized):
                 # Use eigh for symmetric, get largest magnitude e-vectors
                 # Note: Standard spectral often uses smallest e-vectors of L=D-A
                 # Here we use largest of normalized adjacency, common variant.
                 e, v = np.linalg.eigh(laplacian_matrix_normalized)
                 idx = np.argsort(np.abs(e))[::-1] # Sort by magnitude desc
                 v = np.real(v[:, idx[:num_eig_needed]])
             else: # RW case
                 e, v = np.linalg.eig(laplacian_matrix_normalized)
                 idx = np.argsort(np.real(e))[::-1] # Sort by real part desc
                 v = np.real(v[:, idx[:num_eig_needed]])

        # Prepare features for KMeans
        X = v # Shape (n_samples, k)

        # Normalize if using symmetric Laplacian (Ng, Jordan, Weiss)
        if laplacian == "sym":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0 # Avoid division by zero
            X = X / norms

        # KMeans Clustering
        if X.shape[0] > 0 and X.shape[1] > 0: # Ensure valid input for KMeans
            kmeans = KMeans(n_clusters=current_k, random_state=42, n_init=10) # Use n_init=10 or 'auto'
            cluster_labels = kmeans.fit_predict(X)
        else:
            print("   ⚠️ Skipping KMeans due to invalid eigenvector matrix.")
            cluster_labels = np.zeros(dimension, dtype=int) # Default assignment

    except np.linalg.LinAlgError as e:
        print(f"   ❌ Linear algebra error during eigen decomposition: {e}. Clustering failed.")
        cluster_labels = np.zeros(dimension, dtype=int) # Assign default cluster
        current_k = 0 # Indicate failure
    except Exception as e:
        print(f"   ❌ Error during eigen/kmeans: {e}")
        cluster_labels = np.zeros(dimension, dtype=int)
        current_k = 0


    clustering_time_combined = time.time() - start_clustering
    print(f"   Clustering (Eigen + KMeans) done. Time: {clustering_time_combined:.2f}s")
    # --- End Clustering ---


    # --- Silhouette Score ---
    start_silhouette = time.time()
    sil_score = -1.0 # Default to indicate not calculated or error
    # Check if calculation is possible and meaningful
    if cluster_labels is not None and current_k > 1 and current_k < dimension:
        try:
            # Using original features (dataframe) for silhouette
            sil_score = silhouette_score(dataframe, cluster_labels)
            print(f"   Silhouette Score: {sil_score:.4f}")
        except ValueError as e:
            print(f"   ⚠️ Could not calculate Silhouette Score: {e}")
            # sil_score remains -1.0
    else:
        if cluster_labels is None:
            print("   ⚠️ Skipping Silhouette Score (clustering failed).")
        elif current_k <= 1:
             print("   ⚠️ Skipping Silhouette Score (k <= 1).")
        elif current_k >= dimension:
             print("   ⚠️ Skipping Silhouette Score (k >= n).")

    silhouette_time = time.time() - start_silhouette
    # --- End Silhouette ---

    # Final check on cluster_labels type before returning
    if cluster_labels is None:
        cluster_labels = np.array([], dtype=int)


    # --- Return Results ---
    # Return the 6 values expected by the calling function
    return cluster_labels, sil_score, affinity_time, laplacian_time, clustering_time_combined, silhouette_time

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
                properties = { (f"feature_{key}" if str(key).isdigit() else key): value for key, value in row.items() }
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

def create_edges(data, driver, node_label, edge_label):
    try:
        edge_data = pd.read_csv(data)
        with driver.session() as session:
            for _, row in edge_data.iterrows():
                source_id = min(row['source_id'], row['target_id'])
                target_id = max(row['source_id'], row['target_id'])
                query = f"MATCH (source:{node_label} {{id: $source_id}}) MATCH (target:{node_label} {{id: $target_id}}) MERGE (source)-[:{edge_label} {{value: 1}}]->(target)"
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
    memory_used = (end_mem - start_mem) / (1024 ** 2)
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

def run_python_experiment_feature(config, file_path, driver, experiment_tag):
    """
    Runs Python spectral clustering, calculates metrics, and writes the
    resulting cluster assignments and centroids to Neo4j using the 'Index' column.
    """
    print(f"\n--- Running Python Feature Experiment & Neo4j Write: {experiment_tag} ---")
    # --- Start: Setup and Data Loading ---
    df = pd.read_csv(file_path)

    # --- Use the 'Index' column directly as the identifier ---
    id_property_in_neo4j = 'Index' # Define the property name used in Neo4j
    if id_property_in_neo4j not in df.columns:
        print(f"❌ Error: Required '{id_property_in_neo4j}' column not found in CSV for linking.")
        return {"error": f"'{id_property_in_neo4j}' column missing"}

    try:
        # Use the actual 'Index' column values, ensuring integer type
        identifiers = df[id_property_in_neo4j].astype(int).values
        print(f"   Using column '{id_property_in_neo4j}' as identifier for Neo4j linking.")
    except ValueError:
         print(f"❌ Error: Cannot convert '{id_property_in_neo4j}' column to integer.")
         return {"error": f"Cannot convert '{id_property_in_neo4j}' column to integer"}
    # --- End Identifier Setup ---


    # Ensure 'Index' is NOT removed by the config setting 'remove_columns'
    cols_to_remove_config = config["remove_columns"].split(',') if config.get("remove_columns") else []
    cols_to_remove = [col.strip() for col in cols_to_remove_config]
    if id_property_in_neo4j in cols_to_remove:
         cols_to_remove.remove(id_property_in_neo4j)
         print(f"   Kept '{id_property_in_neo4j}' column for identification.")
    # Remove previous 'nodeId' logic if it exists (no longer needed)
    if 'nodeId' in cols_to_remove:
         cols_to_remove.remove('nodeId')

    features_df = df.drop(columns=cols_to_remove, errors='ignore')

    # --- (Feature matrix conversion, true labels, timing start - as before) ---
    try:
        feature_matrix = features_df.to_numpy(dtype=float)
    except ValueError as e:
        print(f"   ⚠️ Warning: Could not convert features to numeric numpy array: {e}. Centroid calculation might fail.")
        features_df_numeric = features_df.select_dtypes(include=np.number)
        if not features_df_numeric.empty:
            feature_matrix = features_df_numeric.to_numpy(dtype=float)
            print(f"   Using only numeric columns for features: {features_df_numeric.columns.tolist()}")
        else:
            print("   ❌ Error: No numeric feature columns found. Cannot proceed.")
            return {"error": "No numeric features for clustering/centroids"}

    true_labels = None
    if config["target_column"] in df.columns:
        true_labels = df[config["target_column"]].values
    else:
        print(f"   Target column '{config['target_column']}' not found.")

    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    start_cpu_times = process.cpu_times()
    start_python_execution = time.time() # Time Python part separately
    # --- End: Setup ---

    # --- (Python Spectral Clustering call - as before) ---
    try:
        # Expects: labels, sil_score, aff_time, lap_time, cluster_time(eigen+kmeans), sil_time
        cluster_labels, sil_score_val, affinity_time, laplacian_time, clustering_time, _ = spectral_clustering(
            features_df, # Pass features DataFrame
            config["graph_type"], config["laplacian_type"], config["number_of_eigenvectors"],
            # Pass n, eps, k parameters correctly based on previous logic...
            n=config.get('n_val'), eps=config.get('eps_val'), k_knn=config.get('knn_val'), k_mknn=config.get('mknn_val') # Make sure these are set before calling if needed
        )
    # (Error handling for spectral_clustering call - as before)
    except NameError:
         print("❌ Error: spectral_clustering function is not defined or imported.")
         return {"error": "spectral_clustering function not found"}
    except Exception as e:
         print(f"❌ Error during spectral_clustering execution: {e}")
         return {"error": f"Spectral clustering failed: {e}"}

    python_execution_time = time.time() - start_python_execution
    print(f"   🐍 Python spectral clustering finished. Time: {python_execution_time:.2f}s")
    # --- End: Python Spectral Clustering ---


    # --- (Python Metric Calculation - as before) ---
    ari_time = 0.0
    python_rand_index = np.nan
    if true_labels is not None and cluster_labels is not None and len(cluster_labels) == len(true_labels):
         if len(np.unique(cluster_labels)) > 1:
             ari_start = time.time(); python_rand_index = adjusted_rand_score(true_labels, cluster_labels); ari_time = time.time() - ari_start
    # (Error handling for ARI, checking cluster_labels not None etc. - as before)
    # --- End: Python Metric Calculation ---


    # --- Start: Neo4j Writing ---
    neo4j_write_time = np.nan # Default
    if driver and cluster_labels is not None and len(identifiers) == len(cluster_labels):
        print(f"    GDB Writing Python results to Neo4j for experiment: {experiment_tag}...")
        start_neo4j_write = time.time()
        node_label = config['node_label'] # e.g., 'IrisNode'

        try:
            unique_clusters = sorted(np.unique(cluster_labels))
            if len(unique_clusters) == 0:
                 print("   ⚠️ No clusters found, skipping Neo4j write.")
            else:
                # 1. Calculate Centroids (using feature_matrix - as before)
                centroids = {}
                if feature_matrix is not None:
                     for cluster_id in unique_clusters:
                         cluster_features = feature_matrix[cluster_labels == cluster_id]
                         centroids[cluster_id] = cluster_features.mean(axis=0).tolist() if cluster_features.shape[0] > 0 else [np.nan] * feature_matrix.shape[1]
                     print(f"      Calculated {len(centroids)} centroids.")
                else:
                    print("      ⚠️ Skipping centroid calculation as feature matrix is unavailable.")


                # 2. Write Centroids and Link Nodes in Neo4j session
                with driver.session(database="neo4j") as session:
                    # Create/Update Centroid Nodes (as before)
                    if centroids:
                        centroid_batch = [{"cluster_id": int(cid), "coords": ccoords} for cid, ccoords in centroids.items()]
                        session.run("""
                            UNWIND $batch AS c_data
                            MERGE (c:ClusterCentroid {clusterId: c_data.cluster_id, experiment: $experiment_tag})
                            SET c.coordinates = c_data.coords, c.nodeLabel = $nodeLabel
                            """, batch=centroid_batch, experiment_tag=experiment_tag, nodeLabel=node_label)
                        print(f"      Created/updated {len(centroid_batch)} ClusterCentroid nodes.")

                    # Link Data Nodes to Centroids
                    # Use the 'identifiers' derived from the 'Index' column (already integers)
                    link_data = [{"identifier": int(identifiers[i]), "cluster_id": int(cluster_labels[i])}
                                 for i in range(len(identifiers))]

                    batch_size = 2000
                    print(f"      Linking {len(link_data)} '{node_label}' nodes using property '{id_property_in_neo4j}'...")
                    for i in tqdm(range(0, len(link_data), batch_size), desc="Linking nodes", ncols=80, leave=False):
                        batch = link_data[i:i+batch_size]
                        # Use the correct id_property_in_neo4j ('Index') in the MATCH clause
                        cypher_link = f"""
                            UNWIND $batch AS link
                            MATCH (d:{node_label} {{{id_property_in_neo4j}: link.identifier}}) // Ensure this matches node prop
                            MATCH (c:ClusterCentroid {{clusterId: link.cluster_id, experiment: $experiment_tag}})
                            MERGE (d)-[r:BELONGS_TO_CLUSTER {{experiment: $experiment_tag}}]->(c)
                        """
                        session.run(cypher_link, batch=batch, experiment_tag=experiment_tag)

                neo4j_write_time = time.time() - start_neo4j_write
                print(f"   ✅ Finished writing results to Neo4j. Time: {neo4j_write_time:.2f}s")

        # (Error handling for Neo4j write - as before)
        except Exception as e:
             print(f"   ❌ Error writing results to Neo4j: {e}")
             neo4j_write_time = -1.0 # Indicate error

    # (Checks for driver, cluster_labels None, length mismatch - as before)
    elif not driver: print("   ⚠️ Neo4j driver not provided, skipping write operation.")
    elif cluster_labels is None: print("   ⚠️ Cluster labels are None (clustering failed), skipping Neo4j write.")
    else: print("   ⚠️ Identifiers and cluster_labels length mismatch, skipping Neo4j write.")
    # --- End: Neo4j Writing ---

    # --- (Final Metrics Calculation & Return - as before) ---
    end_cpu_times = process.cpu_times(); cpu_used = (end_cpu_times.user + end_cpu_times.system) - (start_cpu_times.user + start_cpu_times.system)
    end_mem = process.memory_info().rss; memory_used = (end_mem - start_mem) / (1024 ** 2)
    results_dict = {
        "python_silhouette_score": sil_score_val if cluster_labels is not None else np.nan,
        "python_rand_index": python_rand_index,
        "python_total_time": python_execution_time,
        "python_affinity_time": affinity_time if cluster_labels is not None else np.nan,
        "python_laplacian_time": laplacian_time if cluster_labels is not None else np.nan,
        "python_clustering_time": clustering_time if cluster_labels is not None else np.nan,
        "python_adjusted_rand_index_time": ari_time,
        "python_memory_used_MB": memory_used,
        "python_cpu_used_seconds": cpu_used,
        "neo4j_write_time_seconds": neo4j_write_time
    }
    # --- End: Final Metrics ---

    return results_dict


def run_python_experiment_graph(config, node_file_path, edge_file_path, driver, experiment_tag):
    """
    Runs Python spectral clustering on graph structure, calculates metrics,
    calculates centroids from node features, and writes results to Neo4j.

    Args:
        config (dict): Dictionary with experiment configuration. Must include 'node_label'.
        node_file_path (str): Path to the node CSV file (must contain 'id' column).
        edge_file_path (str): Path to the edge CSV file.
        driver (neo4j.Driver): Active Neo4j driver instance.
        experiment_tag (str): A unique string identifying this specific experimental run.

    Returns:
        dict: Dictionary containing Python performance metrics and Neo4j write time.
    """
    print(f"\n--- Running Python Graph Experiment & Neo4j Write: {experiment_tag} ---")
    # --- Start: Setup and Data Loading ---
    try:
        nodes_df = pd.read_csv(node_file_path)
        edge_df = pd.read_csv(edge_file_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found: {e}")
        return {"error": f"Data file not found: {e}"}

    # --- Use the 'id' column as the identifier ---
    id_property_in_neo4j = 'id' # Property name used in Neo4j MATCH for graph nodes
    if id_property_in_neo4j not in nodes_df.columns:
         print(f"❌ Error: Required '{id_property_in_neo4j}' column not found in node file for linking.")
         return {"error": f"'{id_property_in_neo4j}' column missing in node file"}
    # Type of identifier depends on the data (can be int or string)
    identifiers = nodes_df[id_property_in_neo4j].values
    print(f"   Using column '{id_property_in_neo4j}' as identifier for Neo4j linking.")
    # --- End Identifier Setup ---

    # Load true labels if available
    true_labels = None
    if config["target_column"] in nodes_df.columns:
        true_labels = nodes_df[config["target_column"]].values
    else:
        print(f"   Target column '{config['target_column']}' not found in node file.")

    # Load features for centroid calculation (handle potential 'eval' and errors)
    feature_matrix = None
    node_label = config['node_label'] # Get label early for messages
    if "features" in nodes_df.columns:
        print(f"   Processing 'features' column for {node_label}...")
        try:
            # Attempt to evaluate string representations safely
            def safe_eval(x):
                if isinstance(x, str):
                    try: return eval(x)
                    except: return None # Return None if eval fails
                return x # Assume list/array otherwise
            feature_list = nodes_df["features"].apply(safe_eval).tolist()

            # Check if parsing worked and features are consistent
            if all(isinstance(f, (list, np.ndarray)) for f in feature_list if f is not None):
                 # Get length of first valid feature
                 first_valid_feature = next((f for f in feature_list if f is not None), None)
                 if first_valid_feature is not None:
                     feature_len = len(first_valid_feature)
                     # Check consistency and filter out None entries if needed
                     valid_features = [f for f in feature_list if f is not None and len(f) == feature_len]
                     if len(valid_features) == len(feature_list): # All features were valid and consistent
                         feature_matrix = np.array(valid_features, dtype=float)
                         print(f"   Loaded feature matrix with shape {feature_matrix.shape}")
                     else:
                          print(f"   ⚠️ Warning: {len(feature_list) - len(valid_features)} rows had invalid or inconsistent features. Centroids might be affected.")
                          # Option: Create matrix only with valid features? Needs careful index mapping later.
                          # For simplicity, we'll flag that centroids might be unreliable if this happens.
                          # Setting feature_matrix to None prevents centroid calculation.
                          feature_matrix = None
                 else:
                      print("   ⚠️ No valid features found after parsing.")
            else:
                 print("   ⚠️ Could not parse 'features' column into consistent lists/arrays.")

        except Exception as e:
            print(f"   ⚠️ Warning: Error processing 'features' column: {e}. Cannot calculate centroids.")
    else:
        print("   ⚠️ 'features' column not found in node file. Cannot calculate centroids.")


    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    start_cpu_times = process.cpu_times()
    start_python_execution = time.time() # Time Python part separately

    # --- Start: Python Spectral Clustering on Graph Structure ---
    print(f"   Building adjacency matrix for {len(identifiers)} nodes...")
    id_to_index = {identifier: idx for idx, identifier in enumerate(identifiers)}
    dim = len(identifiers)
    adjacency_matrix = sp.lil_matrix((dim, dim)) # Use sparse matrix
    affinity_start = time.time()
    # Ensure source/target IDs from edge file are looked up correctly
    for _, row in edge_df.iterrows():
        src_idx = id_to_index.get(row["source_id"]) # Use actual ID value
        tgt_idx = id_to_index.get(row["target_id"]) # Use actual ID value
        if src_idx is not None and tgt_idx is not None:
            adjacency_matrix[src_idx, tgt_idx] = 1
            adjacency_matrix[tgt_idx, src_idx] = 1 # Assuming undirected
    adjacency_matrix = adjacency_matrix.tocsr()
    affinity_time = time.time() - affinity_start
    print(f"   Adjacency matrix built. Time: {affinity_time:.2f}s")


    # Compute Laplacian
    laplacian_start = time.time()
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    if np.any(degrees == 0):
        print(f"   ⚠️ Warning: {np.sum(degrees == 0)} nodes have degree 0.")
        degrees += 1e-12 # Add small epsilon

    laplacian_matrix_normalized = None
    try:
        if config["laplacian_type"] == "sym":
            d_inv_sqrt = np.power(degrees, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_half = sp.diags(d_inv_sqrt)
            laplacian_matrix_normalized = d_half @ adjacency_matrix @ d_half # Normalized Adjacency
        elif config["laplacian_type"] == "rw":
            d_inv = np.power(degrees, -1.0)
            d_inv[np.isinf(d_inv)] = 0
            d_inverse = sp.diags(d_inv)
            laplacian_matrix_normalized = d_inverse @ adjacency_matrix # Transition Matrix P
        else:
            raise ValueError(f"Unsupported laplacian type: {config['laplacian_type']}")
    except Exception as e:
        print(f"   ❌ Error calculating Laplacian: {e}")
        return {"error": f"Laplacian calculation failed: {e}"}

    laplacian_time = time.time() - laplacian_start
    print(f"   Laplacian calculated. Time: {laplacian_time:.2f}s")

    # Eigen-decomposition and clustering
    clustering_start = time.time()
    cluster_labels = None
    sil_score_val = np.nan
    try:
        # Determine number of clusters (k) - using simplified logic from previous example
        number_of_clusters = config["number_of_eigenvectors"]
        current_k = 0
        if isinstance(number_of_clusters, int): current_k = number_of_clusters
        elif number_of_clusters == "fixed2": current_k = 2
        elif number_of_clusters == "fixed3": current_k = 3
        elif isinstance(number_of_clusters, str) and number_of_clusters.startswith("eigengap"):
             # Eigengap logic (simplified - requires check_symmetric, assumes numpy/scipy linalg)
             # ... (eigengap calculation logic would go here, setting current_k) ...
             print("   (Eigengap logic placeholder - using default k=2 for now)")
             current_k = 2 # Placeholder if eigengap logic omitted here
        else: current_k = 2 # Default

        current_k = max(2, min(current_k, dim)) # Ensure valid k
        print(f"   Final number of clusters (k): {current_k}")

        # Eigen decomposition (request k eigenvectors)
        num_eig_needed = min(current_k, dim - 1)
        if num_eig_needed <= 0: raise ValueError("k must be > 0")

        if check_symmetric(laplacian_matrix_normalized):
             # Use eigsh for sparse symmetric, get largest magnitude ('LM')
             # Increase maxiter for potentially better convergence
             eigenvalues, eigenvectors = sp.linalg.eigsh(laplacian_matrix_normalized, k=num_eig_needed, which='LM', tol=1e-6, maxiter=dim*5)
             idx = np.argsort(np.abs(eigenvalues))[::-1]
             eigenvectors = np.real(eigenvectors[:, idx])
        else: # Not symmetric (RW case)
             # Use eigs for sparse non-symmetric, get largest real part ('LR')
             eigenvalues, eigenvectors = sp.linalg.eigs(laplacian_matrix_normalized, k=num_eig_needed, which='LR', tol=1e-6, maxiter=dim*5)
             eigenvectors = np.real(eigenvectors) # Eigenvectors for real eigenvalues are real

        X = eigenvectors # Features for KMeans

        # Normalize if using symmetric Laplacian
        if config["laplacian_type"] == "sym":
            norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1.0; X = X / norms

        # KMeans Clustering
        if X.shape[0] > 0 and X.shape[1] > 0:
            kmeans = KMeans(n_clusters=current_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
        else:
            print("   ⚠️ Skipping KMeans due to invalid eigenvector matrix.")
            cluster_labels = np.zeros(dim, dtype=int)

        # Calculate Silhouette score (on eigenvectors X)
        if cluster_labels is not None and len(np.unique(cluster_labels)) > 1 and len(np.unique(cluster_labels)) < X.shape[0]:
            try:
                 sil_score_val = silhouette_score(X, cluster_labels)
                 print(f"   Silhouette Score (on Eigenvectors): {sil_score_val:.4f}")
            except ValueError as e: print(f"   ⚠️ Could not calculate Silhouette Score: {e}")
        else: print("   ⚠️ Skipping Silhouette Score (k<=1 or k>=n).")

    except (np.linalg.LinAlgError, sp.linalg.ArpackNoConvergence) as e:
         print(f"   ❌ Linear algebra/Convergence error during eigen/kmeans: {e}. Clustering failed.")
         # cluster_labels remains None or default zeros
         if cluster_labels is None: cluster_labels = np.zeros(dim, dtype=int)
    except Exception as e:
         print(f"   ❌ Error during clustering steps: {e}")
         if cluster_labels is None: cluster_labels = np.zeros(dim, dtype=int)

    clustering_time = time.time() - clustering_start
    print(f"   Clustering (Eigen + KMeans) done. Time: {clustering_time:.2f}s")
    # --- End: Python Spectral Clustering ---

    # --- Start: Python Metric Calculation (ARI) ---
    ari_time = 0.0
    python_rand_index = np.nan
    if true_labels is not None and cluster_labels is not None and len(cluster_labels) == len(true_labels):
         if len(np.unique(cluster_labels)) > 1:
             ari_start = time.time(); python_rand_index = adjusted_rand_score(true_labels, cluster_labels); ari_time = time.time() - ari_start
    # (Error handling as before)
    # --- End: Python Metric Calculation ---


    # --- Start: Neo4j Writing ---
    neo4j_write_time = np.nan # Default
    if driver and cluster_labels is not None and len(identifiers) == len(cluster_labels):
        print(f"    GDB Writing Python results to Neo4j for experiment: {experiment_tag}...")
        start_neo4j_write = time.time()
        # node_label = config['node_label'] # Already got this

        try:
            unique_clusters = sorted(np.unique(cluster_labels))
            if len(unique_clusters) == 0:
                 print("   ⚠️ No clusters found, skipping Neo4j write.")
            else:
                # 1. Calculate Centroids (using feature_matrix loaded earlier)
                centroids = {}
                if feature_matrix is not None: # Check if features were loaded successfully
                     for cluster_id in unique_clusters:
                         # Make sure indices align: labels correspond to rows in feature_matrix
                         cluster_features = feature_matrix[cluster_labels == cluster_id]
                         if cluster_features.shape[0] > 0:
                             centroids[cluster_id] = cluster_features.mean(axis=0).tolist()
                         else: # Handle potentially empty cluster
                             centroids[cluster_id] = [np.nan] * feature_matrix.shape[1]
                             print(f"      Warning: Cluster {cluster_id} is empty.")
                     print(f"      Calculated {len(centroids)} centroids.")
                else:
                    print("      ⚠️ Cannot calculate centroids as features are unavailable.")


                # 2. Write Centroids and Link Nodes in Neo4j session
                with driver.session(database="neo4j") as session:
                    # Create/Update Centroid Nodes
                    if centroids: # Only write if centroids were calculated
                        centroid_batch = [{"cluster_id": int(cid), "coords": ccoords}
                                          for cid, ccoords in centroids.items() if not np.isnan(ccoords).any()] # Avoid writing NaN centroids
                        if centroid_batch:
                             session.run("""
                                 UNWIND $batch AS c_data
                                 MERGE (c:ClusterCentroid {clusterId: c_data.cluster_id, experiment: $experiment_tag})
                                 SET c.coordinates = c_data.coords, c.nodeLabel = $nodeLabel
                                 """, batch=centroid_batch, experiment_tag=experiment_tag, nodeLabel=node_label)
                             print(f"      Created/updated {len(centroid_batch)} ClusterCentroid nodes.")
                        else:
                             print("      No valid centroids to write (all might have been NaN).")


                    # Link Data Nodes to Centroids
                    # Use the 'identifiers' derived from the 'id' column
                    # Ensure type consistency (identifier vs node property type)
                    link_data = [{"identifier": identifiers[i], "cluster_id": int(cluster_labels[i])}
                                 for i in range(len(identifiers))]

                    batch_size = 2000
                    print(f"      Linking {len(link_data)} '{node_label}' nodes using property '{id_property_in_neo4j}'...")
                    for i in tqdm(range(0, len(link_data), batch_size), desc="Linking nodes", ncols=80, leave=False):
                        batch = link_data[i:i+batch_size]
                        # Use the correct id_property_in_neo4j ('id') in the MATCH clause
                        # Ensure link.identifier type matches node property type
                        cypher_link = f"""
                            UNWIND $batch AS link
                            MATCH (d:{node_label} {{{id_property_in_neo4j}: link.identifier}}) // Match using 'id'
                            MATCH (c:ClusterCentroid {{clusterId: link.cluster_id, experiment: $experiment_tag}})
                            MERGE (d)-[r:BELONGS_TO_CLUSTER {{experiment: $experiment_tag}}]->(c)
                        """
                        # Note: If IDs in Neo4j are strings, convert identifier in batch dict if needed.
                        # Assuming create_graph_nodes stored IDs with the correct type from CSV.
                        session.run(cypher_link, batch=batch, experiment_tag=experiment_tag)

                neo4j_write_time = time.time() - start_neo4j_write
                print(f"   ✅ Finished writing results to Neo4j. Time: {neo4j_write_time:.2f}s")

        # (Error handling for Neo4j write - as before)
        except Exception as e:
             print(f"   ❌ Error writing results to Neo4j: {e}")
             neo4j_write_time = -1.0 # Indicate error

    # (Checks for driver, cluster_labels None, length mismatch - as before)
    elif not driver: print("   ⚠️ Neo4j driver not provided, skipping write operation.")
    elif cluster_labels is None: print("   ⚠️ Cluster labels are None (clustering failed), skipping Neo4j write.")
    else: print("   ⚠️ Identifiers and cluster_labels length mismatch, skipping Neo4j write.")
    # --- End: Neo4j Writing ---


    # --- Start: Final Metrics Calculation & Return ---
    end_cpu_times = process.cpu_times(); cpu_used = (end_cpu_times.user + end_cpu_times.system) - (start_cpu_times.user + start_cpu_times.system)
    end_mem = process.memory_info().rss; memory_used = (end_mem - start_mem) / (1024 ** 2) # In MB

    # Calculate total Python execution time from measured components
    python_total_exec_time = laplacian_time + clustering_time + ari_time

    results_dict = {
        "python_silhouette_score": sil_score_val, # Based on eigenvectors
        "python_rand_index": python_rand_index,
        "python_total_time": python_total_exec_time, # Sum of measured Python steps
        "python_affinity_time": affinity_time, # Adjacency matrix creation time
        "python_laplacian_time": laplacian_time,
        "python_clustering_time": clustering_time, # Eigen + KMeans time
        "python_adjusted_rand_index_time": ari_time,
        "python_memory_used_MB": memory_used, # Memory for the whole function
        "python_cpu_used_seconds": cpu_used,  # CPU for the whole function
        "neo4j_write_time_seconds": neo4j_write_time # Add the new timing metric
    }
    # --- End: Final Metrics ---

    return results_dict

def run_experiments(driver, experiments):
    print("Init")
    try:
        with driver.session() as session:
            session.run("RETURN simkit.initSimKit('bolt://localhost:7687', 'neo4j', '123412345')")
    except Exception as e:
        print(f"Error initializing SimKit: {e}")
        return
    print("Init Done")
    results = []
    total_experiments = len(experiments)
    for idx, config in enumerate(experiments, 1):
        print(config)
        query = "WITH simkit.experimental_spectralClustering({ node_label: $node_label, is_feature_based: $is_feature_based, distance_measure: 'euclidean', graph_type: $graph_type, parameter: $parameter, remove_columns: $remove_columns, laplacian_type: $laplacian_type, number_of_eigenvectors: $number_of_eigenvectors, number_of_iterations: 100, distance_measure_kmean: 'euclidean', target_column: $target_column, use_kmean_for_silhouette: $use_kmean_for_silhouette, seed: 42 }) AS result RETURN result.silhouette_score AS silhouette_score, result.rand_index AS rand_index, result.total_time AS total_time, result.affinity_time AS affinity_time, result.laplacian_time AS laplacian_time, result.clustering_time AS clustering_time, result.adjusted_rand_index_time AS adjusted_rand_index_time"
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
            filename = f"{config['node_label'].replace('Node','').lower()}.csv"
            python_result = run_python_experiment_feature(config, os.path.join("datasets", filename), driver, f"experiment_feature_{filename}_{memory_used}")
        else:
            node_file_path = os.path.join("datasets", f"{config['node_label'].replace('Node','').lower()}_nodes.csv")
            edge_file_path = os.path.join("datasets", f"{config['node_label'].replace('Node','').lower()}_edges.csv")
            python_result = run_python_experiment_graph(config, node_file_path, edge_file_path, driver, f"experiment_graph_{config['node_label'].replace('Node','').lower()}_{memory_used}")
        merged_result = {**config, **simkit_result, **python_result}
        results.append(merged_result)
        print(f"Completed experiment {idx}/{total_experiments} with config: {config}")
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

feature_datasets = {
    #"iris": {"label": "IrisNode", "remove_columns": "Index,target", "number_of_eigenvectors": 3, "target_column": "target"}
}

graph_datasets = {
    "citeseer": {"node_label": "CiteSeerNode", "edge_label": "CiteSeerEdge", "remove_columns": "id,label", "number_of_eigenvectors": 6, "target_column": "label"}
}

if __name__ == "__main__":
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
    python_avg = combined_df[['python_total_time', 'python_cpu_used', 'python_silhouette_score', 'python_rand_index']].mean()
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
