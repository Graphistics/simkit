# SimKit Experiments with Neo4j & Python

This repository demonstrates how to run spectral clustering experiments using both the SimKit plugin (loaded into Neo4j) and scikit-learn. It includes:
- A Dockerfile to build a Neo4j container with the SimKit plugin.
- The `experiments_2.py` script which runs experiments against the Neo4j instance.

## Prerequisites

- Docker must be installed on your system.
- Python 3.x and pip installed.
- Ensure your system has enough memory (Neo4j can be memory intensive).
- Place the `simkit.jar` plugin file (compatible with your Neo4j version) in the same folder as the Dockerfile.

## Building and Running the Neo4j Docker Container

1. **Build the Docker Image**

   Open a terminal in the directory containing the Dockerfile and `simkit.jar` and run:

   ```bash
   docker build -t my-neo4j .
   ```

   This builds a Docker image named `my-neo4j` that includes the SimKit plugin.

2. **Run the Docker Container**

   Start the container with:

   ```bash
   docker run --name neo4j -p 7687:7687 -p 7474:7474 -d my-neo4j
   ```

   - **Ports:**  
     - `7687` is the Bolt port for Neo4j.
     - `7474` is the HTTP port (Neo4j Browser).

3. **Verify the Container**

   - Check logs for any errors (especially plugin errors):

     ```bash
     docker logs neo4j
     ```

   - Open [http://localhost:7474](http://localhost:7474) in your browser to access the Neo4j Browser and verify that the SimKit procedures are available (e.g., try a test query like `RETURN simkit.experimental_spectralClustering({ ... })`).

## Running the Experiments

The `experiments_2.py` script runs spectral clustering experiments using both SimKit (via Neo4j procedures) and scikit-learn spectral clustering.

1. **Install Python Dependencies**

   The script will attempt to install missing packages automatically. Alternatively, install manually:

   ```bash
   pip install neo4j pandas psutil tqdm scikit-learn scipy
   ```

2. **Prepare Datasets**

   Place your dataset CSV files under the `datasets/` directory. The script expects file names such as `iris.csv`, `cora_nodes.csv`, `cora_edges.csv`, etc.

3. **Run the Experiment Script**

   Ensure the Neo4j container is running, then execute:

   ```bash
   python experiments_2.py
   ```

   The script will:
   - Delete existing nodes and indexes in Neo4j.
   - Create feature/graph nodes from the datasets.
   - Run experiments using both SimKit and scikit-learn.
   - Save results as CSV files in a `results/` directory.

## Troubleshooting

- **Container Exits Early:**  
  Check the container logs with:

  ```bash
  docker logs neo4j
  ```

  Review for errors related to plugin incompatibility, configuration issues, or memory constraints.

- **SimKit Procedure Not Found:**  
  Ensure the `simkit.jar` file is correctly placed and compatible with the Neo4j version used.

- **Python Errors:**  
  Verify that all dataset files exist and have the expected schema.

## Cleanup

To stop and remove the Docker container:

```bash
docker stop neo4j
docker rm neo4j
```

To remove the Docker image:

```bash
docker rmi my-neo4j
```