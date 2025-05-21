# SimKit Experiments with Neo4j & Python

This repository demonstrates how to run spectral clustering experiments using both the SimKit plugin (loaded into Neo4j) and scikit-learn. It also includes experiments using Neo4j Graph Data Science (GDS) algorithms.

## Features
- Dockerized Neo4j setup with the SimKit plugin.
- Scripts to compare clustering results using:
- SimKit (custom Neo4j plugin)
- scikit-learn
- Neo4j GDS library
- Batch and timing experiments for performance comparison.

## Contents
- Dockerfile and docker-compose.yml: Builds and runs a Neo4j instance with the SimKit plugin.
- requirements.txt: Python dependencies.
- SimKit-0.1.1.jar: SimKit plugin (must be compatible with the Neo4j version used).
- Python scripts:
  - experiment_gds.py: Runs k-means clustering using Neo4j GDS.
  - experiments_simkit-0.1.1.py: Runs and times clustering using SimKit and scikit-learn.

## Prerequisites
- Docker installed on your system.
- Python 3.x and pip installed.
- Memory: Ensure your system has enough RAM (Neo4j can be memory-intensive).
- Place the simkit.jar plugin (compatible with your Neo4j version) in the same directory as the Dockerfile.

## Setup Instructions
1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/simkit-experiments.git
    cd simkit-experiments
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the Neo4j database:
    ```bash
    docker compose up
    ```

    This will build and start the Neo4j container with SimKit.


4. Run experiments:
   - Run GDS clustering experiments on Neo4j:
    ```bash
    python experiment_gds.py
    ```

   - Run SimKit and scikit-learn timing experiments:
    ```bash
    python experiments_simkit-0.1.1.py
    ```

## Notes
- Ensure the Neo4j container is fully up before starting any experiments.
- The scripts assume the Neo4j instance is accessible at the configured bolt:// address and uses default credentials (neo4j/neo4j by default; change if modified).
- You may need to load or generate a sample graph dataset in Neo4j before running experiments.

## License

This project is licensed under the Apache License 2.0.