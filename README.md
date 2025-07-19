# SimKit
Plug-in for Neo4j containing user-defined procedures to perform Spectral Clustering through a single procedure or as individual procedures (Similarity Graph, Laplacian Eigendecomposed Graph, KMeans). Additional procedures available for evaluation with Average Silhouette Score and Adjusted Rand Index, visualization of matrices into String and saving them to CSV.

![SimKit Spectral Clustering Flowchart](./vis/activity_simkit.png)


## [About](https://github.com/Graphistics/simkit/wiki)

### Prerequisites

-   **Neo4j Database:** SimKit has been tested with Neo4j versions 4.2.3 and later. Please check the specific release notes for the latest compatibility information.
-   **Java:** A Java runtime environment (JRE) compatible with your Neo4j version is required (e.g., JDK 11 for Neo4j 4.x).

### Steps

1.  **Download the JAR file**
    [**SimKit v-0.1.2**](https://github.com/Graphistics/simkit/blob/dev-update-readme-and-structure/experiments/SimKit-0.1.2.jar)

2.  **Place the JAR file in the `plugins` directory**
    Copy or move the downloaded `.jar` file into the `plugins` folder of your Neo4j database installation. The path is typically:
    ```bash
    <path-to-your-neo4j-directory>/plugins/
    ```

3.  **Configure Neo4j**
    To allow SimKit's procedures to be called, you must update your Neo4j configuration file, located at `<path-to-your-neo4j-directory>/conf/neo4j.conf`. This is a crucial security step that explicitly grants permission for the plug-in to run. Add or uncomment the following line:
    ```
    dbms.security.procedures.unrestricted=simkit.*
    ```
    This setting explicitly permits procedures within the `simkit.*` and `gds.*` packages to be executed. Including `gds.*` is recommended to ensure compatibility if you are using both libraries.

4.  **Restart Neo4j**
    After placing the `.jar` file and updating the configuration, you must **restart your Neo4j database**. This will load the SimKit plug-in, making its procedures available to call from Cypher.


## Procedure Stack

### Example Data: Iris Dataset

The parameters in the procedure examples above (e.g., `label: "Iris"`, `remove_column: "index,target"`) refer to data that has been imported into Neo4j. For context, here is a sample of what the Iris dataset looks like before being imported. Each row would typically be imported as a node with the label `Iris`.

| index | sepal_length | sepal_width | petal_length | petal_width | target |
| :---- | :----------- | :---------- | :----------- | :---------- | :----- |
| 0     | 5.1          | 3.5         | 1.4          | 0.2         | 0      |
| 1     | 4.9          | 3.0         | 1.4          | 0.2         | 0      |
| 50    | 7.0          | 3.2         | 4.7          | 1.4         | 1      |
| 51    | 6.4          | 3.2         | 4.5          | 1.5         | 1      |
| 100   | 6.3          | 3.3         | 6.0          | 2.5         | 2      |
| 101   | 5.8          | 2.7         | 5.1          | 1.9         | 2      |

### Init SimKit
```java
return simkit.initSimKit('bolt://localhost:7687', 'neo4j', '123412345')
```

### Create Graph From Nodes
```java
return simkit.nodePropertyToGraph({
    label: "Iris",
    distance_measure: "euclidean",
    graph_type: "full",
    parameter: 7,
    remove_column: "index,target"
})
```

### Create EigenGraph
```java
return simkit.nodePropertyEigen({
    label: "affinity_full_7_Iris",
    laplacian_type: "sym",
    number_of_eigenvectors: 3
})
```

### Kmean Clustering
```java
return simkit.kMeans({
    label: "eigen_sym_3_affinity_full_7_Iris",
    number_of_centroids: 3,
    number_of_iterations: 100,
    distance_measure: "euclidean",
    original_set: "Iris",
    overlook: "target,sepal_length,sepal_width,petal_length,petal_width",
    overlook_original: "target",
    silhouette: false
})
```

### Calculate Adjusted rand index
```java
return simkit.adjustedRandIndex({
    label: "Iris",
    true_labels: "target"
})
```

### Spectral Clustering
```java
return simkit.spectralClustering({
    label: "Iris",
    is_feature_based: true,
    distance_measure: "euclidean",
    graph_type: "full",
    parameter: 7,
    remove_columns: "index,target",
    laplacian_type: "sym",
    number_of_eigenvectors: 3,
    number_of_iterations: 100,
    distance_measure_kmean: "euclidean",
    target_column: "target",
    silhouette: false,
    seed: 42
})
```

### Visualize Matrix and save as CSV
```java
return simkit.getMatrix("affinity_full_7_Iris","adjacency","/path/to/folder")
```


## Changelog
- Renamed main. to simkit.
- added initSimkit as initialisation function to remove hardcoded login creds
- Changed test Cases to enable compilation on all devices
