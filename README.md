# SimKit
Plug-in for Neo4j containing user-defined procedures to perform Spectral Clustering through a single procedure or as individual procedures (Similarity Graph, Laplacian Eigendecomposed Graph, KMeans). Additional procedures available for evaluation with Average Silhouette Score and Adjusted Rand Index, visualization of matrices into String and saving them to CSV.

## [About](https://github.com/Graphistics/simkit/wiki)

## Methodology

## Procedure Stack

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
    parameter: "7",
    remove_column: "index,target"
})
```

### Create EigenGraph
```java
return simkit.nodePropertyEigen({
    node_label: "affinity_full_7_Iris",
    laplacian_type: "sym",
    number_of_eigenvectors: 3
})
```

### Kmean Clustering
```java
return simkit.kMeans({
    nodeSet: "eigen_sym_3_affinity_full_7_Iris",
    numberOfCentroid: "3",
    numberOfInteration: "100",
    distanceMeasure: "euclidean",
    originalSet: "Iris",
    overlook: "target,sepal_length,sepal_width,petal_length,petal_width",
    overlookOriginal: "target",
    useKmeanForSilhouette: false
})
```

### Calculate Adjusted rand index
```java
return simkit.adjustedRandIndex({
    nodeSet: "Iris",
    trueLabels: "target"
})
```

### Spectral Clustering
```java
return simkit.spectralClustering({
    node_label: "Iris",
    is_feature_based: true,
    distance_measure: "euclidean",
    graph_type: "full",
    parameter: "7",
    remove_columns: "index,target",
    laplacian_type: "sym",
    number_of_eigenvectors: 3,
    number_of_iterations: "100",
    distance_measure_kmean: "euclidean",
    target_column: "target",
    use_kmean_for_silhouette: false,
    seed: 42
})
```

### Visualize Matrix and save as CSV
```java
return simkit.getMatrix("affinity_full_7_Iris","adjacency","/path/to/folder")
```

## Features 

SimKit contains ... procedures, which read CSV files, map nodes, generate ... similarity graphs using .. different distance measures, generate laplacian eigendecomposed graph, perform K-means clustering, and validate the clustering with silhouette coefficient. 

## Procedures to
---> map data into Neo4j Nodes

---> generate Similarity graph

---> generate Laplacian eigendecomposed graph

---> perform K-means clustering and validate it with silhouette coefficient

---> calculate Adjusted rand index

---> perform Spectral clustering (feature-/graph-based clustering) and validate it with silhouette coefficient

---> visualize and save matrix (adjacency/X matrix) as CSV 

## Changelog
- Renamed main. to simkit.
- added initSimkit as initialisation function to remove hardcoded login creds
- Changed test Cases to enable compilation on all devices



