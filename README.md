# SimKit
Plug-in for Neo4j containing user-defined procedures 
  * Similarity graph
  * Laplacian eigendecomposition graph
  * K-means clustering

## [About](https://github.com/Graphistics/simkit/wiki)

## Methodology

## Procedure Stack

### Init SimKit
```java
return simkit.initSimKit('bolt://localhost:7687', 'neo4j', '123412345')
```

### Create Graph From Nodes
```java
return simkit.createGraphFromNodes('Iris', 'euclidean', 'full', '7', 'index, target')
```

### Create EigenGraph
```java
return simkit.createEigenGraph('full_7new', 'sym', 3)
```

### Kmean Clustering
```java
return simkit.kmean({
    nodeSet: "eigenGraph_rw_full_7new_3",
    numberOfCentroid: "3",
    numberOfInteration: "100",
    distanceMeasure: "euclidean",
    originalSet: "Iris",
    overlook: "target,sepal_length,sepal_width,petal_length,petal_width",
    overlookOriginal: "target",
    useKmeanForSilhouette: false
})
```

## Features 

SimKit contains ... procedures, which read CSV files, map nodes, generate ... similarity graphs using .. different distance measures, generate laplacian eigendecomposed graph, perform K-means clustering, and validate the clustering with silhouette coefficient. 

## Procedures to
---> map data into Neo4j Nodes

---> generate Similarity graph

---> generate Laplacian eigendecomposed graph

---> perform K-means clustering and validate it with silhouette coefficient

## Data [(check wiki)](https://github.com/Graphistics/simkit/wiki)

* [Dataset 1 - Iris ](https://www.kaggle.com/datasets/uciml/iris)

--> [K-means clustering on iris dataset](https://www.kaggle.com/code/khotijahs1/k-means-clustering-of-iris-dataset) \
--> [Spectral clustering on iris dataset](https://evoq-eval.siam.org/Portals/0/Publications/SIURO/Vol4/SPECTRAL_CLUSTERING_AND_VISUALIZATION.pdf?ver=2018-04-06-103239-570)

* [Dataset 2 - Madelon]

* [Dataset 3 - CORA]
 
* [Dataset 4 - 20newsgroups]


## Changelog
- Renamed main. to simkit.
- added initSimkit as initialisation function to remove hardcoded login creds
- Changed test Cases to enable compilation on all devices



