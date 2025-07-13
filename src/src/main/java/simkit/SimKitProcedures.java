package simkit;

import definition.EdgeList2;
import definition.NodeList2;
import eigendecomposed.EigenCalculation;
import eigendecomposed.MatrixCalculation;
import global.Neo4jGraphHandler;
import graph.DistanceMeasureNodes;
import graph.GraphTransform;
import graph.ReadCsvTestData;
import org.apache.commons.math3.linear.RealMatrix;
import org.ejml.simple.SimpleMatrix;
import org.neo4j.driver.*;
import org.neo4j.driver.Record;
import org.neo4j.driver.exceptions.AuthenticationException;
import org.neo4j.driver.exceptions.Neo4jException;
import org.neo4j.driver.exceptions.ServiceUnavailableException;
import org.neo4j.driver.util.Pair;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * This class is used to fetch nodes from graph database or from csv and call the functions to generate decision tree
 * with confusion matrix, generation time and prediction time for the output
 *
 * @author minh dung
 */

public class SimKitProcedures implements AutoCloseable {

    private static Driver driver;
    static List<Record> dataKey = new ArrayList<>();
    static ArrayList<String> mapNodeList = new ArrayList<String>();
    static List<Double> trueNodeLabels = new ArrayList<Double>();
    static List<Double> predictedNodeLabels = new ArrayList<Double>();

    private static String uri;
    private static String username;
    private static String password;

    /**
     * Creation of driver object using bolt protocol
     *
     * @param uri      Uniform resource identifier for bolto
     * @param user     Username
     * @param password Password
     */
    public SimKitProcedures(String uri, String user, String password) {
        driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password));

    }

    /**
     * Empty constructor
     */
    public SimKitProcedures() {
        driver = null;
    }

    /**
     * This function is used to initialize the SimKit
     *
     * @param uri  The URI of DB
     * @param user The Username for DB
     * @param pass The Password for DB
     * @throws Exception if connection to Neo4j fails
     * @author Jonas Heinzmann
     */
    @UserFunction
    public String initSimKit(@Name("URI") String uri, @Name("User") String user, @Name("Password") String pass) {
        String output;
        try {
            driver = GraphDatabase.driver(uri, AuthTokens.basic(user, pass));
// Attempt to establish a connection to verify credentials and availability
            driver.verifyConnectivity();
            output = "Initialization succesful";
// Store credentials if needed
            SimKitProcedures.uri = uri;
            SimKitProcedures.username = user;
            SimKitProcedures.password = pass;
        } catch (ServiceUnavailableException e) {
            output = "Connection error: Service unavailable";
        } catch (AuthenticationException e) {
            output = "Connection error: Authentication failed";
        } catch (Neo4jException e) {
            output = "Connection error: " + e.getMessage();
        } finally {
            if (driver != null) {
                driver.close();
            }
        }
        return output;
    }

    public Driver getDriver() {
        return driver;
    }

    /**
     * Close the driver object
     */
    @Override
    public void close() throws Exception {
        driver.close();
    }

    @UserFunction
    public String csvToGraph(@Name("data_path") String data_path, @Name("distance_measure") String distance_measure, @Name("graph_type") String graph_type, @Name("parameter") String parameter, @Name("remove_column") String remove_columns) throws Exception {


        String confusionMatrix = "";
        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {

            if (data_path == null && distance_measure == null) {
                throw new Exception("Missing data_path or distance measure type");
            } else {
                String graph_name = null;
                Double[][] adj_mat = null;
                String[] remove_list = remove_columns.split(",");
                List<String> remove_list_new = Arrays.stream(remove_list).collect(Collectors.toList());
                ReadCsvTestData readCsvTestData = new ReadCsvTestData(data_path);
// //ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNew(data_path,IndexColumn);
                ArrayList<NodeList2> node_properties_list = readCsvTestData.readCsvFileToMap(data_path);
// ArrayList<NodeList2> nodePropertiesList_copy = readCsvTestData.readCsvFileToMap(data_path);
                Double[][] distance_matrix = getDistanceMatrixFromNodes(distance_measure, node_properties_list, remove_list_new);

                if (graph_type.equals("full")) {
                    Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(distance_matrix, sigmas);
                    graph_name = graph_type.concat("_" + parameter);
                }
                if (graph_type.equals("eps")) {
                    Double epsilon = Double.parseDouble(parameter);
                    adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(distance_matrix, epsilon);
                    graph_name = graph_type.concat("_" + parameter);
                }
                if (graph_type.equals("knn")) {
                    Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateKNNGraph(distance_matrix, knn);
                    graph_name = graph_type.concat("_" + parameter);
                }
                if (graph_type.equals("mknn")) {
                    Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateMutualKNNGraph(distance_matrix, knn);
                    graph_name = graph_type.concat("_" + parameter);
                }
                ArrayList<EdgeList2> edge_list = GraphTransform.calculateEdgeList(node_properties_list, adj_mat);

                for (NodeList2 node : node_properties_list) {
                    Neo4jGraphHandler.createNodeGraph(graph_name.concat("new"), "Created nodes succesfully!\n", node, connector.getDriver());
                }

                for (int i = 0; i < edge_list.size(); i++) {
                    EdgeList2 edgeListDetail = edge_list.get(i);
                    if (edgeListDetail.getWeight() == 0.0) {
                        continue;
                    }
                    Neo4jGraphHandler.createRelationshipGraph(graph_name.concat("new"), "Created similarity graph succesfully!\n", edgeListDetail, connector.getDriver());

                }

            }
            return "Create fully connected graph successful, " + confusionMatrix;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Creates a Similarity Transform graph based on distance measure, affinity matrix type, hyperparameter, excluded columns.
     *
     * @param params Set of parameters
     * @return String indicating the success of the graph creation and creates Graph with Nodes and Relationships in Neo4j.
     * @throws Exception If an error occurs while creating the Laplacian graph in Neo4j.
     */
// @UserFunction
// public String nodePropertyToGraph(@Name("params") Map<String, Object> params) throws Exception {
//
// String label = (String) params.getOrDefault("label", "Iris");
// String distance_measure = (String) params.getOrDefault("distance_measure", "euclidean");
// String graph_type = (String) params.getOrDefault("graph_type", "full");
// String parameter = (String) params.getOrDefault("parameter", "7");
// String remove_columns = (String) params.getOrDefault("remove_column", "index,target");
//
// StringBuilder output_string = new StringBuilder("Graph Data: ");
// try ( SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password) )
// {
//
//
// if(label == null && distance_measure == null) {
// throw new Exception("Missing data_path or distance measure type");
// }else {
// String graph_name = "affinity_";
// //ArrayList<NodeList2> node_properties_list = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(label, connector.getDriver());
// org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodeData = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(label, connector.getDriver());
// ArrayList<NodeList2> node_properties_list = nodeData.getLeft();
// // ArrayList<NodeList2> nodePropertiesList_copy = Neo4jGraphHandler.retrieveNodeListFromNeo4j(label, connector.getDriver());
// String[] remove_list = remove_columns.split(",");
// List<String> remove_list_new = Arrays.stream(remove_list).collect(Collectors.toList());
//// Double[][] distance_matrix = GraphTransform.euclideanDistance(node_properties_list);
// Double[][] distance_matrix = getDistanceMatrixFromNodes(distance_measure,node_properties_list,remove_list_new);
// Double[][] adj_mat = null;
//
// if(graph_type.equals("full")) {
// Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(distance_matrix,parameter);
// adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(distance_matrix,sigmas);
// graph_name = graph_name.concat(graph_type + "_" + parameter.replace(".", "_"));
// }
// else if(graph_type.equals("eps")) {
// Double epsilon = Double.parseDouble(parameter);
// adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(distance_matrix,epsilon);
// graph_name = graph_name.concat(graph_type + "_" + parameter.replace(".", "_"));
//
// }
// else if(graph_type.equals("knn")) {
// int[][] knn = ReadCsvTestData.calculateKNNIndices(distance_matrix,parameter);
// adj_mat = ReadCsvTestData.calculateKNNGraphWithIndices(distance_matrix,knn);
// graph_name = graph_name.concat(graph_type + "_" + parameter.replace(".", "_"));
//
// }
// else if(graph_type.equals("mknn")) {
// Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix,parameter);
// adj_mat = ReadCsvTestData.calculateMutualKNNGraph(distance_matrix,knn);
// graph_name = graph_name.concat(graph_type + "_" + parameter.replace(".", "_"));
// } else {
// throw new Exception("Invalid graph_type specified.");
// }
//
// output_string.append("\n\nAdjacency Matrix:\n").append(doubleToString(adj_mat));
//
// ArrayList<EdgeList2> edge_list = GraphTransform.calculateEdgeList(node_properties_list,adj_mat);
// graph_name += "_" + label;
//// Neo4jGraphHandler.bulkDeleteNodesWithBatching(graph_name, connector.getDriver());
//
//
// //for (edge_list edgeListDetail : edge_list) {
// for (NodeList2 node : node_properties_list) {
// Neo4jGraphHandler.createNodeGraph(graph_name, "Created nodes succesfully!\n", node, connector.getDriver());
// }
//
// for (int i = 0; i < edge_list.size(); i++) {
// EdgeList2 edgeListDetail = edge_list.get(i);
// if(edgeListDetail.getWeight()==0.0){
// continue;
// }
// Neo4jGraphHandler.createRelationshipGraph(graph_name, "Created similarity graph succesfully!\n", edgeListDetail, connector.getDriver());
//
// }
//
// }
// return "Create fully connected graph successful, " + output_string;
// } catch (Exception e) {
// throw new RuntimeException(e);
// }
//
// }
    public static Double[][] getDistanceMatrixFromNodes(String distance_measure, ArrayList<NodeList2> node_properties_list, List<String> remove_list) {
        Double[][] distance_matrix = null;

        switch (distance_measure) {
            case "euclidean":
                distance_matrix = DistanceMeasureNodes.euclideanDistance(node_properties_list, remove_list);
                break;
            case "manhattan":
                distance_matrix = DistanceMeasureNodes.manhattanDistance(node_properties_list, remove_list);
                break;
            case "canberra":
                distance_matrix = DistanceMeasureNodes.canberraDistance(node_properties_list, remove_list);
                break;
            case "cosine":
                distance_matrix = DistanceMeasureNodes.cosineSimilarity(node_properties_list, remove_list);
                break;
            case "jaccard":
                distance_matrix = DistanceMeasureNodes.jaccardCoefficient(node_properties_list, remove_list);
                break;
            case "bray_curtis":
                distance_matrix = DistanceMeasureNodes.brayCurtisDistance(node_properties_list, remove_list);
                break;
            default:
                System.out.println("Invalid distance measure type");
        }

        return distance_matrix;

    }

    @UserFunction
    public String nodePropertyToGraph(@Name("params") Map<String, Object> params) throws Exception {

        String label = (String) params.getOrDefault("label", "Iris");
        String distance_measure = (String) params.getOrDefault("distance_measure", "euclidean");
        String graph_type = (String) params.getOrDefault("graph_type", "full");
        String parameter = (String) params.getOrDefault("parameter", "7");
        String remove_columns = (String) params.getOrDefault("remove_column", "index,target");
        boolean create_new_graph = (boolean) params.getOrDefault("create_new_graph", false);

        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {


            if (label == null && distance_measure == null) {
                throw new Exception("Missing data_path or distance measure type");
            } else {
                ArrayList<NodeList2> node_properties_list;
                String property_names = "";
                String identifier = "";

                org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                node_properties_list = node_data.getLeft();
                identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);

                property_names = node_data.getRight();

                String[] remove_list = remove_columns.split(",");
                List<String> remove_list_new = Arrays.stream(remove_list).collect(Collectors.toList());

                Double[][] distance_matrix = getDistanceMatrixFromNodes(distance_measure, node_properties_list, remove_list_new);
                Double[][] adj_mat = null;

                if (graph_type.equals("full")) {
                    Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(distance_matrix, sigmas);
                } else if (graph_type.equals("eps")) {
                    Double epsilon = Double.parseDouble(parameter);
                    adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(distance_matrix, epsilon);
                } else if (graph_type.equals("knn")) {
                    int[][] knn = ReadCsvTestData.calculateKNNIndices(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateKNNGraphWithIndices(distance_matrix, knn);
                } else if (graph_type.equals("mknn")) {
                    Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateMutualKNNGraph(distance_matrix, knn);
                } else {
                    throw new Exception("Invalid graph_type specified.");
                }

                ArrayList<EdgeList2> edge_list = GraphTransform.calculateEdgeList(node_properties_list, adj_mat);

                if (create_new_graph) {
                    String graph_name = "affinity_" + graph_type + "_" + parameter.replace(".", "_") + "_" + label;
                    Neo4jGraphHandler.bulkDeleteNodesWithBatching(graph_name, connector.getDriver(), identifier);
                    Neo4jGraphHandler.bulkCreateNodes(graph_name, node_properties_list, connector.getDriver(), identifier);
                    Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(graph_name, edge_list, connector.getDriver(), identifier);
                    return "Created new graph with nodes and relationships!";
                } else {
// Only create relationships between existing nodes
                    Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(label, edge_list, connector.getDriver(), identifier);
                    return "Created edges between existing nodes!";
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    @UserFunction
    public String nodePropertyEigen(@Name("params") Map<String, Object> params) throws Exception {

        String label = (String) params.getOrDefault("label", "affinity_full_7_Iris");
        String laplacian_type = (String) params.getOrDefault("laplacian_type", "sym");
        Double number_of_eigenvectors = ((Number) params.getOrDefault("number_of_eigenvectors", 3)).doubleValue();
        if (number_of_eigenvectors == null || number_of_eigenvectors <= 0) {
            throw new Exception("Invalid input: number_of_eigenvectors must be positive");
        }

        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (label == null) {
                throw new Exception("No nodes found with the label: " + label);
            }

// Retrieve node and edge list from Neo4j and compute matrices
            org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
            ArrayList<NodeList2> node_properties_list = node_data.getLeft();
            String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);

            ArrayList<EdgeList2> edge_list = Neo4jGraphHandler.retrieveEdgeList(label, connector.getDriver());

            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list, node_properties_list);
            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);

// Calculate eigenvalues and eigenvectors
            Object eigen_result_obj;
            try {
                eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);
            } catch (IllegalArgumentException e) {
                throw new Exception("Invalid input: " + e.getMessage());
            } catch (RuntimeException e) {
                throw new Exception("Error in eigen decomposition: " + e.getMessage());
            }

// Cast the result to EigenResult
            EigenCalculation.EigenResult eigen_result = (EigenCalculation.EigenResult) eigen_result_obj;

// Prepare a list of property maps for each node
            List<Map<String, Object>> eigenvectorProperties = new ArrayList<>();
            for (int i = 0; i < node_properties_list.size(); i++) {
                Map<String, Object> props = new HashMap<>();
                for (int j = 0; j < eigen_result.X.numCols(); j++) {
                    props.put("eigenvector_" + j, eigen_result.X.get(i, j));
                }
                eigenvectorProperties.add(props);
            }

// Update existing nodes with eigenvector properties
            Neo4jGraphHandler.updateNodeProperties(
                    label,
                    node_properties_list,
                    eigenvectorProperties,
                    connector.getDriver(),
                    identifier
            );

            return "Updated existing nodes with eigenvector properties!";

// --- BEGIN: Optionally regenerate edges based on eigen decomposition ---
// in case the structure should differ from the original.
/*
ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_properties_list, eigen_result.X, edge_list);
Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(label, edge_list_eigen, connector.getDriver(), identifier);
// return "Regenerated edges based on eigenvectors!";
*/
// --- END: Optionally regenerate edges based on eigen decomposition ---

// // Create a new graph name and remove existing nodes with that label
// String graph_name = "eigen_" + laplacian_type + "_" + Math.round(number_of_eigenvectors) + "_" + label;
// Neo4jGraphHandler.bulkDeleteNodesWithBatching(graph_name, connector.getDriver(), identifier);
// Neo4jGraphHandler.bulkCreateNodesWithEigen(graph_name, node_properties_list, eigen_result.X, connector.getDriver(), identifier);
// return "Create eigendecomposed graph successful!";

        } catch (Neo4jException e) {
            throw new Exception("Error creating Eigendecomposed graph: " + e.getMessage());
        }
    }

    @UserFunction
    public String mapNodes(@Name("label") String label, @Name("overlook") String overlook) throws Exception {
        String list_of_data = "";
        String[] overlook_array = new String[0];
        mapNodeList.clear();
        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (!overlook.isEmpty()) {
                overlook_array = overlook.split(",");
            }
            queryData(label);
            for (Record key : dataKey) {
                List<Pair<String, Value>> values = key.fields();
                for (Pair<String, Value> node_values : values) {
                    if ("n".equals(node_values.key())) {
                        Value value = node_values.value();
                        String valueOfNode = getNodeValues(value, overlook_array);
                        mapNodeList.add(valueOfNode);
// list_of_data = list_of_data + valueOfNode + " | ";
                        list_of_data = mapNodeList.toString();
                    }
                }
            }
        }
        return "Map all node data: " + list_of_data;
    }

    /**
     * Procedure for k-means clustering and visualization in neo4j
     *
     * @param params Type of node
     * @return Cluster result and visualization
     * @throws Exception if an error occurs during clustering
     */
    @UserFunction
    @Description("K-means clustering function")
    public double kMeans(@Name("params") Map<String, Object> params) throws Exception {

        predictedNodeLabels.clear();

        String label = (String) params.getOrDefault("label", "eigen_sym_3_affinity_full_7_Iris");
        String number_of_centroids = (String) params.getOrDefault("number_of_centroids", "3");
        String number_of_iterations = (String) params.getOrDefault("number_of_iterations", "100");
        String distance_measure = (String) params.getOrDefault("distance_measure", "euclidean");
        String original_set = (String) params.getOrDefault("original_set", "Iris");
        String overlook = (String) params.getOrDefault("overlook", "target,sepal_length,sepal_width,petal_length,petal_width");
        String overlook_original = (String) params.getOrDefault("overlook_original", "target");
        boolean silhouette = (Boolean) params.getOrDefault("silhouette", false);
        int seed = ((Number) params.getOrDefault("seed", 42)).intValue();
        boolean create_new_graph = (boolean) params.getOrDefault("create_new_graph", false);

        List<String> allowedMeasures = Arrays.asList("manhattan", "cosine", "bray-curtis", "euclidean");
        if (!allowedMeasures.contains(distance_measure.toLowerCase())) {
            throw new IllegalArgumentException("Invalid distance measure: " + distance_measure);
        }

        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            int numCentroids = Integer.parseInt(number_of_centroids);
            int numIterations = Integer.parseInt(number_of_iterations);

            // This list preserves the original order of nodes from the database query
            ArrayList<String> mapNodeList = parseNodeValues(label, overlook.split(","));
            if (mapNodeList == null || mapNodeList.isEmpty()) {
                throw new Exception("No nodes found with the label: " + label);
            }
            ArrayList<String> mapNodeOriginalList = parseNodeValues(original_set, overlook_original.split(","));

            HashMap<String, ArrayList<String>> kmeanAssign = Unsupervised.KmeanClust(
                    mapNodeList, numCentroids, numIterations, distance_measure, false, new ArrayList<>(), seed
            );

            HashMap<String, ArrayList<String>> cleanedKmeanAssign = Unsupervised.removeIndexAndId(kmeanAssign);

            // ========================= START: CORRECTED LOGIC =========================
            // This section ensures predictedNodeLabels is populated in the correct order, fixing the ARI bug.

            // 1. Create a reverse map from a cleaned node's string representation to its cluster ID.
            Map<String, Double> nodeToClusterIdMap = new HashMap<>();
            double clusterId = 1.0; // Start cluster numbering from 1
            for (ArrayList<String> nodesInCluster : cleanedKmeanAssign.values()) {
                for (String cleanedNodeValue : nodesInCluster) {
                    nodeToClusterIdMap.put(cleanedNodeValue, clusterId);
                }
                clusterId++;
            }

            // 2. Populate the global predictedNodeLabels list.
            // Iterate over the original, ordered node list. For each node, clean it
            // and find its corresponding cluster ID from the map.
            predictedNodeLabels.clear(); // Ensure list is empty before populating
            for (String originalNodeStr : mapNodeList) {
                String cleanedNodeStr = Unsupervised.removeIndexAndIdFromString(originalNodeStr);

                if (nodeToClusterIdMap.containsKey(cleanedNodeStr)) {
                    predictedNodeLabels.add(nodeToClusterIdMap.get(cleanedNodeStr));
                } else {
                    // This case handles nodes that might not have been clustered, which is unlikely in k-means.
                    predictedNodeLabels.add(-1.0); // Use -1 as a marker for unassigned nodes.
                }
            }
            // ========================== END: CORRECTED LOGIC ==========================

            double averageSilhouetteCoefficientValue;
            if (silhouette) {
                averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(cleanedKmeanAssign, distance_measure);
            } else {
                HashMap<String, ArrayList<String>> mappedNodes = Unsupervised.replaceValuesWithOriginalSet(kmeanAssign, mapNodeOriginalList);
                averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(mappedNodes, distance_measure);
            }

            if (create_new_graph) {
                processClusters(connector, label, cleanedKmeanAssign, distance_measure);
            } else {
                // Add cluster property to existing nodes
                org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                ArrayList<NodeList2> node_properties_list = node_data.getLeft();
                String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);

                List<Map<String, Object>> clusterProperties = new ArrayList<>();
                for (int i = 0; i < node_properties_list.size(); i++) {
                    Map<String, Object> props = new HashMap<>();
                    // Use the correctly ordered predicted labels list
                    props.put("kmeans_cluster", predictedNodeLabels.get(i).intValue());
                    clusterProperties.add(props);
                }

                Neo4jGraphHandler.updateNodeProperties(
                        label,
                        node_properties_list,
                        clusterProperties,
                        connector.getDriver(),
                        identifier
                );
            }
            return averageSilhouetteCoefficientValue;
        }
    }


    /**
     * Parses node values from the dataset based on overlook fields.
     */
    private ArrayList<String> parseNodeValues(String label, String[] overlookFields) throws Exception {
        ArrayList<String> node_list = new ArrayList<>();
        queryData(label);

        for (Record record : dataKey) {
            for (Pair<String, Value> pair : record.fields()) {
                if ("n".equals(pair.key())) {
                    String value = getNodeValues(pair.value(), overlookFields);
                    node_list.add(value);
                }
            }
        }
        return node_list;
    }

    /**
     * Processes clusters and connects nodes based on the k-means result.
     * This method no longer populates the predictedNodeLabels list.
     */
    private void processClusters(SimKitProcedures connector, String label,
                                 HashMap<String, ArrayList<String>> kmeanAssign,
                                 String distance_measure) throws Exception {
        DecimalFormat decimalFormat = new DecimalFormat("#.###");
        for (String centroid : kmeanAssign.keySet()) {
            ArrayList<String> clusterNodes = kmeanAssign.get(centroid);

            for (String clusterNode : clusterNodes) {
                // The predictedNodeLabels list is now populated correctly in the main kMeans method.
                // This method is now only for graph visualization purposes.
                double distance = Unsupervised.calculateDistance(clusterNode, centroid, distance_measure);
                String formattedDistance = decimalFormat.format(distance);

                connectNodes(label, "create relationship in kmean node", centroid, clusterNode, Double.parseDouble(formattedDistance), getDriver());
            }
        }
    }


    /**
     * Procedure for calculating adjusted rand index
     *
     * @param params Set of parameters (node label, true label column's name)
     * @return Adjusted rand index score
     * @throws Exception
     */
    @UserFunction
// public String adjustedRandIndex(@Name("params") Map<String, Object> params) throws Exception {
    public Map<String, Object> adjustedRandIndex(@Name("params") Map<String, Object> params) throws Exception {

        String label = (String) params.getOrDefault("label", "Iris");
        String true_labels = (String) params.getOrDefault("true_labels", "target");

        if (predictedNodeLabels.size() == 0) {
            throw new Exception("Predicted Labels is null or empty. " +
                    "Details: predictedNodeLabels=" + predictedNodeLabels.toString() + ", " +
                    "label=" + label + ", true_labels=" + true_labels + ", params=" + params.toString());
        } else {
            String list_of_data = "";
            Double adjustedRandIndexValue = 0.0;
            trueNodeLabels.clear();
            List<String> stringTrueNodeLabelsList = new ArrayList<String>();
            try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
                queryData(label);
                for (Record key : dataKey) {
                    List<Pair<String, Value>> values = key.fields();
                    for (Pair<String, Value> node_values : values) {
                        if ("n".equals(node_values.key())) {
                            Value value = node_values.value();
                            StringBuilder nodeLabel = new StringBuilder();
                            for (String nodeKey : value.keys()) {
                                if (nodeKey.equals(true_labels)) {
                                    try {
                                        double num = Double.parseDouble(String.valueOf(value.get(nodeKey)));
                                        trueNodeLabels.add(num);
                                        list_of_data = list_of_data + num;
// nodeLabel.append(getStringValue(nodeLabel)).append(nodeKey).append(":").append(value.get(nodeKey));
                                    } catch (NumberFormatException e) {
                                        System.out.println(value.get(nodeKey) + " is not a number.");
                                        stringTrueNodeLabelsList.add(String.valueOf(value.get(nodeKey)));
                                    }
                                }
                            }
                        }
                    }
                }
                if (stringTrueNodeLabelsList.size() != 0) {
                    trueNodeLabels = convertStringLabels(stringTrueNodeLabelsList);
                }

                if (trueNodeLabels.size() != predictedNodeLabels.size()) {
                    throw new Exception("true labels size: " + trueNodeLabels.size() + " and predicted labels:" + predictedNodeLabels.size() + " does not have the same size");
                } else {
                    adjustedRandIndexValue = calculateAdjustedRandIndex(trueNodeLabels, predictedNodeLabels);
                }
            }

            Map<String, Object> result = new HashMap<>();
            result.put("adjustedRandIndex", adjustedRandIndexValue);
            result.put("true_labels", trueNodeLabels);
            result.put("predictedLabels", predictedNodeLabels);

            return result;

// return adjustedRandIndexValue;
// return "ajusted rand index of " + label + " is: " + adjustedRandIndexValue ;
// return "true labels: " + trueNodeLabels + " -- predicted labels: " + predictedNodeLabels + " -- adjustedRandIndexValue: " + adjustedRandIndexValue;
        }
    }


    @UserFunction
    @Description("Calculate the mean of the Silhouette Coefficients for all points")
    public String averageSilhouetteCoefficient(
            @Name("label") String label,
            @Name("number_of_centroids") String number_of_centroids,
            @Name("number_of_iterations") String number_of_iterations,
            @Name("distance_measure") String distance_measure,
            @Name("seed") Number seed) throws Exception {

        if (label == null) {
            return null;
        }

// ✅ Validate distance_measure early
        List<String> allowedMeasures = Arrays.asList("manhattan", "cosine", "bray-curtis", "euclidean");
        if (!allowedMeasures.contains(distance_measure.toLowerCase())) {
            throw new IllegalArgumentException("Unsupported distance measure: " + distance_measure);
        }

        String averageSilhouetteCoefficientString = "The average Silhouette Coefficient value is: ";
        int numberOfCentroidInt = Integer.parseInt(number_of_centroids);
        int numberOfInterationInt = Integer.parseInt(number_of_iterations);
        ArrayList<String> debug = new ArrayList<>();

        HashMap<String, ArrayList<String>> kmeanAssign = Unsupervised.KmeanClust(
                mapNodeList, numberOfCentroidInt, numberOfInterationInt, distance_measure, false, debug, (int) seed);

        double averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(kmeanAssign, distance_measure);
        return averageSilhouetteCoefficientString + averageSilhouetteCoefficientValue;
    }


    /**
     * This function is used to query the data from graph database
     *
     * @param label type of node
     */
    public void queryData(final String label) {
        try (Session session = getDriver().session()) {
            String greeting = session.writeTransaction(new TransactionWork<String>() {
                @Override
                public String execute(Transaction tx) {
                    Result result = tx.run("MATCH (n:" + label + ") RETURN n");
                    dataKey = result.list();
                    return "Query Successful";
                }
            });
        }
    }

// public void connectNodes(final String label, final String message, final String nodeCentroid, final String nodeCluster, final double distance) throws Exception
// {
// final String name = "kmean";
// try ( Session session = getDriver().session() )
// {
//
// log.info("kmeans connection starts....");
// String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(getDriver(), label);
// String greeting = session.writeTransaction( new TransactionWork<String>()
// {
// @Override
// public String execute( Transaction tx )
// {
// // First, find and delete the existing node if it exists
// String deleteQuery = "MATCH (a:Clustering_" + label + " {" + nodeCentroid + "}) " +
// "DETACH DELETE a";
// //tx.run(deleteQuery);
// // Then create the new nodes and relationship
// String createQuery = "MERGE (a:Clustering_" + label + " {" + nodeCentroid + "}) " +
// "MERGE (b:Clustering_" + label + " {" + nodeCluster + "}) " +
// "MERGE (a)-[r:link]->(b) " +
// "SET r.distance = " + distance + " " +
// "RETURN a.message";
//

    /// / String createQuery = "MERGE (a:Clustering_" + label + " {" + identifier + ": '" + nodeCentroid + "'}) " +
    /// / "MERGE (b:Clustering_" + label + " {" + identifier + ": '" + nodeCluster + "'}) " +
    /// / "MERGE (a)-[r:link]->(b) " +
    /// / "SET r.distance = " + distance + " " +
    /// / "RETURN a";
//
//
// Result result = tx.run(createQuery);
// return result.single().get(0).asString();
// }
// } );
// }
// }
    public void connectNodes(final String label, final String message, final String nodeCentroid, final String nodeCluster, final double distance, Driver driver) throws Exception {
        final String name = "kmean";
        try (Session session = driver.session()) {
            String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(getDriver(), label);

            String greeting = session.writeTransaction(new TransactionWork<String>() {
                @Override
                public String execute(Transaction tx) {

// Resolve identifier for dynamic node matching
                    String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(getDriver(), label);


// First, find and delete the existing node if it exists
                    String deleteQuery = "MATCH (a:Clustering_" + label + " {" + nodeCentroid + "}) " +
                            "DETACH DELETE a";
//tx.run(deleteQuery);
// Then create the new nodes and relationship
                    String createQuery = "MERGE (a:Clustering_" + label + " {" + nodeCentroid + "}) " +
                            "MERGE (b:Clustering_" + label + " {" + nodeCluster + "}) " +
                            "MERGE (a)-[r:link]->(b) " +
                            "SET r.distance = " + distance + " " +
                            "RETURN a.message";

                    Result result = tx.run(createQuery);
                    return result.single().get(0).asString();
                }
            });
        }
    }

    public String getNodeValues(Value value, String[] overlook_array) {
        StringBuilder valueOfNode = new StringBuilder();
        for (String nodeKey : value.keys()) {
            if (overlook_array.length > 0 && Arrays.asList(overlook_array).contains(nodeKey)) {
                continue;
            }
            try {
                double num = Double.parseDouble(String.valueOf(value.get(nodeKey)));
                if (value.get(nodeKey).getClass().equals(String.class)) {
                    valueOfNode.append(getStringValue(valueOfNode)).append(nodeKey).append(":").append(value.get(nodeKey));
                } else {
                    valueOfNode.append(getStringValue(valueOfNode)).append(nodeKey).append(":").append(value.get(nodeKey));
                }
            } catch (NumberFormatException e) {
                System.out.println(value.get(nodeKey) + " is not a number.");
            }
        }
        return valueOfNode.toString();
    }

    private String getStringValue(StringBuilder valueOfNode) {
        return valueOfNode.length() > 0 ? ", " : "";
    }

    public static List<Double> convertStringLabels(List<String> strings) {
        Map<String, Double> labelMap = new HashMap<>();
        List<Double> labels = new ArrayList<>();

        double currentLabel = 0.0;
        for (String s : strings) {
            if (!labelMap.containsKey(s)) {
                labelMap.put(s, currentLabel++);
            }
            labels.add(labelMap.get(s));
        }

        return labels;
    }

    public static double calculateAdjustedRandIndex(List<Double> trueLabels, List<Double> predictedLabels) {
        if (trueLabels.size() != predictedLabels.size()) {
            throw new IllegalArgumentException("Input lists must have the same length");
        }

        int n = trueLabels.size();
        Map<Double, Map<Double, Double>> contingencyTable = new HashMap<>();
        Map<Double, Double> trueLabelCounts = new HashMap<>();
        Map<Double, Double> predictedLabelCounts = new HashMap<>();

// Build the contingency table and label counts
        for (int i = 0; i < n; i++) {
            double trueLabel = trueLabels.get(i);
            double predictedLabel = predictedLabels.get(i);

            contingencyTable.computeIfAbsent(trueLabel, k -> new HashMap<>());
            contingencyTable.get(trueLabel).merge(predictedLabel, 1.0, Double::sum);

            trueLabelCounts.merge(trueLabel, 1.0, Double::sum);
            predictedLabelCounts.merge(predictedLabel, 1.0, Double::sum);
        }

        double a = 0.0; // Number of pairs in the same cluster in both true and predicted
        for (Map<Double, Double> row : contingencyTable.values()) {
            for (double count : row.values()) {
                a += count * (count - 1) / 2.0;
            }
        }

        double b = 0.0; // Number of pairs in the same cluster in trueLabels
        for (double count : trueLabelCounts.values()) {
            b += count * (count - 1) / 2.0;
        }

        double c = 0.0; // Number of pairs in the same cluster in predictedLabels
        for (double count : predictedLabelCounts.values()) {
            c += count * (count - 1) / 2.0;
        }

        double totalPairs = n * (n - 1) / 2.0;
        double expectedIndex = (b * c) / totalPairs;
        double maxIndex = 0.5 * (b + c);
        double adjustedRandIndex = (a - expectedIndex) / (maxIndex - expectedIndex);

        return adjustedRandIndex;
    }

    /**
     * Procedure for spectral clustering and visualization in neo4j
     *
     * @param params Set of parameters
     * @return String indicating the successful creation of graphs and results of clustering.
     * @throws Exception If any error occurs during the spectral clustering process.
     */
    @UserFunction
    public double spectralClustering(@Name("params") Map<String, Object> params) throws Exception {
        String label = (String) params.getOrDefault("label", "Iris");
        Boolean is_feature_based = (Boolean) params.getOrDefault("is_feature_based", true);
        String distance_measure = (String) params.getOrDefault("distance_measure", "euclidean");
        String graph_type = (String) params.getOrDefault("graph_type", "full");
        String parameter = (String) params.getOrDefault("parameter", "7");
        String remove_columns = (String) params.getOrDefault("remove_columns", "index,target");
        String laplacian_type = (String) params.getOrDefault("laplacian_type", "sym");
        Double number_of_eigenvectors = ((Number) params.getOrDefault("number_of_eigenvectors", 3)).doubleValue();
        String number_of_iterations = (String) params.getOrDefault("number_of_iterations", "100");
        String distance_measure_kmean = (String) params.getOrDefault("distance_measure_kmean", "euclidean");
        String target_column = (String) params.getOrDefault("target_column", "target");
        Boolean silhouette = (Boolean) params.getOrDefault("silhouette", false);
        int seed = ((Number) params.getOrDefault("seed", 42)).intValue();
        boolean create_new_graph = (boolean) params.getOrDefault("create_new_graph", false);

        predictedNodeLabels.clear();

        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (label == null) {
                throw new Exception("Missing node label");
            }

// String graph_name = "affinity_" + graph_type + "_" + parameter.replace(".", "_") + "_" + label;
            ArrayList<NodeList2> node_properties_list;
            String property_names = "";
            String identifier = "";
            ArrayList<EdgeList2> edge_list_2;

// Step 1: Create affinity graph
            if (is_feature_based) {
                org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                node_properties_list = node_data.getLeft();
                identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);

                property_names = node_data.getRight();

                String[] remove_list = remove_columns.split(",");
                List<String> remove_list_new = Arrays.stream(remove_list).collect(Collectors.toList());
                Double[][] distance_matrix = getDistanceMatrixFromNodes(distance_measure, node_properties_list, remove_list_new);
                Double[][] adj_mat;
                if (graph_type.equals("full")) {
                    Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(distance_matrix, sigmas);
                } else if (graph_type.equals("eps")) {
                    Double epsilon_value = Double.parseDouble(parameter);
                    adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(distance_matrix, epsilon_value);
                } else if (graph_type.equals("knn")) {
                    int[][] knn = ReadCsvTestData.calculateKNNIndices(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateKNNGraphWithIndices(distance_matrix, knn);
                } else if (graph_type.equals("mknn")) {
                    Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateMutualKNNGraph(distance_matrix, knn);
                } else {
                    throw new Exception("Invalid graph_type specified.");
                }
                ArrayList<EdgeList2> edge_list = GraphTransform.calculateEdgeList(node_properties_list, adj_mat);
                edge_list_2 = edge_list;
// Neo4jGraphHandler.bulkDeleteNodesWithBatching(graph_name, connector.getDriver(), identifier);
// Neo4jGraphHandler.bulkCreateNodes(graph_name, node_properties_list, connector.getDriver(), identifier);
// Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(graph_name, edge_list, connector.getDriver(), identifier);

            } else {
                org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                node_properties_list = node_data.getLeft();
                if (node_properties_list.isEmpty()) {
                    throw new Exception("Affinity graph exists but no nodes retrieved.");
                }

                property_names = node_data.getRight();
                identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);
                edge_list_2 = Neo4jGraphHandler.retrieveEdgeList(label, connector.getDriver());
            }

// Step 2: Compute Laplacian matrix
            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list_2, node_properties_list);
            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
            Object eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);

            if (eigen_result_obj instanceof String) {
                throw new Exception((String) eigen_result_obj);
            }
            EigenCalculation.EigenResult eigen_result = (EigenCalculation.EigenResult) eigen_result_obj;

            ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_properties_list, eigen_result.X, edge_list_2);

// String graph_name_eigen = "eigen_" + laplacian_type + "_" + Math.round(number_of_eigenvectors) + "_" + graph_name;

// Neo4jGraphHandler.bulkDeleteNodesWithBatching(graph_name_eigen, connector.getDriver(), identifier);
// Neo4jGraphHandler.bulkCreateNodesWithEigen(graph_name_eigen, node_properties_list, eigen_result.X, connector.getDriver(), identifier);
// Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(graph_name_eigen, edge_list_eigen, connector.getDriver(), identifier);

// --- Only update existing nodes with eigenvector properties ---
            List<Map<String, Object>> eigenvectorProperties = new ArrayList<>();
            for (int i = 0; i < node_properties_list.size(); i++) {
                Map<String, Object> props = new HashMap<>();
                for (int j = 0; j < eigen_result.X.numCols(); j++) {
                    props.put("eigenvector_" + j, eigen_result.X.get(i, j));
                }
                eigenvectorProperties.add(props);
            }
            Neo4jGraphHandler.updateNodeProperties(
                    label,
                    node_properties_list,
                    eigenvectorProperties,
                    connector.getDriver(),
                    identifier
            );

// --- BEGIN: Optionally regenerate edges based on eigen decomposition ---
// in case the edges tructure should differ from the original.
/*
ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_properties_list, eigen_result.X, edge_list);
Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(label, edge_list_eigen, connector.getDriver(), identifier);
// return "Regenerated edges based on eigenvectors!";
*/
// --- END: Optionally regenerate edges based on eigen decomposition ---

// Step 3: Perform clustering
            String number_of_clusters = Integer.toString(number_of_eigenvectors.intValue());
            String[] properties_array = property_names.split(",");
            StringBuilder filtered_properties = new StringBuilder();
            for (String property : properties_array) {
                if (!property.equalsIgnoreCase(identifier) && !property.equalsIgnoreCase("target")) {
                    if (filtered_properties.length() > 0) {
                        filtered_properties.append(",");
                    }
                    filtered_properties.append(property);
                }
            }

// System.out.println("filtered_properties: " + filtered_properties);
            double kmean_result = kMeans(Map.of(
                    "label", label,
                    "number_of_centroids", number_of_clusters,
                    "number_of_iterations", number_of_iterations,
                    "distance_measure", distance_measure_kmean,
                    "original_set", label,
                    "overlook", target_column + "," + filtered_properties.toString(),
                    "overlook_original", target_column,
                    "silhouette", silhouette,
                    "seed", seed,
                    "create_new_graph", create_new_graph
            ));

// Return metrics
            return kmean_result;
        }
    }

    /**
     * Procedure for experiments to debug spectral clustering procedures
     *
     * @param params Set of parameters
     * @return String indicating the successful creation of graphs and results of clustering.
     * @throws Exception If any error occurs during the spectral clustering process.
     */
    @UserFunction
    public Map<String, Object> experimentalSpectralClustering(@Name("params") Map<String, Object> params) throws Exception {
        String label = (String) params.getOrDefault("label", "Iris");
        boolean duplicate_node_set = (boolean) params.getOrDefault("duplicate_node_set", false);
        Boolean is_feature_based = (Boolean) params.getOrDefault("is_feature_based", true);
        String distance_measure = (String) params.getOrDefault("distance_measure", "euclidean");
        String graph_type = (String) params.getOrDefault("graph_type", "full");
        String parameter = (String) params.getOrDefault("parameter", "7");
        String remove_columns = (String) params.getOrDefault("remove_columns", "index,target");
        String laplacian_type = (String) params.getOrDefault("laplacian_type", "sym");
        Double number_of_eigenvectors = ((Number) params.getOrDefault("number_of_eigenvectors", 3)).doubleValue();
        String number_of_iterations = params.getOrDefault("number_of_iterations", 100).toString();
        String distance_measure_kmean = (String) params.getOrDefault("distance_measure_kmean", "euclidean");
        String target_column = (String) params.getOrDefault("target_column", "target");
        Boolean silhouette = (Boolean) params.getOrDefault("silhouette", false);
        int seed = ((Number) params.getOrDefault("seed", 42)).intValue();
        boolean create_new_graph = (boolean) params.getOrDefault("create_new_graph", false);

        try {
            if (!(params.getOrDefault("number_of_eigenvectors", 3) instanceof Number)) {
                throw new IllegalArgumentException("Invalid type for parameter: number_of_eigenvectors. Expected a number.");
            }
            if (!(params.getOrDefault("seed", 42) instanceof Number)) {
                throw new IllegalArgumentException("Invalid type for parameter: seed. Expected a number.");
            }
            if (!(params.getOrDefault("is_feature_based", true) instanceof Boolean)) {
                throw new IllegalArgumentException("Invalid type for parameter: is_feature_based. Expected a boolean.");
            }
            if (!(params.getOrDefault("silhouette", false) instanceof Boolean)) {
                throw new IllegalArgumentException("Invalid type for parameter: silhouette. Expected a boolean.");
            }
            if (!(params.getOrDefault("duplicate_node_set", false) instanceof Boolean)) {
                throw new IllegalArgumentException("Invalid type for parameter: duplicate_node_set. Expected a boolean.");
            }
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("Parameter casting error: " + e.getMessage());
        }

        predictedNodeLabels.clear();

        long totalStartTime = System.nanoTime();
        long affinityTime = 0, laplacianTime = 0, clusteringTime = 0, adjustedRandIndexTime = 0;

// StringBuilder output_string = new StringBuilder("Debug Data: ");

        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (label == null) {
                throw new Exception("Missing node label");
            }

            String originalLabel = label; // Keep track of the original label

            // Optional: Duplicate node set for the experiment
            if (duplicate_node_set) {
                String newLabel = originalLabel + "_" + graph_type + "_" + parameter.replace(".", "_")
                                + "_" + laplacian_type + "_" + Math.round(number_of_eigenvectors);
                Neo4jGraphHandler.duplicateNodeSet(originalLabel, newLabel, connector.getDriver());
                label = newLabel; // Override the label to use the new duplicated set
            }

// String graph_name = "affinity_" + graph_type + "_" + parameter.replace(".", "_") + "_" + label;
            ArrayList<NodeList2> node_properties_list;
            String property_names = "";
            String identifier = "";
            Map<String, Object> results;
            ArrayList<EdgeList2> edge_list_2;
            try (Session sessions = connector.getDriver().session()) {
// Step 1: Create affinity graph
                updateProgress(sessions, "🔄 Step 1: Creating Affinity Graph...");
                long startAffinityTime = System.nanoTime();

                if (is_feature_based) {
                    updateProgress(sessions, "🔄 Creating Affinity Graph - Feature Based...");

// Retrieve node data
                    updateProgress(sessions, "📥 Retrieving node list...");
                    org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                    node_properties_list = node_data.getLeft();
                    identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);
                    property_names = node_data.getRight();
                    updateProgress(sessions, "✅ Node list retrieved.");

// Process remove columns
                    updateProgress(sessions, "🗑️ Processing remove columns...");
                    String[] remove_list = remove_columns.split(",");
                    List<String> remove_list_new = Arrays.stream(remove_list).collect(Collectors.toList());
                    updateProgress(sessions, "✅ Remove columns processed.");

// Compute distance matrix
                    updateProgress(sessions, "📏 Calculating distance matrix...");
                    Double[][] distance_matrix = getDistanceMatrixFromNodes(distance_measure, node_properties_list, remove_list_new);
                    updateProgress(sessions, "✅ Distance matrix computed.");

// Compute adjacency matrix
                    updateProgress(sessions, "🔗 Calculating adjacency matrix...");
                    Double[][] adj_mat;
                    if (graph_type.equals("full")) {
                        updateProgress(sessions, "📊 Using full graph type...");
                        Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(distance_matrix, parameter);
                        adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(distance_matrix, sigmas);
                    } else if (graph_type.equals("eps")) {
                        updateProgress(sessions, "📊 Using eps graph type...");
                        Double epsilon_value = Double.parseDouble(parameter);
                        adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(distance_matrix, epsilon_value);
                    } else if (graph_type.equals("knn")) {
                        updateProgress(sessions, "📊 Using knn graph type...");
                        int[][] knn = ReadCsvTestData.calculateKNNIndices(distance_matrix, parameter);
                        adj_mat = ReadCsvTestData.calculateKNNGraphWithIndices(distance_matrix, knn);
                    } else if (graph_type.equals("mknn")) {
                        updateProgress(sessions, "📊 Using mknn graph type...");
                        Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix, parameter);
                        adj_mat = ReadCsvTestData.calculateMutualKNNGraph(distance_matrix, knn);
                    } else {
                        throw new Exception("❌ Invalid graph_type specified.");
                    }
                    updateProgress(sessions, "✅ Adjacency matrix computed.");

// Compute edge list
                    updateProgress(sessions, "🔗 Creating edge list...");
                    ArrayList<EdgeList2> edge_list = GraphTransform.calculateEdgeList(node_properties_list, adj_mat);
                    edge_list_2 = edge_list;
                    updateProgress(sessions, "✅ Edge list created.");

// // Delete existing graph nodes
// updateProgress(sessions, "🗑️ Deleting existing graph nodes...");
// Neo4jGraphHandler.bulkDeleteNodesWithBatching(graph_name, connector.getDriver(), identifier);
// updateProgress(sessions, "✅ Existing nodes deleted.");

// // Create new graph nodes
// updateProgress(sessions, "🆕 Creating new graph nodes...");
// Neo4jGraphHandler.bulkCreateNodes(graph_name, node_properties_list, connector.getDriver(), identifier);
// updateProgress(sessions, "✅ New graph nodes created.");

// // Create new relationships
// updateProgress(sessions, "🔗 Creating new relationships...");
// Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(graph_name, edge_list, connector.getDriver(), identifier);
// updateProgress(sessions, "✅ New relationships created.");

                } else {
                    updateProgress(sessions, "🔄 Creating Affinity Graph - Non-Feature Based...");

// Retrieve existing node list
                    updateProgress(sessions, "📥 Retrieving node list for non-feature based graph...");
                    org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                    node_properties_list = node_data.getLeft();
                    if (node_properties_list.isEmpty()) {
                        throw new Exception("❌ Affinity graph exists but no nodes retrieved.");
                    }
                    property_names = node_data.getRight();
                    identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);
                    updateProgress(sessions, "✅ Node list retrieved for non-feature based graph.");


// Retrieve Edge List
                    updateProgress(sessions, "📥 Retrieving edge list...");
                    edge_list_2 = Neo4jGraphHandler.retrieveEdgeList(label, connector.getDriver());
                    updateProgress(sessions, "✅ Edge list retrieved. Total edges: " + edge_list_2.size());
                }

                affinityTime = System.nanoTime() - startAffinityTime;
                updateProgress(sessions, "✅ Step 1 Completed: Affinity Graph Created. Took " + (affinityTime / 1e6) + " ms.");

// Step 2: Compute Laplacian matrix
                updateProgress(sessions, "🔄 Step 2: Computing Laplacian Matrix...");
                long startLaplacianTime = System.nanoTime();

// Convert to Adjacency Matrix
                updateProgress(sessions, "🔗 Converting edge list to adjacency matrix...");
                RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list_2, node_properties_list);
                updateProgress(sessions, "✅ Adjacency matrix computed.");

// Compute Degree Matrix
                updateProgress(sessions, "📊 Computing degree matrix...");
                RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
                updateProgress(sessions, "✅ Degree matrix computed.");

// Compute Laplacian Matrix
                updateProgress(sessions, "📉 Computing Laplacian matrix (Type: " + laplacian_type + ")...");
                RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
                updateProgress(sessions, "✅ Laplacian matrix computed.");

// Compute Eigenvalues & Eigenvectors
                updateProgress(sessions, "🧠 Computing eigenvalues and eigenvectors...");
                Object eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);

                if (eigen_result_obj instanceof String) {
                    throw new Exception((String) eigen_result_obj);
                }
                EigenCalculation.EigenResult eigen_result = (EigenCalculation.EigenResult) eigen_result_obj;
                updateProgress(sessions, "✅ Eigenvalues and eigenvectors computed.");

// // Create Edge List for Eigenvectors
// updateProgress(sessions, "🔗 Creating edge list from eigenvectors...");
// ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_properties_list, eigen_result.X, edge_list_2);
// updateProgress(sessions, "✅ Edge list from eigenvectors created.");

// // Define Eigen Graph Name
// String graph_name_eigen = "eigen_" + laplacian_type + "_" + Math.round(number_of_eigenvectors) + "_" + graph_name;
// updateProgress(sessions, "📌 Created new graph name: " + graph_name_eigen);

// // Delete Existing Eigen Nodes
// updateProgress(sessions, "🗑️ Deleting existing nodes for eigen graph...");
// Neo4jGraphHandler.bulkDeleteNodesWithBatching(graph_name_eigen, connector.getDriver(), identifier);
// updateProgress(sessions, "✅ Existing eigen graph nodes deleted.");

// // Create New Eigen Nodes
// updateProgress(sessions, "🆕 Creating new eigen nodes...");
// Neo4jGraphHandler.bulkCreateNodesWithEigen(graph_name_eigen, node_properties_list, eigen_result.X, connector.getDriver(), identifier);
// updateProgress(sessions, "✅ New eigen nodes created.");

// // Create New Relationships for Eigen Graph
// updateProgress(sessions, "🔗 Creating relationships for eigen graph...");
// Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(graph_name_eigen, edge_list_eigen, connector.getDriver(), identifier);
// updateProgress(sessions, "✅ Relationships for eigen graph created.");

// --- Only update existing nodes with eigenvector properties and optionally regenerate edges ---
                updateProgress(sessions, "🔗 Updating node properties with calculated eigenvectors ...");
                List<Map<String, Object>> eigenvectorProperties = new ArrayList<>();
                for (int i = 0; i < node_properties_list.size(); i++) {
                    Map<String, Object> props = new HashMap<>();
                    for (int j = 0; j < eigen_result.X.numCols(); j++) {
                        props.put("eigenvector_" + j, eigen_result.X.get(i, j));
                    }
                    eigenvectorProperties.add(props);
                }
                Neo4jGraphHandler.updateNodeProperties(
                        label,
                        node_properties_list,
                        eigenvectorProperties,
                        connector.getDriver(),
                        identifier
                );
                updateProgress(sessions, "✅ Node properties with eigenvectors updated.");

// updateProgress(sessions, "🔗 Regenerating new edges based on eigen calculations ...");
// --- BEGIN: Optionally regenerate edges based on eigen decomposition ---
// in case the edges structure should differ from the original.
/*
ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_properties_list, eigen_result.X, edge_list);
Neo4jGraphHandler.bulkCreateRelationshipsWithBatching(label, edge_list_eigen, connector.getDriver(), identifier);
// return "Regenerated edges based on eigenvectors!";
*/
// --- END: Optionally regenerate edges based on eigen decomposition ---
// updateProgress(sessions, "✅ Regenerated edges based on eigenvectors.");

// Record total Laplacian computation time
                laplacianTime = System.nanoTime() - startLaplacianTime;
                updateProgress(sessions, "✅ Step 2 Completed: Laplacian Matrix and Eigendecomposition Computed. Took " + (laplacianTime / 1e6) + " ms.");

// Step 3: Perform Clustering
                updateProgress(sessions, "🔄 Step 3: Performing Clustering...");
                long startClusteringTime = System.nanoTime();

// Determine Number of Clusters
                updateProgress(sessions, "📊 Determining number of clusters...");
                String number_of_clusters = Integer.toString(number_of_eigenvectors.intValue());
                updateProgress(sessions, "✅ Number of clusters: " + number_of_clusters);

// Filter Properties
                updateProgress(sessions, "🔍 Filtering properties for clustering...");
                String[] properties_array = property_names.split(",");
                StringBuilder filtered_properties = new StringBuilder();

                for (String property : properties_array) {
                    if (!property.equalsIgnoreCase(identifier) && !property.equalsIgnoreCase("target")) {
                        if (filtered_properties.length() > 0) {
                            filtered_properties.append(",");
                        }
                        filtered_properties.append(property);
                    }
                }
                updateProgress(sessions, "✅ Filtered properties: " + filtered_properties);

// Print filtered properties for debugging
                System.out.println("filtered_properties: " + filtered_properties);

// Execute k-Means Clustering
                updateProgress(sessions, "⚡ Running k-Means clustering...");
                double kmean_result = kMeans(Map.of(
                        "label", label,
                        "number_of_centroids", number_of_clusters,
                        "number_of_iterations", number_of_iterations,
                        "distance_measure", distance_measure_kmean,
                        "original_set", originalLabel, // Use original label for fetching true labels
                        "overlook", target_column + "," + filtered_properties.toString(),
                        "overlook_original", target_column,
                        "silhouette", silhouette,
                        "seed", seed,
                        "create_new_graph", create_new_graph
                ));
                updateProgress(sessions, "✅ k-Means clustering completed. Silhouette Score: " + kmean_result);


// Record total clustering computation time
                clusteringTime = System.nanoTime() - startClusteringTime;
                updateProgress(sessions, "✅ Step 3 Completed: Clustering Done. Took " + (clusteringTime / 1e6) + " ms.");

// Step 4: Compute Adjusted Rand Index
                updateProgress(sessions, "🔄 Step 4: Computing Adjusted Rand Index...");
                long startAdjustedRandIndexTime = System.nanoTime();

// Start computation
                updateProgress(sessions, "📊 Running adjusted Rand Index calculation...");
                Map<String, Object> adjustedRandIndexResult = adjustedRandIndex(Map.of(
                        "label", originalLabel,
                        "true_labels", target_column
                ));

// Extract Results
                updateProgress(sessions, "📥 Extracting adjusted Rand Index results...");
                double adjustedRandIndexValue = (double) adjustedRandIndexResult.get("adjustedRandIndex");
                List<Integer> true_labels = (List<Integer>) adjustedRandIndexResult.get("true_labels");
                List<Integer> predictedLabels = (List<Integer>) adjustedRandIndexResult.get("predictedLabels");

                updateProgress(sessions, "✅ Adjusted Rand Index Computed: " + adjustedRandIndexValue);
                updateProgress(sessions, "📋 True Labels Count: " + true_labels.size());
                updateProgress(sessions, "📋 Predicted Labels Count: " + predictedLabels.size());


// Record computation time
                adjustedRandIndexTime = System.nanoTime() - startAdjustedRandIndexTime;
                updateProgress(sessions, "✅ Step 4 Completed: Adjusted Rand Index Computation Done. Took " + (adjustedRandIndexTime / 1e6) + " ms.");

// Compute Total Execution Time
                long totalTime = System.nanoTime() - totalStartTime;
                updateProgress(sessions, "🕒 Total Execution Time: " + (totalTime / 1e6) + " ms.");

// Return metrics
                results = new HashMap<>();
                results.put("silhouette_score", kmean_result);
                results.put("rand_index", adjustedRandIndexValue);
                results.put("total_time", totalTime / 1e6);
                results.put("affinity_time", affinityTime / 1e6);
                results.put("laplacian_time", laplacianTime / 1e6);
                results.put("clustering_time", clusteringTime / 1e6);
                results.put("adjusted_rand_index_time", adjustedRandIndexTime / 1e6);
            }
            return results;
        }
    }

    // Helper method to update progress in Neo4j
    // private void updateProgress(Session session, String step) {
    //     String progressQuery = "MERGE (p:Progress {id: 'current'}) SET p.step = $step, p.timestamp = datetime()";
    //     session.run(progressQuery, Values.parameters("step", step));
//
    //     try {
    //         Thread.sleep(2000); // Delay of 2 seconds
    //     } catch (InterruptedException e) {
    //         Thread.currentThread().interrupt(); // Restore interrupt status
    //         System.err.println("Thread was interrupted during sleep.");
    //     }
    // }

    // Helper method to update progress in Neo4j
    private void updateProgress(Session session, String step) {
        String progressQuery = "MERGE (p:Progress {id: 'current'}) SET p.step = $step, p.timestamp = datetime()";
        session.run(progressQuery, Values.parameters("step", step));
    }

    // Helper method to filter properties
    private String filterProperties(String property_names, String identifier) {
        return Arrays.stream(property_names.split(","))
                .filter(prop -> !prop.equalsIgnoreCase(identifier) && !prop.equalsIgnoreCase("target"))
                .collect(Collectors.joining(","));
    }

    @UserFunction
    public String displayEdgeList(@Name("label") String label, @Name("data_path") String data_path, @Name("distance_measure") String distance_measure, @Name("graph_type") String graph_type, @Name("method") String method, @Name("parameter") String parameter, @Name("remove_column") String remove_columns) throws Exception {

        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {

            if (data_path == null && distance_measure == null) {
                return "Missing data_path or distance measure type";
            } else {

// Display edge list


                String graph_name = "affinity_";
                Double[][] adj_mat = null;
                ArrayList<NodeList2> node_properties_list = null;
                String[] remove_list = remove_columns.split(",");
                List<String> remove_list_new = Arrays.stream(remove_list).collect(Collectors.toList());
                if (label.isEmpty()) {
                    ReadCsvTestData readCsvTestData = new ReadCsvTestData(data_path);
// //ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNew(data_path,IndexColumn);
                    node_properties_list = readCsvTestData.readCsvFileToMap(data_path);
                }
                org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodeData = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                node_properties_list = nodeData.getLeft();
// node_properties_list = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(label, connector.getDriver());

// ArrayList<NodeList2> nodePropertiesList_copy = readCsvTestData.readCsvFileToMap(data_path);
                Double[][] distance_matrix = getDistanceMatrixFromNodes(distance_measure, node_properties_list, remove_list_new);

                StringBuilder output_string = new StringBuilder("Graph Data: ");
                output_string.append("\n\nDistance Matrix:\n").append(doubleToString(distance_matrix));


                if (graph_type.equals("full")) {
                    Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(distance_matrix, sigmas);
                    graph_name = graph_type.concat("_" + parameter);
                }
                if (graph_type.equals("eps")) {
                    Double epsilon = Double.parseDouble(parameter);
                    adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(distance_matrix, epsilon);
                    graph_name = graph_type.concat("_" + parameter);
                }
                if (graph_type.equals("knn")) {
                    Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateKNNGraph(distance_matrix, knn);
                    graph_name = graph_type.concat("_" + parameter);
                }
                if (graph_type.equals("mknn")) {
                    Double[][] knn = ReadCsvTestData.calculateKNN(distance_matrix, parameter);
                    adj_mat = ReadCsvTestData.calculateMutualKNNGraph(distance_matrix, knn);
                    graph_name = graph_type.concat("_" + parameter);
                }

                output_string.append("\n\nAdjacency Matrix:\n").append(doubleToString(adj_mat));

                ArrayList<EdgeList2> edge_list = GraphTransform.calculateEdgeList(node_properties_list, adj_mat);

                output_string.append("\n\nEdge List:\n");
                for (EdgeList2 edge : edge_list) {
                    output_string.append(" | ").append(edge.toString());
                }


                return output_string.toString() + '\n' + "nodes " + node_properties_list;
            }
        }
    }

    /**
     * Displays information about a graph based on the specified node label and desired number of eigenvectors.
     * The Laplacian matrix is set based on the specified laplacian_type.
     *
     * @param label                  The label of the nodes in the graph.
     * @param number_of_eigenvectors The number of desired eigenvectors to compute during eigen decomposition.
     * @param laplacian_type         The type of Laplacian matrix to be used.
     * @return String containing information about the graph, including edge list, adjacency matrix, degree matrix,
     * Laplacian matrix, and eigen decomposition results.
     * @throws Exception If an error occurs during the process.
     */
    @UserFunction
    public String displayGraphList(@Name("label") String label,
                                   @Name("number_of_eigenvectors") Double number_of_eigenvectors,
                                   @Name("laplacian_type") String laplacian_type) throws Exception {
        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (label == null) {
                return "Error: Missing node label.";
            }

            StringBuilder output_string = new StringBuilder("Edge List Data: ");
            ArrayList<EdgeList2> edge_list = Neo4jGraphHandler.retrieveEdgeListFromNeo4j(label, connector.getDriver());
            for (EdgeList2 edge : edge_list) {
                output_string.append(" | ").append(edge.toString());
            }
            org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
            ArrayList<NodeList2> node_list = node_data.getLeft();
// Calculate matrices
            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list, node_list);
            output_string.append("\n\nAdjacency Matrix:\n").append(matrixToString(adjacency_matrix));

            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
            output_string.append("\n\nDegree Matrix:\n").append(matrixToString(degree_matrix));

            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
            output_string.append("\n\nLaplacian Matrix:\n").append(matrixToString(laplacian_matrix));

// Attempt to calculate eigen decomposition
            Object eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);

// Check if the result is an error message
            if (eigen_result_obj instanceof String) {
                return (String) eigen_result_obj; // Return the error message directly
            }

// Cast the result to EigenResult and output eigenvalues, eigenvectors, and X matrix
            EigenCalculation.EigenResult eigen_result = (EigenCalculation.EigenResult) eigen_result_obj;

            output_string.append("\n\nEigenvalues:\n").append(Arrays.toString(eigen_result.eigenvalues));
            output_string.append("\n\nEigenvectors:\n").append(matrixToString2(eigen_result.eigenvectors));
            output_string.append("\n\nX Matrix (Top eigenvectors):\n").append(matrixToString2(eigen_result.X));

            return output_string.toString();
        } catch (Neo4jException e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    @UserFunction
    public String displayFinalResults(@Name("label") String label, @Name("number_of_centroids") String number_of_centroids, @Name("number_of_iterations") String number_of_iterations, @Name("distance_measure") String distance_measure, @Name("seed") Number seed) throws Exception {

        predictedNodeLabels.clear();
        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {

            if (label == null && distance_measure == null) {
                return "Missing data_path or distance measure type";
            } else {

                StringBuilder output_string = new StringBuilder("The average Silhouette Coefficient value is: ");
                HashMap<String, ArrayList<String>> kmeanAssign = new HashMap<String, ArrayList<String>>();
                int numberOfCentroidInt = Integer.parseInt(number_of_centroids);
                int numberOfInterationInt = Integer.parseInt(number_of_iterations);
                double centroidNumber = 1.0;

                ArrayList<String> debug = new ArrayList<>();
                kmeanAssign = Unsupervised.KmeanClust(mapNodeList, numberOfCentroidInt, numberOfInterationInt, distance_measure, false, debug, (int) seed);
                double averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(kmeanAssign, distance_measure);
                output_string.append(averageSilhouetteCoefficientValue);

                output_string.append("\n\nKmean assign:\n").append(hashMapToString(kmeanAssign));

                for (String centroid : kmeanAssign.keySet()) {
                    ArrayList<String> clusterNode = kmeanAssign.get(centroid);
                    for (int i = 0; i < clusterNode.size(); i++) {
//Add predict labels
                        predictedNodeLabels.add(centroidNumber);

                        DecimalFormat decimalFormat = new DecimalFormat("#.###");
                        double distance = Unsupervised.calculateDistance(clusterNode.get(i), centroid, distance_measure);
                        connectNodes(label, "create relationship in kmean node", centroid, clusterNode.get(i), distance, connector.getDriver());
// connector.connectNodes(label, "create relationship in kmean node", centroid, clusterNode.get(i), roundedDistance);
                    }
                    centroidNumber = centroidNumber + 1;
                }

                return output_string + " predicted labels: " + predictedNodeLabels;
            }
        }
    }

    /**
     * Retrieves and saves a specified matrix (adjacency or X matrix) from the graph in Neo4j.
     *
     * @param label       The label of the nodes in the graph.
     * @param matrix_type The type of matrix to retrieve ("adjacency" or "x_matrix").
     * @param save_path   The path where the matrix CSV file should be saved.
     * @return String indicating the success of the operation.
     * @throws Exception If an error occurs during the process.
     */
    @UserFunction
    public String getMatrix(@Name("label") String label,
                            @Name("matrix_type") String matrix_type,
                            @Name("save_path") String save_path) throws Exception {
        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (label == null) {
                return "Missing node label";
            } else {
                StringBuilder output_string = new StringBuilder();
                if ("adjacency".equals(matrix_type)) {
                    ArrayList<EdgeList2> edge_list = Neo4jGraphHandler.retrieveEdgeListFromNeo4j(label, connector.getDriver());
                    org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                    ArrayList<NodeList2> node_list = node_data.getLeft();
                    RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list, node_list);
                    output_string.append("Adjacency matrix:\n").append(matrixToString(adjacency_matrix));

// Save adjacency matrix to CSV at user-defined or default path
                    saveMatrixToCSV(adjacency_matrix, label + "_adjacency_matrix.csv", save_path);
                } else if ("x_matrix".equals(matrix_type)) {
                    org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodes = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                    RealMatrix x_matrix = MatrixCalculation.convertToXMatrix(nodes.getLeft());
                    output_string.append("X matrix with eigenvectors:\n").append(matrixToString(x_matrix));

// Save X matrix to CSV at user-defined or default path
                    saveMatrixToCSV(x_matrix, label + "_x_matrix.csv", save_path);
                } else {
                    return "Invalid matrix type specified";
                }
                return output_string.toString();
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error in Neo4j operation: " + e.getMessage());
        }
    }

    /**
     * Saves a given matrix to a CSV file at the specified path.
     *
     * @param matrix    The matrix to save.
     * @param filename  The name of the CSV file.
     * @param save_path The path where the CSV file should be saved.
     * @throws IOException If an error occurs during file writing.
     */
    public void saveMatrixToCSV(RealMatrix matrix, String filename, String save_path) throws IOException {
// Determine the file path based on user input or default to the current directory
        String file_path = (save_path == null || save_path.isEmpty()) ? "./" + filename : save_path + "/" + filename;

// Write the matrix data to the CSV file
        try (FileWriter csv_writer = new FileWriter(file_path)) {
            for (int i = 0; i < matrix.getRowDimension(); i++) {
                for (int j = 0; j < matrix.getColumnDimension(); j++) {
                    csv_writer.append(Double.toString(matrix.getEntry(i, j)));
                    if (j < matrix.getColumnDimension() - 1) {
                        csv_writer.append(",");
                    }
                }
                csv_writer.append("\n");
            }
        }
    }

    public static String doubleToString(Double[][] matrix) {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sb.append(matrix[i][j]);
                if (j < matrix[i].length - 1) {
                    sb.append(" "); // Separate columns with a space
                }
            }
            sb.append("\n"); // Newline after each row
        }

        return sb.toString();
    }

    private String matrixToString(RealMatrix matrix) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                result.append(matrix.getEntry(i, j)).append("\t");
            }
            result.append("\n");
        }
        return result.toString();
    }

    private String matrixToString2(SimpleMatrix matrix) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < matrix.getNumRows(); i++) {
            for (int j = 0; j < matrix.getNumCols(); j++) {
                result.append(matrix.get(i, j)).append("\t");
            }
            result.append("\n");
        }
        return result.toString();
    }


    public String hashMapToString(HashMap<String, ArrayList<String>> kmeanAssign) {
        StringBuilder sb = new StringBuilder();

// Iterate through the entries of the HashMap
        for (HashMap.Entry<String, ArrayList<String>> entry : kmeanAssign.entrySet()) {
            String key = entry.getKey();
            ArrayList<String> values = entry.getValue();

            sb.append(key).append(": ");

// Append the values in the ArrayList to the StringBuilder
            sb.append(values.toString());

// Add a newline after each key-value pair
            sb.append("\n");
        }

        return sb.toString();
    }
}