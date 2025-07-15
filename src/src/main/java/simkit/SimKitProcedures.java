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
import org.neo4j.driver.summary.ResultSummary;
import org.neo4j.driver.types.Node;
import org.neo4j.driver.util.Pair;
import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
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

    @Context
    public Log log;

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
                ArrayList<NodeList2> node_properties_list = readCsvTestData.readCsvFileToMap(data_path);
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

            org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
            ArrayList<NodeList2> node_properties_list = node_data.getLeft();
            String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);

            ArrayList<EdgeList2> edge_list = Neo4jGraphHandler.retrieveEdgeList(label, connector.getDriver());

            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list, node_properties_list);
            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);

            Object eigen_result_obj;
            try {
                eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);
            } catch (IllegalArgumentException e) {
                throw new Exception("Invalid input: " + e.getMessage());
            } catch (RuntimeException e) {
                throw new Exception("Error in eigen decomposition: " + e.getMessage());
            }

            EigenCalculation.EigenResult eigen_result = (EigenCalculation.EigenResult) eigen_result_obj;

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

            return "Updated existing nodes with eigenvector properties!";

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
                        list_of_data = mapNodeList.toString();
                    }
                }
            }
        }
        return "Map all node data: " + list_of_data;
    }

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

            log.info("kMeans - Parsing node values for label: %s", label);
            ArrayList<String> mapNodeList = parseNodeValues(label, overlook.split(","));
            if (mapNodeList == null || mapNodeList.isEmpty()) {
                throw new Exception("No nodes found with the label: " + label);
            }
            log.info("kMeans - Found %d nodes. First 5 vectors: %s", mapNodeList.size(), mapNodeList.stream().limit(5).collect(Collectors.toList()));


            ArrayList<String> mapNodeOriginalList = parseNodeValues(original_set, overlook_original.split(","));

            HashMap<String, ArrayList<String>> kmeanAssign = Unsupervised.KmeanClust(
                    mapNodeList, numCentroids, numIterations, distance_measure, false, new ArrayList<>(), seed
            );
            log.info("kMeans - Clustering complete. Number of resulting clusters: %d", kmeanAssign.keySet().size());

            Map<String, String> nodeToCentroidMap = new HashMap<>();
            for (Map.Entry<String, ArrayList<String>> entry : kmeanAssign.entrySet()) {
                String centroid = entry.getKey();
                ArrayList<String> nodesInCluster = entry.getValue();
                for (String node : nodesInCluster) {
                    nodeToCentroidMap.put(node, centroid);
                }
            }

            Map<String, Double> centroidToClusterIdMap = new HashMap<>();
            double clusterIdCounter = 1.0;
            for (String centroid : kmeanAssign.keySet()) {
                centroidToClusterIdMap.put(centroid, clusterIdCounter++);
            }

            predictedNodeLabels.clear();
            for (String originalNodeStr : mapNodeList) {
                String centroidStr = nodeToCentroidMap.get(originalNodeStr);
                if (centroidStr != null) {
                    predictedNodeLabels.add(centroidToClusterIdMap.get(centroidStr));
                } else {
                    log.warn("kMeans - Node not assigned to any cluster: %s", originalNodeStr);
                    predictedNodeLabels.add(-1.0);
                }
            }


            HashMap<String, ArrayList<String>> cleanedKmeanAssign = Unsupervised.removeIndexAndId(kmeanAssign);
            double averageSilhouetteCoefficientValue;
            if (silhouette) {
                averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(cleanedKmeanAssign, distance_measure);
            } else {
                HashMap<String, ArrayList<String>> mappedNodes = Unsupervised.replaceValuesWithOriginalSet(kmeanAssign, mapNodeOriginalList);
                averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(mappedNodes, distance_measure);
            }
            log.info("kMeans - Calculated silhouette score: %f", averageSilhouetteCoefficientValue);


            if (create_new_graph) {
                processClusters(connector, label, cleanedKmeanAssign, distance_measure);
            } else {
                org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                ArrayList<NodeList2> node_properties_list = node_data.getLeft();
                String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), label);

                List<Map<String, Object>> clusterProperties = new ArrayList<>();
                for (int i = 0; i < node_properties_list.size(); i++) {
                    Map<String, Object> props = new HashMap<>();
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

    private void processClusters(SimKitProcedures connector, String label,
                                 HashMap<String, ArrayList<String>> kmeanAssign,
                                 String distance_measure) throws Exception {
        DecimalFormat decimalFormat = new DecimalFormat("#.###");
        for (String centroid : kmeanAssign.keySet()) {
            ArrayList<String> clusterNodes = kmeanAssign.get(centroid);

            for (String clusterNode : clusterNodes) {
                double distance = Unsupervised.calculateDistance(clusterNode, centroid, distance_measure);
                String formattedDistance = decimalFormat.format(distance);

                connectNodes(label, "create relationship in kmean node", centroid, clusterNode, Double.parseDouble(formattedDistance), getDriver());
            }
        }
    }

    @UserFunction
    public Map<String, Object> adjustedRandIndex(@Name("params") Map<String, Object> params) throws Exception {

        String label = (String) params.getOrDefault("label", "Iris");
        String true_labels_col = (String) params.getOrDefault("true_labels", "target");

        if (predictedNodeLabels.isEmpty()) {
            throw new Exception("Predicted Labels list is empty. Cannot calculate Adjusted Rand Index.");
        }

        trueNodeLabels.clear();
        List<String> stringTrueNodeLabelsList = new ArrayList<>();

        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            queryData(label);

            for (Record record : dataKey) {
                Node nodeValue = record.get("n").asNode();
                if (nodeValue.containsKey(true_labels_col)) {
                    Value labelValue = nodeValue.get(true_labels_col);
                    try {
                        trueNodeLabels.add(labelValue.asDouble());
                    } catch (Exception e) {
                        stringTrueNodeLabelsList.add(labelValue.asString());
                    }
                } else {
                    stringTrueNodeLabelsList.add("MISSING_LABEL");
                }
            }

            if (!stringTrueNodeLabelsList.isEmpty()) {
                if (trueNodeLabels.isEmpty()) {
                    trueNodeLabels = convertStringLabels(stringTrueNodeLabelsList);
                } else {
                    List<String> fullStringList = new ArrayList<>();
                    queryData(label);
                    for (Record record : dataKey) {
                        Node nodeValue = record.get("n").asNode();
                        if (nodeValue.containsKey(true_labels_col)) {
                            fullStringList.add(nodeValue.get(true_labels_col).toString());
                        } else {
                            fullStringList.add("MISSING_LABEL");
                        }
                    }
                    trueNodeLabels = convertStringLabels(fullStringList);
                }
            }

            log.info("AdjustedRandIndex - True labels size: %d, Predicted labels size: %d", trueNodeLabels.size(), predictedNodeLabels.size());
            log.debug("AdjustedRandIndex - First 10 True Labels: %s", trueNodeLabels.stream().limit(10).collect(Collectors.toList()));
            log.debug("AdjustedRandIndex - First 10 Predicted Labels: %s", predictedNodeLabels.stream().limit(10).collect(Collectors.toList()));


            if (trueNodeLabels.size() != predictedNodeLabels.size()) {
                throw new Exception("Mismatch in label sizes. True labels: " + trueNodeLabels.size() + ", Predicted labels: " + predictedNodeLabels.size());
            }

            double adjustedRandIndexValue = calculateAdjustedRandIndex(trueNodeLabels, predictedNodeLabels);
            log.info("AdjustedRandIndex - Calculated ARI: %f", adjustedRandIndexValue);


            Map<String, Object> result = new HashMap<>();
            result.put("adjustedRandIndex", adjustedRandIndexValue);
            result.put("true_labels", trueNodeLabels);
            result.put("predictedLabels", predictedNodeLabels);

            return result;
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


    public void queryData(final String label) {
        try (Session session = getDriver().session()) {
            session.readTransaction(tx -> {
                Result result = tx.run("MATCH (n:" + label + ") RETURN n ORDER BY id(n)");
                dataKey = result.list();
                return "Query Successful";
            });
        }
    }

    public void connectNodes(final String label, final String message, final String nodeCentroid, final String nodeCluster, final double distance, Driver driver) throws Exception {
        final String name = "kmean";
        try (Session session = driver.session()) {
            String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(getDriver(), label);

            String greeting = session.writeTransaction(new TransactionWork<String>() {
                @Override
                public String execute(Transaction tx) {

                    String identifier = Neo4jGraphHandler.resolveDynamicIdentifier(getDriver(), label);

                    String deleteQuery = "MATCH (a:Clustering_" + label + " {" + nodeCentroid + "}) " +
                            "DETACH DELETE a";
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
        List<String> keys = new ArrayList<>();
        value.keys().forEach(keys::add);
        Collections.sort(keys);

        for (String nodeKey : keys) {
            boolean shouldOverlook = false;
            for (String overlook : overlook_array) {
                if (overlook.trim().equalsIgnoreCase(nodeKey.trim())) {
                    shouldOverlook = true;
                    break;
                }
            }
            if (shouldOverlook) {
                continue;
            }

            Value propertyValue = value.get(nodeKey);
            String typeName = propertyValue.type().name();
            if (typeName.equalsIgnoreCase("INTEGER")
                    || typeName.equalsIgnoreCase("FLOAT")
                    || typeName.equalsIgnoreCase("DOUBLE")) {
                valueOfNode.append(getStringValue(valueOfNode))
                        .append(nodeKey)
                        .append(":")
                        .append(propertyValue.asDouble());
            }
        }
        if (valueOfNode.length() == 0) {
            log.warn("Generated empty vector for node with properties: %s", value.toString());
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

        for (int i = 0; i < n; i++) {
            double trueLabel = trueLabels.get(i);
            double predictedLabel = predictedLabels.get(i);

            contingencyTable.computeIfAbsent(trueLabel, k -> new HashMap<>());
            contingencyTable.get(trueLabel).merge(predictedLabel, 1.0, Double::sum);

            trueLabelCounts.merge(trueLabel, 1.0, Double::sum);
            predictedLabelCounts.merge(predictedLabel, 1.0, Double::sum);
        }

        double a = 0.0;
        for (Map<Double, Double> row : contingencyTable.values()) {
            for (double count : row.values()) {
                a += count * (count - 1) / 2.0;
            }
        }

        double b = 0.0;
        for (double count : trueLabelCounts.values()) {
            b += count * (count - 1) / 2.0;
        }

        double c = 0.0;
        for (double count : predictedLabelCounts.values()) {
            c += count * (count - 1) / 2.0;
        }

        double totalPairs = n * (n - 1) / 2.0;
        double expectedIndex = (b * c) / totalPairs;
        double maxIndex = 0.5 * (b + c);
        double adjustedRandIndex = (a - expectedIndex) / (maxIndex - expectedIndex);

        return adjustedRandIndex;
    }

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

            ArrayList<NodeList2> node_properties_list;
            String property_names = "";
            String identifier = "";
            ArrayList<EdgeList2> edge_list_2;

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

            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list_2, node_properties_list);
            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
            Object eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);

            if (eigen_result_obj instanceof String) {
                throw new Exception((String) eigen_result_obj);
            }
            EigenCalculation.EigenResult eigen_result = (EigenCalculation.EigenResult) eigen_result_obj;

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

            return kmean_result;
        }
    }

    @UserFunction
    public Map<String, Object> experimentalSpectralClustering(@Name("params") Map<String, Object> params) throws Exception {
        log.info("Starting experimentalSpectralClustering with params: %s", params.toString());

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


        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (label == null) {
                throw new Exception("Missing node label");
            }

            String originalLabel = label;
            String experimentLabel = label;

            if (duplicate_node_set) {
                String newLabel = originalLabel + "_exp";
                log.info("Duplicating node set from '%s' to '%s' for experiment.", originalLabel, newLabel);

                try (Session session = connector.getDriver().session()) {
                    // 1. Delete old experimental nodes
                    session.run(String.format("MATCH (n:`%s`) DETACH DELETE n", newLabel)).consume();
                    log.info("Cleaned up old experimental nodes with label '%s'.", newLabel);

                    // 2. Copy nodes
                    String copyNodesQuery = String.format("MATCH (n:`%s`) CREATE (m:`%s`) SET m = properties(n)", originalLabel, newLabel);
                    ResultSummary copyNodesSummary = session.run(copyNodesQuery).consume();
                    log.info("Duplicated %d nodes to new label '%s'.", copyNodesSummary.counters().nodesCreated(), newLabel);

                    // 3. If graph-based, copy relationships
                    if (!is_feature_based) {
                        String copyRelsQuery = String.format(
                                "MATCH (a:`%s`)-[r]->(b:`%s`) " +
                                        "MATCH (a_new:`%s` {id: a.id}), (b_new:`%s` {id: b.id}) " +
                                        "MERGE (a_new)-[new_r:REL]->(b_new) SET new_r = properties(r)",
                                originalLabel, originalLabel, newLabel, newLabel
                        );
                        ResultSummary copyRelsSummary = session.run(copyRelsQuery).consume();
                        log.info("Duplicated %d relationships for new label '%s'.", copyRelsSummary.counters().relationshipsCreated(), newLabel);
                    }
                }
                experimentLabel = newLabel;
            }


            ArrayList<NodeList2> node_properties_list;
            String property_names = "";
            String identifier = "";
            Map<String, Object> results;
            ArrayList<EdgeList2> edge_list_2;
            try (Session sessions = connector.getDriver().session()) {
                updateProgress(sessions, "🔄 Step 1: Creating Affinity Graph...");
                long startAffinityTime = System.nanoTime();

                log.info("Processing label: '%s'. is_feature_based: %s", experimentLabel, is_feature_based);
                if (is_feature_based) {
                    org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(experimentLabel, connector.getDriver());
                    node_properties_list = node_data.getLeft();
                    identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), experimentLabel);
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
                        throw new Exception("❌ Invalid graph_type specified.");
                    }
                    edge_list_2 = GraphTransform.calculateEdgeList(node_properties_list, adj_mat);

                } else {
                    org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> node_data = Neo4jGraphHandler.retrieveNodeList(experimentLabel, connector.getDriver());
                    node_properties_list = node_data.getLeft();
                    if (node_properties_list.isEmpty()) {
                        throw new Exception("❌ Affinity graph exists but no nodes retrieved for label: " + experimentLabel);
                    }
                    property_names = node_data.getRight();
                    identifier = Neo4jGraphHandler.resolveDynamicIdentifier(connector.getDriver(), experimentLabel);
                    edge_list_2 = Neo4jGraphHandler.retrieveEdgeList(experimentLabel, connector.getDriver());
                    log.info("Graph-based: Retrieved %d nodes and %d edges for label '%s'", node_properties_list.size(), edge_list_2.size(), experimentLabel);
                }

                affinityTime = System.nanoTime() - startAffinityTime;
                updateProgress(sessions, "✅ Step 1 Completed: Affinity Graph Created. Took " + (affinityTime / 1e6) + " ms.");

                updateProgress(sessions, "🔄 Step 2: Computing Laplacian Matrix...");
                long startLaplacianTime = System.nanoTime();

                RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list_2, node_properties_list);
                RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
                RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
                log.info("Computed Laplacian of type '%s'. Matrix dimensions: %d x %d", laplacian_type, laplacian_matrix.getRowDimension(), laplacian_matrix.getColumnDimension());

                Object eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);

                if (eigen_result_obj instanceof String) {
                    throw new Exception((String) eigen_result_obj);
                }
                EigenCalculation.EigenResult eigen_result = (EigenCalculation.EigenResult) eigen_result_obj;
                log.info("Eigen decomposition complete. Eigen-matrix dimensions: %d x %d", eigen_result.X.numRows(), eigen_result.X.numCols());

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
                        experimentLabel,
                        node_properties_list,
                        eigenvectorProperties,
                        connector.getDriver(),
                        identifier
                );
                updateProgress(sessions, "✅ Node properties with eigenvectors updated.");

                laplacianTime = System.nanoTime() - startLaplacianTime;
                updateProgress(sessions, "✅ Step 2 Completed: Laplacian Matrix and Eigendecomposition Computed. Took " + (laplacianTime / 1e6) + " ms.");

                updateProgress(sessions, "🔄 Step 3: Performing Clustering...");
                long startClusteringTime = System.nanoTime();

                String number_of_clusters = Integer.toString(number_of_eigenvectors.intValue());

                String[] properties_array = property_names.split(",");
                StringBuilder filtered_properties = new StringBuilder();

                for (String property : properties_array) {
                    if (!property.equalsIgnoreCase(identifier) && !property.equalsIgnoreCase(target_column)) {
                        if (filtered_properties.length() > 0) {
                            filtered_properties.append(",");
                        }
                        filtered_properties.append(property);
                    }
                }

                String overlook_for_kmeans = target_column + "," + filtered_properties.toString();
                log.info("Calling kMeans on label '%s' with overlook columns: %s", experimentLabel, overlook_for_kmeans);

                double kmean_result = kMeans(Map.of(
                        "label", experimentLabel,
                        "number_of_centroids", number_of_clusters,
                        "number_of_iterations", number_of_iterations,
                        "distance_measure", distance_measure_kmean,
                        "original_set", originalLabel,
                        "overlook", overlook_for_kmeans,
                        "overlook_original", target_column,
                        "silhouette", silhouette,
                        "seed", seed,
                        "create_new_graph", create_new_graph
                ));
                updateProgress(sessions, "✅ k-Means clustering completed. Silhouette Score: " + kmean_result);

                clusteringTime = System.nanoTime() - startClusteringTime;
                updateProgress(sessions, "✅ Step 3 Completed: Clustering Done. Took " + (clusteringTime / 1e6) + " ms.");

                updateProgress(sessions, "🔄 Step 4: Computing Adjusted Rand Index...");
                long startAdjustedRandIndexTime = System.nanoTime();

                Map<String, Object> adjustedRandIndexResult = adjustedRandIndex(Map.of(
                        "label", originalLabel,
                        "true_labels", target_column
                ));

                double adjustedRandIndexValue = (double) adjustedRandIndexResult.get("adjustedRandIndex");
                updateProgress(sessions, "✅ Adjusted Rand Index Computed: " + adjustedRandIndexValue);

                adjustedRandIndexTime = System.nanoTime() - startAdjustedRandIndexTime;
                updateProgress(sessions, "✅ Step 4 Completed: Adjusted Rand Index Computation Done. Took " + (adjustedRandIndexTime / 1e6) + " ms.");

                long totalTime = System.nanoTime() - totalStartTime;
                updateProgress(sessions, "🕒 Total Execution Time: " + (totalTime / 1e6) + " ms.");

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

    private void updateProgress(Session session, String step) {
        log.info(step);
    }

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

                String graph_name = "affinity_";
                Double[][] adj_mat = null;
                ArrayList<NodeList2> node_properties_list = null;
                String[] remove_list = remove_columns.split(",");
                List<String> remove_list_new = Arrays.stream(remove_list).collect(Collectors.toList());
                if (label.isEmpty()) {
                    ReadCsvTestData readCsvTestData = new ReadCsvTestData(data_path);
                    node_properties_list = readCsvTestData.readCsvFileToMap(data_path);
                }
                org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodeData = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                node_properties_list = nodeData.getLeft();
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
            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list, node_list);
            output_string.append("\n\nAdjacency Matrix:\n").append(matrixToString(adjacency_matrix));

            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
            output_string.append("\n\nDegree Matrix:\n").append(matrixToString(degree_matrix));

            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
            output_string.append("\n\nLaplacian Matrix:\n").append(matrixToString(laplacian_matrix));

            Object eigen_result_obj = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);

            if (eigen_result_obj instanceof String) {
                return (String) eigen_result_obj;
            }

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
                        predictedNodeLabels.add(centroidNumber);

                        DecimalFormat decimalFormat = new DecimalFormat("#.###");
                        double distance = Unsupervised.calculateDistance(clusterNode.get(i), centroid, distance_measure);
                        connectNodes(label, "create relationship in kmean node", centroid, clusterNode.get(i), distance, connector.getDriver());
                    }
                    centroidNumber = centroidNumber + 1;
                }

                return output_string + " predicted labels: " + predictedNodeLabels;
            }
        }
    }

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

                    saveMatrixToCSV(adjacency_matrix, label + "_adjacency_matrix.csv", save_path);
                } else if ("x_matrix".equals(matrix_type)) {
                    org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodes = Neo4jGraphHandler.retrieveNodeList(label, connector.getDriver());
                    RealMatrix x_matrix = MatrixCalculation.convertToXMatrix(nodes.getLeft());
                    output_string.append("X matrix with eigenvectors:\n").append(matrixToString(x_matrix));

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

    public void saveMatrixToCSV(RealMatrix matrix, String filename, String save_path) throws IOException {
        String file_path = (save_path == null || save_path.isEmpty()) ? "./" + filename : save_path + "/" + filename;

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
                    sb.append(" ");
                }
            }
            sb.append("\n");
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

        for (HashMap.Entry<String, ArrayList<String>> entry : kmeanAssign.entrySet()) {
            String key = entry.getKey();
            ArrayList<String> values = entry.getValue();

            sb.append(key).append(": ");

            sb.append(values.toString());

            sb.append("\n");
        }

        return sb.toString();
    }
}
