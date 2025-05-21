package simkit;

import definition.NodeList2;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

import org.neo4j.driver.*;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class SimKitProceduresTest {

    private Driver driver;

    @BeforeAll
    void setUpDatabase() {
        driver = GraphDatabase.driver("neo4j://localhost:7687", AuthTokens.basic("neo4j", "123412345"));

        try (Session session = driver.session()) {
            // Delete previous test nodes to avoid duplicates
            session.run("MATCH (n:Iris) DETACH DELETE n");

            // Load data from CSV
            String csvFilePath = "./data/d1.csv";
            String csvContent = Files.readString(Paths.get(csvFilePath));

            // Prepare CSV to import folder dynamically
            session.run("LOAD CSV WITH HEADERS FROM 'file:///d1.csv' AS row " +
                    "CREATE (:Iris { " +
                    "  index: toInteger(row.index), " +
                    "  sepal_length: toFloat(row.sepal_length), " +
                    "  sepal_width: toFloat(row.sepal_width), " +
                    "  petal_length: toFloat(row.petal_length), " +
                    "  petal_width: toFloat(row.petal_width), " +
                    "  target: row.target " +
                    "});"
            );
        } catch (Exception e) {
            fail("Failed to load CSV data: " + e.getMessage());
        }
    }

    @AfterAll
    void tearDownDatabase() {
        try (Session session = driver.session()) {
            session.run("MATCH (n:Iris) DETACH DELETE n");
        }
        driver.close();
    }

    private final SimKitProcedures simKitProcedures = new SimKitProcedures();

    @Test
    void initSimKit() {
        SimKitProcedures procedures = new SimKitProcedures();
        String result = procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");
        assertEquals("Initialization succesful", result);
    }

    @Test
    void csvToGraph() {
        SimKitProcedures procedures = new SimKitProcedures();

        String dataPath = "./data/junit.csv";
        String distanceMeasure = "euclidean";
        String graphType = "full";
        String parameter = "7";
        String removeColumns = "";

        try {
            String result = procedures.csvToGraph(dataPath, distanceMeasure, graphType, parameter, removeColumns);

            assertNotNull(result, "Result should not be null");
            assertTrue(result.contains("Create fully connected graph successful"),
                    "Result should indicate successful graph creation");
        } catch (Exception e) {
            fail("Exception thrown during test execution: " + e.getMessage());
        }
    }

    @Test
    void getDistanceMatrixFromNodes_euclidean() {
        ArrayList<NodeList2> nodeList = new ArrayList<>();

        Map<String, Object> properties1 = new HashMap<>();
        properties1.put("property1", 1.0);
        properties1.put("property2", 2.0);

        Map<String, Object> properties2 = new HashMap<>();
        properties2.put("property1", 4.0);
        properties2.put("property2", 6.0);

        nodeList.add(new NodeList2(1.0f, properties1));
        nodeList.add(new NodeList2(2.0f, properties2));

        List<String> removeList = Arrays.asList("property2");

        Double[][] distanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes("euclidean", nodeList, removeList);

        assertNotNull(distanceMatrix, "Euclidean distance matrix should not be null");
        assertEquals(nodeList.size(), distanceMatrix.length, "Matrix size should match node list size");
    }

    @Test
    void getDistanceMatrixFromNodes_invalidMeasure() {
        ArrayList<NodeList2> nodeList = new ArrayList<>();
        nodeList.add(new NodeList2(/* initialize properties */));
        nodeList.add(new NodeList2(/* initialize properties */));

        List<String> removeList = new ArrayList<>();

        Double[][] distanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes("invalid_measure", nodeList, removeList);

        assertNull(distanceMatrix, "Distance matrix should be null for invalid measure");
    }

    @Test
    void getDistanceMatrixFromNodes_emptyNodeList() {
        ArrayList<NodeList2> nodeList = new ArrayList<>();
        List<String> removeList = new ArrayList<>();

        Double[][] distanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes("euclidean", nodeList, removeList);

        assertNotNull(distanceMatrix, "Distance matrix should not be null even for empty node list");
        assertEquals(0, distanceMatrix.length, "Distance matrix should have size zero for empty node list");
    }

    @Test
    void testNodePropertyToGraphWithValidParams() {
        SimKitProcedures procedures = new SimKitProcedures();
        String _result = procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("label", "Iris");
        params.put("distance_measure", "euclidean");
        params.put("graph_type", "full");
        params.put("parameter", "7");
        params.put("remove_column", "index,target");

        String result = null;
        try {
            result = procedures.nodePropertyToGraph(params);
        } catch (Exception e) {
            fail("Exception should not be thrown with valid parameters: " + e.getMessage());
        }

        assertNotNull(result);
        assertEquals("Create fully connected graph successful!", result);
    }

    @Test
    void testNodePropertyToGraphWithInvalidGraphType() {
        SimKitProcedures procedures = new SimKitProcedures();
        String _result = procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("label", "Iris");
        params.put("distance_measure", "euclidean");
        params.put("graph_type", "invalid_graph");
        params.put("parameter", "7");
        params.put("remove_column", "index,target");

        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            procedures.nodePropertyToGraph(params);
        });

        String expectedMessage = "Invalid graph_type specified.";
        String actualMessage = exception.getCause() != null
                ? exception.getCause().getMessage()
                : exception.getMessage();

        System.out.println("Actual message: " + actualMessage); // For debugging

        assertTrue(actualMessage.contains(expectedMessage),
                "Expected message: " + expectedMessage + ", Actual message: " + actualMessage);
    }

    @Test
    void testNodePropertyToGraphWithMissingParams() {
        SimKitProcedures procedures = new SimKitProcedures();
        String _result = procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("label", null);
        params.put("distance_measure", null);

        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            procedures.nodePropertyToGraph(params);
        });

        String expectedMessage = "Missing dataPath or distance measure type";
        String actualMessage = exception.getCause() != null
                ? exception.getCause().getMessage()
                : exception.getMessage();

        System.out.println("Actual message: " + actualMessage); // For debugging

        assertTrue(actualMessage.contains(expectedMessage),
                "Expected message: " + expectedMessage + ", Actual message: " + actualMessage);
    }

    @Test
    void testNodePropertyEigenWithValidParams() {
        SimKitProcedures procedures = new SimKitProcedures();

        String _result = procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "affinity_full_7_Iris");
        params.put("laplacian_type", "sym");
        params.put("number_of_eigenvectors", 3.0);

        String result = null;
        try {
            result = procedures.nodePropertyEigen(params);
        } catch (Exception e) {
            fail("Exception should not be thrown with valid parameters: " + e.getMessage());
        }

        assertNotNull(result);
        assertEquals("Create eigendecomposed graph successful!", result);
    }

    @Test
    void testNodePropertyEigenWithInvalidNodeLabel() {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "nonexistent_label");
        params.put("laplacian_type", "sym");
        params.put("number_of_eigenvectors", 3.0);

        Exception exception = assertThrows(RuntimeException.class, () -> {
            procedures.nodePropertyEigen(params);
        });

        String expectedMessage = "No valid identifier found for nodes with label";
        String actualMessage = exception.getCause() != null
                ? exception.getCause().getMessage()
                : exception.getMessage();

        assertTrue(actualMessage.contains(expectedMessage),
                "Expected message containing: " + expectedMessage + ", Actual message: " + actualMessage);
    }

    @Test
    void testNodePropertyEigenWithInvalidEigenvectors() {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "affinity_full_7_Iris");
        params.put("laplacian_type", "sym");
        params.put("number_of_eigenvectors", -1.0);

        Exception exception = assertThrows(Exception.class, () -> {
            procedures.nodePropertyEigen(params);
        });

        String expectedMessage = "Invalid input";
        String actualMessage = exception.getMessage();

        assertTrue(actualMessage.contains(expectedMessage),
                "Expected message containing: " + expectedMessage + ", Actual message: " + actualMessage);
    }

    @Test
    void testMapNodesWithValidParams() throws Exception {
        // Initialize your SimKitProcedures
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        // Define your test parameters
        String nodeSet = "Iris";
        String overlook = "index,target";

        // Invoke the method under test
        String result = procedures.mapNodes(nodeSet, overlook);

        // Assert the result is not null and contains expected structure
        assertNotNull(result);
        assertTrue(result.startsWith("Map all node data: "));
        assertTrue(result.contains(":"));
    }

    @Test
    void testMapNodesWithEmptyOverlook() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        String nodeSet = "Iris";
        String overlook = "";

        String result = procedures.mapNodes(nodeSet, overlook);

        assertNotNull(result);
        assertTrue(result.startsWith("Map all node data: "));
        assertTrue(result.contains(":"));
    }

    @Test
    void testKMeansWithValidParams() {
        SimKitProcedures procedures = new SimKitProcedures();

        Map<String, Object> params = new HashMap<>();
        params.put("nodeSet", "eigen_sym_3_affinity_full_7_Iris");
        params.put("numberOfCentroid", "3");
        params.put("numberOfInteration", "100");
        params.put("distanceMeasure", "euclidean");
        params.put("originalSet", "Iris");
        params.put("overlook", "target,sepal_length,sepal_width,petal_length,petal_width");
        params.put("overlookOriginal", "target");
        params.put("useKmeanForSilhouette", true);
        params.put("seed", 42);

        double silhouetteCoefficient = 0.0;
        try {
            silhouetteCoefficient = procedures.kMeans(params);
        } catch (Exception e) {
            fail("Exception should not be thrown with valid parameters: " + e.getMessage());
        }

        assertTrue(silhouetteCoefficientValid(silhouetteCoefficient));
    }

    @Test
    void testKMeansWithInvalidNodeSet() {
        SimKitProcedures procedures = new SimKitProcedures();

        Map<String, Object> params = new HashMap<>();
        params.put("nodeSet", "nonexistent_node_set");
        params.put("numberOfCentroid", "3");
        params.put("numberOfInteration", "100");
        params.put("distanceMeasure", "euclidean");
        params.put("originalSet", "Iris");
        params.put("overlook", "target,sepal_length,sepal_width,petal_length,petal_width");
        params.put("overlookOriginal", "target");
        params.put("useKmeanForSilhouette", false);
        params.put("seed", 42);

        Exception exception = assertThrows(Exception.class, () -> {
            procedures.kMeans(params);
        });

        String expectedMessage = "No nodes found with the nodeSet";
        String actualMessage = exception.getMessage();

        assertTrue(actualMessage.contains(expectedMessage));
    }

    @Test
    void testKMeansWithInvalidDistanceMeasure() {
        SimKitProcedures procedures = new SimKitProcedures();

        Map<String, Object> params = new HashMap<>();
        params.put("nodeSet", "eigen_sym_3_affinity_full_7_Iris");
        params.put("numberOfCentroid", "3");
        params.put("numberOfInteration", "100");
        params.put("distanceMeasure", "invalid_measure");
        params.put("originalSet", "Iris");
        params.put("overlook", "target,sepal_length,sepal_width,petal_length,petal_width");
        params.put("overlookOriginal", "target");
        params.put("useKmeanForSilhouette", false);
        params.put("seed", 42);

        Exception exception = assertThrows(Exception.class, () -> {
            procedures.kMeans(params);
        });

        String actualMessage = exception.getMessage();

        assertTrue(actualMessage.contains("Invalid distance measure"));
    }

    private boolean silhouetteCoefficientValid(double coefficient) {
        return coefficient >= -1.0 && coefficient <= 1.0;
    }


    @Test
    void testAdjustedRandIndexPredictedLabelsEmpty() {
        SimKitProcedures simKitProcedures = new SimKitProcedures();
        simKitProcedures.predictedNodeLabels.clear();

        Map<String, Object> params = new HashMap<>();
        params.put("nodeSet", "Iris");
        params.put("trueLabels", "target");

        Exception exception = assertThrows(Exception.class, () -> simKitProcedures.adjustedRandIndex(params));

        assertTrue(exception.getMessage().contains("predicted Labels is null"));
    }

    @Test
    void testAdjustedRandIndexWithSingleCluster() throws Exception {
        SimKitProcedures simKitProcedures = new SimKitProcedures();
        simKitProcedures.predictedNodeLabels = Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0);

        Map<String, Object> params = new HashMap<>();
        params.put("nodeSet", "Iris");
        params.put("trueLabels", "target");

        Map<String, Object> result = simKitProcedures.adjustedRandIndex(params);
        assertEquals(1.0, (double) result.get("adjustedRandIndex"), 1e-6);
    }

    @Test
    void testAverageSilhouetteCoefficientWithValidInputs() throws Exception {
        String result = simKitProcedures.averageSilhouetteCoefficient("Iris", "3", "100", "euclidean", 42);
        assertNotNull(result);
        assertTrue(result.startsWith("The average Silhouette Coefficient value is:"));
    }

    @Test
    void testAverageSilhouetteCoefficientWithNullNodeSet() throws Exception {
        String result = simKitProcedures.averageSilhouetteCoefficient(null, "3", "100", "euclidean", 42);
        assertNull(result);
    }

    @Test
    void testAverageSilhouetteCoefficientWithInvalidCentroids() throws Exception {
        assertThrows(NumberFormatException.class, () ->
                simKitProcedures.averageSilhouetteCoefficient("Iris", "invalid", "100", "euclidean", 42)
        );
    }

    @Test
    void testAverageSilhouetteCoefficientWithInvalidIterations() throws Exception {
        assertThrows(NumberFormatException.class, () ->
                simKitProcedures.averageSilhouetteCoefficient("Iris", "3", "invalid", "euclidean", 42)
        );
    }

    @Test
    void testAverageSilhouetteCoefficientWithInvalidDistanceMeasure() throws Exception {
        assertThrows(IllegalArgumentException.class, () ->
                simKitProcedures.averageSilhouetteCoefficient("Iris", "3", "100", "invalidMeasure", 42)
        );
    }

    @Test
    void testGetNodeValuesFiltersOverlookKeys() {
        SimKitProcedures procedures = new SimKitProcedures();

        Map<String, Object> properties = new HashMap<>();
        properties.put("index", 1.0);
        properties.put("sepal_length", 5.1);
        properties.put("sepal_width", 3.5);
        properties.put("species", "setosa");

        Value value = Values.value(properties);
        String[] overlook = {"index", "species"};

        String result = procedures.getNodeValues(value, overlook);

        // Only sepal_length and sepal_width should be included
        assertTrue(result.contains("sepal_length:5.1"));
        assertTrue(result.contains("sepal_width:3.5"));
        assertFalse(result.contains("index:"));
        assertFalse(result.contains("species:"));
    }

    @Test
    void calculateAdjustedRandIndex_perfectMatch() {
        SimKitProcedures procedures = new SimKitProcedures();
        List<Double> trueLabels = Arrays.asList(1.0, 1.0, 2.0, 2.0);
        List<Double> predictedLabels = Arrays.asList(1.0, 1.0, 2.0, 2.0);

        double ari = procedures.calculateAdjustedRandIndex(trueLabels, predictedLabels);
        assertEquals(1.0, ari, 1e-6, "ARI should be 1.0 for perfect match");
    }

    @Test
    void calculateAdjustedRandIndex_randomMatch() {
        SimKitProcedures procedures = new SimKitProcedures();
        List<Double> trueLabels = Arrays.asList(1.0, 1.0, 2.0, 2.0);
        List<Double> predictedLabels = Arrays.asList(1.0, 2.0, 1.0, 2.0);

        double ari = procedures.calculateAdjustedRandIndex(trueLabels, predictedLabels);
        assertTrue(ari < 0.1, "ARI should be close to 0 for random match");
    }

    @Test
    void calculateAdjustedRandIndex_partialMatch() {
        SimKitProcedures procedures = new SimKitProcedures();
        List<Double> trueLabels = Arrays.asList(1.0, 1.0, 2.0, 2.0, 3.0, 3.0);
        List<Double> predictedLabels = Arrays.asList(1.0, 1.0, 2.0, 3.0, 2.0, 3.0);

        double ari = procedures.calculateAdjustedRandIndex(trueLabels, predictedLabels);
        assertTrue(ari > 0.0 && ari < 1.0, "ARI should be between 0 and 1 for partial match");
    }

    @Test
    void calculateAdjustedRandIndex_completelyDifferent() {
        SimKitProcedures procedures = new SimKitProcedures();
        List<Double> trueLabels = Arrays.asList(1.0, 1.0, 2.0, 2.0);
        List<Double> predictedLabels = Arrays.asList(3.0, 3.0, 4.0, 4.0);

        double ari = procedures.calculateAdjustedRandIndex(trueLabels, predictedLabels);
        assertEquals(1.0, ari, 1e-6, "Different labels but same clustering structure should give ARI = 1");
    }

    @Test
    void calculateAdjustedRandIndex_throwsOnDifferentSizes() {
        SimKitProcedures procedures = new SimKitProcedures();
        List<Double> trueLabels = Arrays.asList(1.0, 1.0, 2.0);
        List<Double> predictedLabels = Arrays.asList(1.0, 1.0);

        assertThrows(IllegalArgumentException.class, () ->
                procedures.calculateAdjustedRandIndex(trueLabels, predictedLabels)
        );
    }

    @Test
    void spectralClustering_validParams_featureBased_fullGraph() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "Iris");
        params.put("is_feature_based", true);
        params.put("distance_measure", "euclidean");
        params.put("graph_type", "full");
        params.put("parameter", "7");
        params.put("remove_columns", "index,target");
        params.put("laplacian_type", "sym");
        params.put("number_of_eigenvectors", 3.0);
        params.put("number_of_iterations", "100");
        params.put("distance_measure_kmean", "euclidean");
        params.put("target_column", "target");
        params.put("use_kmean_for_silhouette", true);
        params.put("seed", 42);

        double result = procedures.spectralClustering(params);
        assertTrue(result >= -1.0 && result <= 1.0, "Silhouette coefficient should be between -1 and 1");
    }

    @Test
    void spectralClustering_validParams_nonFeatureBased() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "affinity_full_7_Iris");
        params.put("is_feature_based", false);
        params.put("laplacian_type", "sym");
        params.put("number_of_eigenvectors", 3.0);
        params.put("number_of_iterations", "100");
        params.put("distance_measure_kmean", "euclidean");
        params.put("target_column", "target");
        params.put("use_kmean_for_silhouette", false);
        params.put("seed", 42);

        double result = procedures.spectralClustering(params);
        assertTrue(result >= -1.0 && result <= 1.0, "Silhouette coefficient should be between -1 and 1");
    }

    @Test
    void spectralClustering_missingNodeLabel() {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", null);

        Exception exception = assertThrows(Exception.class, () -> {
            procedures.spectralClustering(params);
        });

        assertTrue(exception.getMessage().contains("Missing node label"));
    }

    @Test
    void spectralClustering_invalidGraphType() {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "Iris");
        params.put("graph_type", "invalid_graph");

        Exception exception = assertThrows(Exception.class, () -> {
            procedures.spectralClustering(params);
        });

        assertTrue(exception.getMessage().contains("Invalid graph_type specified."));
    }

    @Test
    void experimental_spectralClustering_validParams_featureBased() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "Iris");
        params.put("is_feature_based", true);
        params.put("distance_measure", "euclidean");
        params.put("graph_type", "full");
        params.put("parameter", "7");
        params.put("remove_columns", "index,target");
        params.put("laplacian_type", "sym");
        params.put("number_of_eigenvectors", 3.0);
        params.put("number_of_iterations", "100");
        params.put("distance_measure_kmean", "euclidean");
        params.put("target_column", "target");
        params.put("use_kmean_for_silhouette", true);
        params.put("seed", 42);

        Map<String, Object> result = procedures.experimental_spectralClustering(params);

        assertNotNull(result);
        assertTrue(result.containsKey("silhouette_score"));
        assertTrue(result.containsKey("rand_index"));
        assertTrue(result.containsKey("total_time"));
        assertTrue((Double) result.get("silhouette_score") >= -1.0 && (Double) result.get("silhouette_score") <= 1.0);
    }

    @Test
    void experimental_spectralClustering_validParams_nonFeatureBased() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "affinity_full_7_Iris");
        params.put("is_feature_based", false);
        params.put("laplacian_type", "sym");
        params.put("number_of_eigenvectors", 3.0);
        params.put("number_of_iterations", "100");
        params.put("distance_measure_kmean", "euclidean");
        params.put("target_column", "target");
        params.put("use_kmean_for_silhouette", false);
        params.put("seed", 42);

        Map<String, Object> result = procedures.experimental_spectralClustering(params);

        assertNotNull(result);
        assertTrue(result.containsKey("rand_index"));
        assertTrue(result.containsKey("laplacian_time"));
    }

    @Test
    void experimental_spectralClustering_missingNodeLabel() {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", null);

        Exception exception = assertThrows(Exception.class, () -> {
            procedures.experimental_spectralClustering(params);
        });

        assertTrue(exception.getMessage().contains("Missing node label"));
    }

    @Test
    void experimental_spectralClustering_invalidGraphType() {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        Map<String, Object> params = new HashMap<>();
        params.put("node_label", "Iris");
        params.put("graph_type", "invalid_graph");

        Exception exception = assertThrows(Exception.class, () -> {
            procedures.experimental_spectralClustering(params);
        });

        assertTrue(exception.getMessage().contains("Invalid graph_type specified."));
    }

    @Test
    void displayEdgeList_validParams_fullGraph() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        String nodeType = "Iris";
        String dataPath = "./data/d1.csv";
        String distanceMeasure = "euclidean";
        String graphType = "full";
        String method = "dummy"; // currently unused
        String parameter = "7";
        String removeColumns = "index,target";

        String result = procedures.displayEdgeList(nodeType, dataPath, distanceMeasure, graphType, method, parameter, removeColumns);

        assertNotNull(result);
        assertTrue(result.contains("Distance Matrix"));
        assertTrue(result.contains("Adjacency Matrix"));
        assertTrue(result.contains("Edge List"));
    }

    @Test
    void displayEdgeList_validParams_epsGraph() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        String nodeType = "Iris";
        String dataPath = "./data/d1.csv";
        String distanceMeasure = "euclidean";
        String graphType = "eps";
        String method = "dummy"; // currently unused
        String parameter = "0.5";
        String removeColumns = "index,target";

        String result = procedures.displayEdgeList(nodeType, dataPath, distanceMeasure, graphType, method, parameter, removeColumns);

        assertNotNull(result);
        assertTrue(result.contains("Distance Matrix"));
        assertTrue(result.contains("Adjacency Matrix"));
        assertTrue(result.contains("Edge List"));
    }

    @Test
    void displayEdgeList_missingParams() throws Exception {
        SimKitProcedures procedures = new SimKitProcedures();
        procedures.initSimKit("neo4j://localhost:7687", "neo4j", "123412345");

        String nodeType = "Iris";
        String dataPath = null;
        String distanceMeasure = null;
        String graphType = "full";
        String method = "dummy"; // currently unused
        String parameter = "7";
        String removeColumns = "index,target";

        String result = procedures.displayEdgeList(nodeType, dataPath, distanceMeasure, graphType, method, parameter, removeColumns);

        assertEquals("Missing data_path or distance measure type", result);
    }
}