package simkit;


import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;

/**
 * @author 49171
 * This class contains implementations for unsupervised learning algorithms like K-Means and DBSCAN,
 * optimized for parallel execution to improve performance on multi-core systems.
 */
public class Unsupervised {
    public static ArrayList<String> dummyData() {
        ArrayList<String> data = new ArrayList<String>();
        String line1 = "y_cordinate:7.0,x_cordinate:1.0";
        String line2 = "y_cordinate:6.0,x_cordinate:1.0";
        String line3 = "y_cordinate:2.0,x_cordinate:6.0";
        String line4 = "y_cordinate:1.0,x_cordinate:8.0";
        String line5 = "y_cordinate:2.0,x_cordinate:10.0";
        data.add(line1);
        data.add(line2);
        data.add(line3);
        data.add(line4);
        data.add(line5);
        return data;
    }

    public static void main(String[] args) throws Exception {
        ArrayList<String> inputData = dummyData();
        ArrayList<String> debug = new ArrayList<>();
        HashMap<String, ArrayList<String>> dbAssign = KmeanClust(inputData, 2, 20, "Euclidean", false, debug, 42);
        for (String centroid : dbAssign.keySet()) {
            System.out.println("1");
            ArrayList<String> clusterNode = dbAssign.get(centroid);
            System.out.println("number of centroid " + centroid);
            System.out.println(clusterNode.toString());
        }
        Double silhouetteValue = averageSilhouetteCoefficient(dbAssign, "Euclidean");
        System.out.println(averageSilhouetteCoefficient(dbAssign, "Euclidean"));

    }
/**
 -------------------------------------------------------------------------------DBSCAN Clustering ------------------------------------------------------------------------------------------
 */
    /**
     * Determines the neighbours of a given input value.
     *
     * @param inputValue value for which neighbours are to be calculated
     * @return list of neighbours
     */
    public static ArrayList<String> getNeighbours(final String inputValue, final String distanceMetric) {
        ArrayList<String> neighbours = new ArrayList<String>();
        double distance = 0;
        for (int i = 0; i < inputValues.size(); i++) {
            String candidate = inputValues.get(i);
            if (distanceMetric.equals("bray-curtis")) {
                distance = calBrayCurtis(inputValue, candidate);
            } else if (distanceMetric.equals("manhattan")) {
                distance = calManhattanDist(inputValue, candidate);
            } else if (distanceMetric.equals("cosine")) {
                distance = calCosineSimilarity(inputValue, candidate);
            } else {
                distance = calEuclideanDist(inputValue, candidate);
            }
            System.out.println(distance);
            if (distance <= epsilon) {
                neighbours.add(candidate);
            }
        }
        return neighbours;
    }

    /**
     * Merges the elements of the right collection to the left one and returns
     * the combination.
     *
     * @param neighbours1 left collection
     * @param neighbours2 right collection
     * @return Modified left collection
     */
    public static ArrayList<String> mergeRightToLeftCollection(final ArrayList<String> neighbours1, final ArrayList<String> neighbours2) {
        for (int i = 0; i < neighbours2.size(); i++) {
            String tempPt = neighbours2.get(i);
            if (!neighbours1.contains(tempPt)) {
                neighbours1.add(tempPt);
            }
        }
        return neighbours1;
    }

    /**
     * Applies the clustering and returns a collection of clusters (i.e. a list
     * of lists of the respective cluster members).
     *
     * @return
     */
    public static double epsilon;
    public static ArrayList<String> inputValues;

    public static HashMap<String, ArrayList<String>> DbClust(final ArrayList<String> inputData, double eps, int minPts, String distanceMetric) {

        inputValues = inputData;
        epsilon = eps;
        HashMap<String, ArrayList<String>> resultHashmap = new HashMap<String, ArrayList<String>>();
        ArrayList<ArrayList<String>> resultList = new ArrayList<ArrayList<String>>();
        ArrayList<String> visitedPoints = new ArrayList<String>();
        visitedPoints.clear();

        ArrayList<String> neighbours;
        int index = 0;
        while (inputValues.size() > index) {
            String p = inputValues.get(index);
            if (!visitedPoints.contains(p)) {
                visitedPoints.add(p);
                neighbours = getNeighbours(p, distanceMetric);
                if (neighbours.size() >= minPts) {
                    int ind = 0;
                    while (neighbours.size() > ind) {
                        String r = neighbours.get(ind);
                        if (!visitedPoints.contains(r)) {
                            visitedPoints.add(r);
                            ArrayList<String> individualNeighbours = getNeighbours(r, distanceMetric);
                            if (individualNeighbours.size() >= minPts) {
                                neighbours = mergeRightToLeftCollection(neighbours, individualNeighbours);
                            }
                        }
                        ind++;
                    }
                    resultList.add(neighbours);
                }
            }
            index++;
        }
        for (ArrayList<String> cluster : resultList) {
            String newCentroid = calculateNewCentroid(cluster);
            resultHashmap.put(newCentroid, cluster);
        }
        return resultHashmap;
    }

    /**
     * -------------------------------------------------------------------------------K-MEANS Clustering ------------------------------------------------------------------------------------------
     */


    static HashMap<String, ArrayList<String>> replaceValuesWithOriginalSet(
            HashMap<String, ArrayList<String>> clusterAssign,
            ArrayList<String> originalNodeSet
    ) {
// Create a mapping from ID to original node (without 'eigenvector' attributes)
        HashMap<String, String> idToOriginalNode = new HashMap<>();
        for (String originalNode : originalNodeSet) {
// Remove attributes starting with 'eigenvector'
            String cleanedNode = removeEigenvectorAttributes(originalNode);

            String id = extractID(cleanedNode);
            if (id != null) {
                idToOriginalNode.put(id, cleanedNode);
            }
        }

// Initialize the updated cluster assignments
        HashMap<String, ArrayList<String>> updatedClusterAssign = new HashMap<>();

// Keep track of all node IDs that have been assigned to clusters
        Set<String> assignedNodeIDs = new HashSet<>();

// Replace centroids and cluster nodes with values from originalNodeSet
        for (Map.Entry<String, ArrayList<String>> entry : clusterAssign.entrySet()) {
            String centroid = entry.getKey();
            ArrayList<String> clusterNodes = entry.getValue();

// Replace centroid
            String centroidID = extractID(centroid);
            float floatCentroidID = Float.parseFloat(centroidID);
            int centroidIDInt = (int) floatCentroidID;
            String originalCentroid = idToOriginalNode.getOrDefault(String.valueOf(centroidIDInt), centroid);

// Replace cluster nodes
            ArrayList<String> updatedClusterNodes = new ArrayList<>();
            for (String node : clusterNodes) {
                String nodeID = extractID(node);
                String originalNode = idToOriginalNode.getOrDefault(nodeID, node);
                updatedClusterNodes.add(originalNode);
                assignedNodeIDs.add(nodeID);
            }

            updatedClusterAssign.put(originalCentroid, updatedClusterNodes);
            assignedNodeIDs.add(centroidID);
        }

// Add all original nodes to the clusters if they haven't been assigned yet
        ArrayList<String> unassignedNodes = new ArrayList<>();
        for (String originalNode : idToOriginalNode.values()) {
            String nodeID = extractID(originalNode);
            if (!assignedNodeIDs.contains(nodeID)) {
                unassignedNodes.add(originalNode);
            }
        }

// Optionally, assign unassigned nodes to a separate cluster or distribute them
        if (!unassignedNodes.isEmpty()) {
            updatedClusterAssign.put("Unassigned", unassignedNodes);
        }

// Define the prefixes of attributes to remove
        String[] prefixesToRemove = new String[]{"index", "id"};

// Create a new HashMap to store the cleaned entries
        HashMap<String, ArrayList<String>> cleanedClusterAssign = new HashMap<>();

        for (Map.Entry<String, ArrayList<String>> entry : updatedClusterAssign.entrySet()) {
            String key = entry.getKey();
            ArrayList<String> valueList = entry.getValue();

// Remove 'index' and 'id' attributes from the key
            String cleanedKey = removeAttributes(key, prefixesToRemove);

// Create a new list to store cleaned values
            ArrayList<String> cleanedValueList = new ArrayList<>();

// Iterate over the values in the valueList
            for (String value : valueList) {
// Remove 'index' and 'id' attributes from the value
                String cleanedValue = removeAttributes(value, prefixesToRemove);
// Add the cleaned value to the cleanedValueList
                cleanedValueList.add(cleanedValue);
            }

// Put the cleaned key and cleaned value list into the cleanedClusterAssign
            cleanedClusterAssign.put(cleanedKey, cleanedValueList);
        }

// Replace the original updatedClusterAssign with the cleaned version
        updatedClusterAssign = cleanedClusterAssign;

        return updatedClusterAssign;
    }

    /**
     * Removes attributes starting with specified prefixes from a node string.
     *
     * @param node     The node string
     * @param prefixes The array of prefixes to remove
     * @return The node string without the specified attributes
     */
    private static String removeAttributes(String node, String[] prefixes) {
        String[] attributes = node.split(",");
        ArrayList<String> filteredAttributes = new ArrayList<>();
        for (String attr : attributes) {
            String[] keyValue = attr.split(":");
            if (keyValue.length < 2) continue; // Skip malformed attributes

            String key = keyValue[0].trim().toLowerCase();
            boolean remove = false;
            for (String prefix : prefixes) {
                if (key.startsWith(prefix.toLowerCase())) {
                    remove = true;
                    break;
                }
            }
            if (!remove) {
                filteredAttributes.add(attr);
            }
        }
        return String.join(",", filteredAttributes);
    }

    /**
     * Removes attributes starting with 'eigenvector' from a node string.
     *
     * @param node The node string
     * @return The node string without 'eigenvector' attributes
     */
    private static String removeEigenvectorAttributes(String node) {
        String[] attributes = node.split(",");
        ArrayList<String> filteredAttributes = new ArrayList<>();
        for (String attr : attributes) {
            String key = attr.split(":")[0].trim();
            if (!key.startsWith("eigenvector")) {
                filteredAttributes.add(attr);
            }
        }
        return String.join(",", filteredAttributes);
    }

    /**
     * Extracts the ID from a node string.
     * Assumes the ID is in the format "ID:value"
     *
     * @param node The node string
     * @return The extracted ID, or null if not found
     */
    private static String extractID(String node) {
        String[] attributes = node.split(",");
        for (String attr : attributes) {
            String[] keyValue = attr.split(":");
            if (keyValue[0].trim().equalsIgnoreCase("index") || keyValue[0].trim().equalsIgnoreCase("id")) {
                return keyValue[1].trim();
            }
        }
        return null; // ID not found
    }

    // Initialize the random generator on the FastAPI server with a specific seed
    private static void initializeRandomGenerator(int seed) throws Exception {
        String urlString = "http://91.107.235.104:8000/init?seed=" + seed;
        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.getResponseCode(); // Ensure the request completes
        connection.disconnect();
    }

    // Fetch the next random integer from the FastAPI server
    private static int getNextRandomInt(int maxValue) throws Exception {
        String urlString = "http://91.107.235.104:8000/next-int?max_value=" + maxValue;
        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");

        BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        String inputLine;
        StringBuilder content = new StringBuilder();

        while ((inputLine = in.readLine()) != null) {
            content.append(inputLine);
        }

        in.close();
        connection.disconnect();

        JSONObject jsonResponse = new JSONObject(content.toString());
        return jsonResponse.getInt("next_int");
    }

    /**
     * This is the main method to perform k-means clustering.
     *
     * @param inputData          is a variable where the nodes from Neo4j are stored
     * @param numberOfCentroids  store the number of centroids specified by user for clustering
     * @param numberOfInteration saves user specified iteration to find convergence
     * @param distanceMeasure    the distance measure to use
     * @param useOriginalNodeSet boolean to control whether to replace values with originalNodeSet
     * @param originalNodeSet    the original nodes to replace in the output
     * @return HashMap with centroids and their assigned nodes
     */
    public static HashMap<String, ArrayList<String>> KmeanClust(
            ArrayList<String> inputData,
            int numberOfCentroids,
            int numberOfInteration,
            String distanceMeasure,
            boolean useOriginalNodeSet,
            ArrayList<String> originalNodeSet,
            int seed
    ) throws Exception {
        HashMap<String, ArrayList<String>> kmeanAssign = new HashMap<>();
        ArrayList<String> listOfCentroid = new ArrayList<>();
        ArrayList<String> listOfRemain = new ArrayList<>(inputData);

        // Initializing centroids by random choice
        initializeRandomGenerator(seed);

        while (listOfCentroid.size() < numberOfCentroids) {
            int randomIndex = getNextRandomInt(inputData.size()); // Fetch the next random index
            String potentialCentroid = inputData.get(randomIndex);

            if (!listOfCentroid.contains(potentialCentroid)) {
                listOfCentroid.add(potentialCentroid);
            }
        }

        // Remove centroids from the remaining list
        listOfRemain.removeAll(listOfCentroid);

        // First clusters
        HashMap<String, ArrayList<String>> hashClusterAssign = distanceAssign(listOfCentroid, listOfRemain, distanceMeasure);
        // All iterations
        kmeanAssign = kmeanIteration(hashClusterAssign, numberOfInteration, inputData, distanceMeasure);

        return kmeanAssign;
    }

    /**
     * Method to perform the iterations of k-means
     *
     * @param clusterAssign      contains the first cluster assignments
     * @param numberOfInteration specified by user
     * @param inputData          specified by user
     * @return
     */
    public static HashMap<String, ArrayList<String>> kmeanIteration(HashMap<String, ArrayList<String>> clusterAssign, int numberOfInteration, ArrayList<String> inputData, String distanceMeasure) {
        for (int i = 0; i < numberOfInteration; i++) {
            // Calculate new centroids and update clusterAssign
            clusterAssign = calculateAndUpdateCentroids(clusterAssign);

            // Perform distance assignment again with the updated centroids
            clusterAssign = distanceAssign(new ArrayList<>(clusterAssign.keySet()), inputData, distanceMeasure);
        }
        return clusterAssign;
    }

    /**
     * Method to calculate new centroid points after each iteration.
     * This method is optimized to run in parallel.
     *
     * @param listOfNodesInCluster nodes assigned to each cluster
     * @return returns new centroids after each iteration
     */
    public static String calculateNewCentroid(ArrayList<String> listOfNodesInCluster) {
        if (listOfNodesInCluster == null || listOfNodesInCluster.isEmpty()) {
            return null; // Handle empty cluster case
        }

        // Extract attribute names once from the first node.
        String[] firstNodeAttributes = listOfNodesInCluster.get(0).split(",");
        int numAttributes = firstNodeAttributes.length;
        String[] attributeNames = new String[numAttributes];

        for (int i = 0; i < numAttributes; i++) {
            attributeNames[i] = firstNodeAttributes[i].split(":")[0].trim();
        }

        // Use a parallel stream to compute the sum of attributes efficiently.
        double[] attributeSums = new double[numAttributes];

        listOfNodesInCluster.parallelStream()
                .map(node -> Arrays.stream(node.split(","))
                        .map(attr -> attr.split(":")[1].trim())
                        .mapToDouble(Double::parseDouble)
                        .toArray())
                .forEach(values -> {
                    // Synchronize access to the shared attributeSums array to ensure thread safety.
                    synchronized (attributeSums) {
                        for (int i = 0; i < numAttributes; i++) {
                            attributeSums[i] += values[i];
                        }
                    }
                });

        // Compute averages and construct the new centroid string using StringJoiner for efficiency.
        int totalNodes = listOfNodesInCluster.size();
        StringJoiner newCentroid = new StringJoiner(",");

        for (int i = 0; i < numAttributes; i++) {
            newCentroid.add(attributeNames[i] + ":" + (attributeSums[i] / totalNodes));
        }

        return newCentroid.toString();
    }

    /**
     * Method to calculate and update centroids in the clusterAssign.
     * This method is optimized to run in parallel.
     *
     * @param clusterAssign The current cluster assignments
     * @return Updated cluster assignments with new centroids
     */
    public static HashMap<String, ArrayList<String>> calculateAndUpdateCentroids(HashMap<String, ArrayList<String>> clusterAssign) {
        // Use ConcurrentHashMap for thread-safe parallel computation of new centroids.
        ConcurrentHashMap<String, ArrayList<String>> updatedClusterAssign = new ConcurrentHashMap<>();

        // Process each cluster in parallel to calculate its new centroid.
        clusterAssign.entrySet().parallelStream().forEach(entry -> {
            ArrayList<String> clusterNodes = entry.getValue();

            String newCentroid = calculateNewCentroid(clusterNodes);
            if (newCentroid != null) { // Handle cases where a cluster might become empty.
                updatedClusterAssign.put(newCentroid, clusterNodes);
            }
        });

        return new HashMap<>(updatedClusterAssign); // Convert back to a standard HashMap before returning.
    }

    /**
     * Assigns each point in `listOfRemain` to the nearest centroid.
     * This method is optimized to run in parallel.
     *
     * @param listOfCentroid List of current centroids.
     * @param listOfRemain List of points to be assigned.
     * @param distanceMeasure The distance metric to use.
     * @return A map of centroids to their assigned points.
     */
    public static HashMap<String, ArrayList<String>> distanceAssign(
            ArrayList<String> listOfCentroid, ArrayList<String> listOfRemain, String distanceMeasure) {

        HashMap<String, ArrayList<String>> hashClusterAssign = new HashMap<>();

        // Pre-compute cleaned versions of centroid & remain data to avoid redundant string operations in the loop.
        Map<String, String> cleanedRemain = new ConcurrentHashMap<>();
        listOfRemain.parallelStream().forEach(remain ->
                cleanedRemain.put(remain, removeIndexAndIdEntries(remain))
        );

        Map<String, String> cleanedCentroids = new ConcurrentHashMap<>();
        listOfCentroid.parallelStream().forEach(centroid ->
                cleanedCentroids.put(centroid, removeIndexAndIdEntries(centroid))
        );

        // Assign each point to the nearest centroid in parallel.
        listOfRemain.parallelStream().forEach(remain -> {
            double minDistance = Double.MAX_VALUE;
            String closestCentroid = null;
            String cleanedRemainEntry = cleanedRemain.get(remain);

            for (String centroid : listOfCentroid) {
                double distance = calculateDistance(cleanedRemainEntry, cleanedCentroids.get(centroid), distanceMeasure);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroid = centroid;
                }
            }

            // Synchronize access to the shared cluster assignment map for thread safety.
            if (closestCentroid != null) {
                synchronized (hashClusterAssign) {
                    hashClusterAssign.computeIfAbsent(closestCentroid, k -> new ArrayList<>()).add(remain);
                }
            }
        });

        // Ensure all centroids exist as keys in the map, even if no points are assigned to them.
        listOfCentroid.forEach(centroid -> hashClusterAssign.putIfAbsent(centroid, new ArrayList<>()));

        return hashClusterAssign;
    }

    /**
     * Removes "index" and "id" key-value pairs from a node string representation.
     * This version is optimized using the Stream API.
     * @param nodeString The input string (e.g., "id:1,prop1:val1").
     * @return The cleaned string.
     */
    public static String removeIndexAndIdEntries(String nodeString) {
        if (nodeString == null || nodeString.isEmpty()) {
            return ""; // Handle empty or null input safely.
        }

        return Arrays.stream(nodeString.split(","))
                .map(String::trim)
                .filter(entry -> entry.contains(":")) // Ensure it's a valid key-value pair.
                .filter(entry -> {
                    String key = entry.split(":")[0].trim().toLowerCase(); // Extract key and normalize case.
                    return !(key.equals("index") || key.equals("id")); // Exclude "index" and "id".
                })
                .collect(Collectors.joining(","));
    }


    static double calculateDistance(String point1, String point2, String distanceMeasure) {
        switch (distanceMeasure.toLowerCase()) {
            case "manhattan":
                return calManhattanDist(point1, point2);
            case "cosine":
                return calCosineSimilarity(point1, point2);
            case "bray-curtis":
                return calBrayCurtis(point1, point2);
            case "euclidean":
                return calEuclideanDist(point1, point2);
            default:
                throw new IllegalArgumentException("Unsupported distance measure: " + distanceMeasure);
        }
    }
/**
 --------------------------------------------------------------------------Distance Measures for Clustering ------------------------------------------------------------------------------------------
 */
    /**
     * Euclidean distance calculation from point A to point B
     *
     * @param start point A
     * @param end   point B
     * @return
     */
    private static double robustEuclideanDist(String start, String end) {
        if (start == null || end == null || start.isEmpty() || end.isEmpty()) {
            throw new IllegalArgumentException("Input vectors cannot be null or empty. start: '" + start + "', end: '" + end + "'");
        }
        String[] startSplit = start.split(",");
        String[] endSplit = end.split(",");

        if (startSplit.length != endSplit.length) {
            throw new IllegalArgumentException("Input vectors have different dimensions. start: " + startSplit.length + ", end: " + endSplit.length + ". start_vec: " + start + ", end_vec: " + end);
        }

        double distance = 0.0;
        for (int i = 0; i < startSplit.length; i++) {
            String[] startPair = startSplit[i].split(":");
            String[] endPair = endSplit[i].split(":");

            if (startPair.length < 2) {
                throw new IllegalArgumentException("Malformed start vector component: '" + startSplit[i] + "' in vector '" + start + "'");
            }
            if (endPair.length < 2) {
                throw new IllegalArgumentException("Malformed end vector component: '" + endSplit[i] + "' in vector '" + end + "'");
            }

            try {
                float startValue = Float.parseFloat(startPair[1].trim());
                float endValue = Float.parseFloat(endPair[1].trim());
                distance += Math.pow(startValue - endValue, 2);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Could not parse float from vector component. Start: '" + startPair[1] + "', End: '" + endPair[1] + "'", e);
            }
        }
        return Math.sqrt(distance);
    }

    public static double calEuclideanDist(String start, String end) {
        try {
            return robustEuclideanDist(start, end);
        } catch (IllegalArgumentException e) {
            System.err.println("Error in calEuclideanDist: " + e.getMessage());
            // Return a large distance or re-throw as a different exception type if needed
            return Double.MAX_VALUE;
        }
    }

    /**
     * Calculate Manhattan distance between point A and B
     *
     * @param start point A
     * @param end   point B
     * @return
     */
    public static double calManhattanDist(String start, String end) {
        if (start == null || end == null || start.isEmpty() || end.isEmpty()) return Double.MAX_VALUE;
        String[] startSplit = start.split(",");
        String[] endSplit = end.split(",");
        if (startSplit.length != endSplit.length) return Double.MAX_VALUE;

        double distance = 0.00;
        for (int i = 0; i < startSplit.length; i++) {
            String[] startPair = startSplit[i].split(":");
            String[] endPair = endSplit[i].split(":");
            if (startPair.length < 2 || endPair.length < 2) return Double.MAX_VALUE;

            float startValue = Float.parseFloat(startPair[1].trim());
            float endValue = Float.parseFloat(endPair[1].trim());
            distance = distance + Math.abs(startValue - endValue);
        }
        return distance;
    }

    /**
     * Calculate Cosine similarity between point A and B
     *
     * @param start point A
     * @param end   point B
     * @return
     */
    public static double calCosineSimilarity(String start, String end) {
        if (start == null || end == null || start.isEmpty() || end.isEmpty()) return 0.0;
        String[] startSplit = start.split(",");
        String[] endSplit = end.split(",");
        if (startSplit.length != endSplit.length) return 0.0;

        double dotProduct = 0.00;
        double normA = 0.00;
        double normB = 0.00;
        for (int i = 0; i < startSplit.length; i++) {
             String[] startPair = startSplit[i].split(":");
            String[] endPair = endSplit[i].split(":");
            if (startPair.length < 2 || endPair.length < 2) return 0.0;

            float startValue = Float.parseFloat(startPair[1].trim());
            float endValue = Float.parseFloat(endPair[1].trim());
            dotProduct += startValue * endValue;
            normA += Math.pow(startValue, 2);
            normB += Math.pow(endValue, 2);
        }
        if (normA == 0 || normB == 0) return 0.0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Calculate Bray-Curtis dissimilarity between point A and B
     *
     * @param start point A
     * @param end   point B
     * @return
     */
    public static double calBrayCurtis(String start, String end) {
        if (start == null || end == null || start.isEmpty() || end.isEmpty()) return 1.0;
        String[] startSplit = start.split(",");
        String[] endSplit = end.split(",");
        if (startSplit.length != endSplit.length) return 1.0;

        double num = 0.00;
        double den = 0.00;
        for (int i = 0; i < startSplit.length; i++) {
            String[] startPair = startSplit[i].split(":");
            String[] endPair = endSplit[i].split(":");
            if (startPair.length < 2 || endPair.length < 2) return 1.0;

            float startValue = Float.parseFloat(startPair[1].trim());
            float endValue = Float.parseFloat(endPair[1].trim());
            num = num + Math.abs(startValue - endValue);
            den = den + Math.abs(startValue + endValue);
        }
        if (den == 0) return 0.0;
        return num / den;
    }
    /**
     * Calculate the average intra-cluster distance for a given point.
     * This method is optimized to run in parallel.
     *
     * @param point The point for which to calculate the average distance.
     * @param cluster The cluster containing the point.
     * @param distanceMeasure The distance metric to use.
     * @return The average intra-cluster distance.
     */
    public static double averageIntraClusterDistance(String point, ArrayList<String> cluster, String distanceMeasure) {
        if (cluster.size() <= 1) {
            return 0.0; // No intra-cluster distance for single-point clusters
        }

        double sumDistance = 0.0;
        for (String otherPoint : cluster) {
            if (!point.equals(otherPoint)) { // Exclude the point itself
                sumDistance += calculateDistance(point, otherPoint, distanceMeasure);
            }
        }
        return sumDistance / (cluster.size() - 1); // Exclude the point itself
    }

    /**
     * Calculate the smallest average inter-cluster distance for a given point.
     * This method is optimized to run in parallel.
     *
     * @param point The point for which to calculate the smallest inter-cluster distance.
     * @param allCluster A map of all clusters.
     * @param ownCluster The cluster containing the point.
     * @param distanceMeasure The distance metric to use.
     * @return The smallest average inter-cluster distance.
     */
    public static double smallestInterClusterDistance(String point, HashMap<String, ArrayList<String>> allCluster,
                                                      ArrayList<String> ownCluster, String distanceMeasure) {
        double smallestAverage = Double.MAX_VALUE;
        for (String key : allCluster.keySet()) {
            ArrayList<String> cluster = allCluster.get(key);
            if (cluster.equals(ownCluster) || cluster.isEmpty()) continue;
            double sumDistance = 0.0;
            for (String otherPoint : cluster) {
                sumDistance += calculateDistance(point, otherPoint, distanceMeasure);
            }
            double averageDistance = sumDistance / cluster.size();
            if (averageDistance < smallestAverage) {
                smallestAverage = averageDistance;
            }
        }
        return smallestAverage;
    }

    //Calculate the Silhouette Coefficient for Each Point
    public static double silhouetteCoefficient(String point, ArrayList<String> ownCluster
            , HashMap<String, ArrayList<String>> allClusters, String distanceMeasure) {
        double a = averageIntraClusterDistance(point, ownCluster, distanceMeasure);
        double b = smallestInterClusterDistance(point, allClusters, ownCluster, distanceMeasure);
        if (Math.max(a, b) == 0) return 0; // Avoid division by zero
        return (b - a) / Math.max(a, b);
    }

    /**
     * Calculate the mean of the Silhouette Coefficients for all points.
     * This method is optimized to run in parallel.
     * @param allCluster The map of all clusters.
     * @param distanceMeasure The distance metric to use.
     * @return The average silhouette coefficient.
     */
    public static double averageSilhouetteCoefficient(HashMap<String, ArrayList<String>> allCluster, String distanceMeasure) {
        if (allCluster == null || allCluster.size() < 2) {
            return 0.0; // Silhouette is not well-defined for less than 2 clusters.
        }

        // Use atomic types for thread-safe accumulation during parallel processing.
        DoubleAdder sumSilhouette = new DoubleAdder();
        AtomicInteger numPoints = new AtomicInteger();

        // Use a parallel stream for faster processing of each cluster.
        allCluster.values().parallelStream().forEach(cluster -> {
            // A nested parallel stream processes each point within the cluster concurrently.
            cluster.parallelStream().forEach(point -> {
                sumSilhouette.add(silhouetteCoefficient(point, cluster, allCluster, distanceMeasure));
                numPoints.incrementAndGet();
            });
        });

        return numPoints.get() == 0 ? 0.0 : sumSilhouette.sum() / numPoints.get();
    }

    // Function to remove "index" and "id" attributes from centroid keys and cluster points
    public static HashMap<String, ArrayList<String>> removeIndexAndId(HashMap<String, ArrayList<String>> kmeanAssign) {
        HashMap<String, ArrayList<String>> cleanedKmeanAssign = new HashMap<>();

        // Iterate over the original kmeanAssign map
        for (Map.Entry<String, ArrayList<String>> entry : kmeanAssign.entrySet()) {
            // Clean the centroid by removing index and id
            String cleanedCentroid = removeIndexAndIdFromString(entry.getKey());

            // Clean the cluster points
            ArrayList<String> cleanedCluster = new ArrayList<>();
            for (String point : entry.getValue()) {
                cleanedCluster.add(removeIndexAndIdFromString(point));
            }

            // Add the cleaned centroid and cleaned cluster to the new map
            cleanedKmeanAssign.put(cleanedCentroid, cleanedCluster);
        }

        return cleanedKmeanAssign;
    }

    /**
     * Helper function to remove attributes like "index" and "id" from a node's string representation.
     * This is made public to be accessible from SimKitProcedures.
     *
     * @param input The raw node string (e.g., "id:1, eigenvector_0:0.5, ...")
     * @return A cleaned node string (e.g., "eigenvector_0:0.5, ...")
     */
    public static String removeIndexAndIdFromString(String input) {
        if (input == null || input.isEmpty()) {
            return "";
        }

        // Use Stream API for a more concise and potentially faster implementation
        return Arrays.stream(input.split(","))
                .map(String::trim)
                .filter(attribute -> {
                    String[] parts = attribute.split(":");
                    if (parts.length == 2) {
                        String attributeName = parts[0].trim();
                        return !attributeName.equalsIgnoreCase("index") && !attributeName.equalsIgnoreCase("id");
                    }
                    return false;
                })
                .collect(Collectors.joining(","));
    }
}
