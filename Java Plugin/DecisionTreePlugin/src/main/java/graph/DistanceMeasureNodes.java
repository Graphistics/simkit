package graph;

import definition.NodeList2;

import java.util.*;
import java.util.stream.IntStream;
import java.util.Map;

import static graph.ReadCsvTestData.convertToDoubleArray;

public class DistanceMeasureNodes {

    public static Double[][] euclideanDistance(List<NodeList2> nodeList, List<String> removeList) {
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        // Compute only upper triangle and mirror the values
        IntStream.range(0, size).parallel().forEach(i -> {
            for (int j = i; j < size; j++) { // Start from i (avoids redundant work)
                if (i == j) {
                    distanceMatrix[i][j] = 0.0; // Distance to itself is 0
                } else {
                    double distance = calculateEuclideanDistance(nodeList.get(i), nodeList.get(j), removeList);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance; // Mirror value
                }
            }
        });

        return distanceMatrix;
    }

    private static double calculateEuclideanDistance(NodeList2 node1, NodeList2 node2, List<String> removeList) {
    Map<String, Object> properties1 = node1.getProperties();
    Map<String, Object> properties2 = node2.getProperties();

    // Convert removeList to HashSet for O(1) lookups
    Set<String> removeSet = new HashSet<>(removeList);

    double sum = 0.0;

    // Iterate over key-value pairs directly (avoid double hashmap lookups)
    for (Map.Entry<String, Object> entry : properties1.entrySet()) {
        String propertyKey = entry.getKey();

        // Ignore properties in removeSet (O(1) lookup)
        if (removeSet.contains(propertyKey)) {
            continue;
        }

        Object value1 = entry.getValue();
        Object value2 = properties2.get(propertyKey); // Only 1 hashmap lookup

        if (value1 instanceof Number && value2 instanceof Number) {
            double diff = ((Number) value1).doubleValue() - ((Number) value2).doubleValue();
            sum += diff * diff; // Avoid unnecessary Math.pow()
        }
    }

    return Math.sqrt(sum);
}


public static Double[][] manhattanDistance(List<NodeList2> nodeList, List<String> removeList) {
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        // Convert removeList to HashSet for O(1) lookups
        Set<String> removeSet = new HashSet<>(removeList);

        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {  // Only compute upper triangle
                if (i == j) {
                    distanceMatrix[i][j] = 0.0;  // Distance to itself is always 0
                } else {
                    double distance = calculateManhattanDistance(nodeList.get(i), nodeList.get(j), removeSet);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance;  // Symmetric copy
                }
            }
        }
        return distanceMatrix;
    }

    private static double calculateManhattanDistance(NodeList2 node1, NodeList2 node2, Set<String> removeSet) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

        double sum = 0.0;

        // Iterate over entrySet for direct key-value access
        for (Map.Entry<String, Object> entry : properties1.entrySet()) {
            String propertyKey = entry.getKey();

            // Skip properties that should be removed
            if (removeSet.contains(propertyKey)) {
                continue;
            }

            Object value1 = entry.getValue();
            Object value2 = properties2.get(propertyKey);  // Single hashmap lookup instead of multiple

            if (value1 instanceof Number && value2 instanceof Number) {
                sum += Math.abs(((Number) value1).doubleValue() - ((Number) value2).doubleValue());
            }
        }

        return sum;
    }
    public static Double[][] canberraDistance(List<NodeList2> nodeList, List<String> removeList) {
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        // Convert removeList to HashSet for O(1) lookups
        Set<String> removeSet = new HashSet<>(removeList);

        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {  // Compute only upper triangle
                if (i == j) {
                    distanceMatrix[i][j] = 0.0;  // Distance to itself is always 0
                } else {
                    double distance = calculateCanberraDistance(nodeList.get(i), nodeList.get(j), removeSet);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance;  // Symmetric copy
                }
            }
        }
        return distanceMatrix;
    }

    private static double calculateCanberraDistance(NodeList2 node1, NodeList2 node2, Set<String> removeSet) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

        double sum = 0.0;

        // Iterate over entrySet for direct key-value access
        for (Map.Entry<String, Object> entry : properties1.entrySet()) {
            String propertyKey = entry.getKey();

            // Skip properties that should be removed
            if (removeSet.contains(propertyKey)) {
                continue;
            }

            Object value1 = entry.getValue();
            Object value2 = properties2.get(propertyKey);  // Single hashmap lookup instead of multiple

            if (value1 instanceof Number && value2 instanceof Number) {
                double num1 = ((Number) value1).doubleValue();
                double num2 = ((Number) value2).doubleValue();
                double denominator = Math.abs(num1) + Math.abs(num2);

                if (denominator != 0) { // Avoid division by zero
                    sum += Math.abs(num1 - num2) / denominator;
                }
            }
        }

        return sum;
    }
    private static double calculateCosineSimilarity(NodeList2 node1, NodeList2 node2, Set<String> removeSet) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

        double sum = 0.0, sum1 = 0.0, sum2 = 0.0;

        for (Map.Entry<String, Object> entry : properties1.entrySet()) {
            String propertyKey = entry.getKey();

            // Ignore properties in removeSet
            if (removeSet.contains(propertyKey)) continue;

            Object value1 = entry.getValue();
            Object value2 = properties2.get(propertyKey); // Single lookup

            if (value1 instanceof Number && value2 instanceof Number) {
                double v1 = ((Number) value1).doubleValue();
                double v2 = ((Number) value2).doubleValue();

                // Use Math.fma for fused multiply-add to improve precision and speed
                sum = Math.fma(v1, v2, sum);
                sum1 = Math.fma(v1, v1, sum1);
                sum2 = Math.fma(v2, v2, sum2);
            }
        }

        return sum / (Math.sqrt(sum1) * Math.sqrt(sum2));
    }

    public static Double[][] cosineSimilarity(List<NodeList2> nodeList, List<String> removeList) {
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        // Convert removeList to HashSet for faster lookups
        Set<String> removeSet = new HashSet<>(removeList);

        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {  // Only compute upper triangle
                if (i == j) {
                    distanceMatrix[i][j] = 0.0;
                } else {
                    double distance = calculateCosineSimilarity(nodeList.get(i), nodeList.get(j), removeSet);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance;  // Copy for symmetry
                }
            }
        }
        return distanceMatrix;
    }
    public static Double[][] jaccardCoefficient(List<NodeList2> nodeList, List<String> removeList) {
        int size = nodeList.size();
        Double[][] similarityMatrix = new Double[size][size];

        // Convert removeList to HashSet for O(1) lookups
        Set<String> removeSet = new HashSet<>(removeList);

        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {  // Compute only upper triangle
                if (i == j) {
                    similarityMatrix[i][j] = 1.0;  // Jaccard similarity to itself is always 1
                } else {
                    double similarity = calculateJaccardCoefficient(nodeList.get(i), nodeList.get(j), removeSet);
                    similarityMatrix[i][j] = similarity;
                    similarityMatrix[j][i] = similarity;  // Symmetric copy
                }
            }
        }
        return similarityMatrix;
    }

    private static double calculateJaccardCoefficient(NodeList2 node1, NodeList2 node2, Set<String> removeSet) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

        double intersectionSum = 0.0;
        double unionSum = 0.0;

        // Iterate over entrySet for direct key-value access
        for (Map.Entry<String, Object> entry : properties1.entrySet()) {
            String propertyKey = entry.getKey();

            // Skip properties that should be removed
            if (removeSet.contains(propertyKey)) {
                continue;
            }

            Object value1 = entry.getValue();
            Object value2 = properties2.get(propertyKey);  // Single hashmap lookup

            if (value1 instanceof Number && value2 instanceof Number) {
                double num1 = ((Number) value1).doubleValue();
                double num2 = ((Number) value2).doubleValue();

                intersectionSum += Math.min(num1, num2);
                unionSum += Math.max(num1, num2);
            }
        }

        // Prevent division by zero
        return (unionSum != 0) ? intersectionSum / unionSum : 0.0;
    }
    public static Double[][] brayCurtisDistance(List<NodeList2> nodeList, List<String> removeList) {
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        // Convert removeList to HashSet for O(1) lookups
        Set<String> removeSet = new HashSet<>(removeList);

        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {  // Compute only upper triangle
                if (i == j) {
                    distanceMatrix[i][j] = 0.0;  // Distance to itself is always 0
                } else {
                    double distance = calculateBrayCurtisDistance(nodeList.get(i), nodeList.get(j), removeSet);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance;  // Symmetric copy
                }
            }
        }
        return distanceMatrix;
    }

    private static double calculateBrayCurtisDistance(NodeList2 node1, NodeList2 node2, Set<String> removeSet) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

        double numerator = 0.0;
        double denominator = 0.0;

        // Iterate over entrySet for direct key-value access
        for (Map.Entry<String, Object> entry : properties1.entrySet()) {
            String propertyKey = entry.getKey();

            // Skip properties that should be removed
            if (removeSet.contains(propertyKey)) {
                continue;
            }

            Object value1 = entry.getValue();
            Object value2 = properties2.get(propertyKey);  // Single hashmap lookup instead of multiple

            if (value1 instanceof Number && value2 instanceof Number) {
                double num1 = ((Number) value1).doubleValue();
                double num2 = ((Number) value2).doubleValue();

                numerator += Math.abs(num1 - num2);
                denominator += Math.abs(num1 + num2);
            }
        }

        // Prevent division by zero
        return (denominator != 0) ? numerator / denominator : 0.0;
    }
    
    
    
    
    


}
