package graph;

import definition.NodeList2;

import java.util.List;
import java.util.Map;

public class DistanceMeasureNodes {

    public static Double[][] euclideanDistance(List<NodeList2> nodeList,List<String> removeList) {

        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double distance = calculateEuclideanDistance(nodeList.get(i), nodeList.get(j),removeList);
                distanceMatrix[i][j] = distance;
            }
        }
        return distanceMatrix;
    }

    private static double calculateEuclideanDistance(NodeList2 node1, NodeList2 node2,List<String> removeList) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();


//        removeList.forEach(properties1.keySet()::remove);
//        removeList.forEach(properties2.keySet()::remove);

        double sum = 0.0;

        for (String propertyKey : properties1.keySet()) {
            // ignore the properties in the removeList
            if (removeList.contains(propertyKey)){
                continue;
            }
            Object value1 = properties1.get(propertyKey);
            Object value2 = properties2.get(propertyKey);

            if (value1 instanceof Number && value2 instanceof Number) {
                double diff = ((Number) value1).doubleValue() - ((Number) value2).doubleValue();
                sum += Math.pow(diff, 2);
            }
        }
        return Math.sqrt(sum);
    }

    public static Double[][] manhattanDistance(List<NodeList2> nodeList,List<String> removeList){
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                    double distance = calculateManhattanDistance(nodeList.get(i), nodeList.get(j),removeList);
                    distanceMatrix[i][j] = distance;
            }
        }
        return distanceMatrix;
    }
    private static double calculateManhattanDistance(NodeList2 node1, NodeList2 node2,List<String> removeList) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

//        removeList.forEach(properties1.keySet()::remove);
//        removeList.forEach(properties2.keySet()::remove);

        double sum = 0.0;



        for (String propertyKey : properties1.keySet()) {
            if (removeList.contains(propertyKey)){
                continue;
            }
            Object value1 = properties1.get(propertyKey);
            Object value2 = properties2.get(propertyKey);

            if (value1 instanceof Number && value2 instanceof Number) {
                double diff = ((Number) value1).doubleValue() - ((Number) value2).doubleValue();
                sum += Math.abs(diff);
            }
        }
        return sum;
    }
    public static Double[][] canberraDistance(List<NodeList2> nodeList,List<String> removeList){
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                    double distance = calculateCanberraDistance(nodeList.get(i), nodeList.get(j),removeList);
                    distanceMatrix[i][j] = distance;
            }
        }
        return distanceMatrix;
    }
    private static double calculateCanberraDistance(NodeList2 node1, NodeList2 node2,List<String> removeList) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

//        removeList.forEach(properties1.keySet()::remove);
//        removeList.forEach(properties2.keySet()::remove);

        double sum = 0.0;

        for (String propertyKey : properties1.keySet()) {
            if (removeList.contains(propertyKey)){
                continue;
            }
            Object value1 = properties1.get(propertyKey);
            Object value2 = properties2.get(propertyKey);

            if (value1 instanceof Number && value2 instanceof Number) {
                double diff = ((Number) value1).doubleValue() - ((Number) value2).doubleValue();
                sum += Math.abs(diff) / (Math.abs(((Number) value1).doubleValue()) + Math.abs(((Number) value2).doubleValue()));
            }
        }
        return sum;
    }
    public static Double[][] cosineSimilarity(List<NodeList2> nodeList,List<String> removeList){
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {

                if (i == j) {
                    distanceMatrix[i][j] = 0.00;
                } else {

                    double distance = calculateCosineSimilarity(nodeList.get(i), nodeList.get(j), removeList);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance;
                }
            }
        }
        return distanceMatrix;
    }
    private static double calculateCosineSimilarity(NodeList2 node1, NodeList2 node2,List<String> removeList) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

//        removeList.forEach(properties1.keySet()::remove);
//        removeList.forEach(properties2.keySet()::remove);

        double sum = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;

        for (String propertyKey : properties1.keySet()) {
            if (removeList.contains(propertyKey)){
                continue;
            }
            Object value1 = properties1.get(propertyKey);
            Object value2 = properties2.get(propertyKey);

            if (value1 instanceof Number && value2 instanceof Number) {
                sum += ((Number) value1).doubleValue() * ((Number) value2).doubleValue();
                sum1 += Math.pow(((Number) value1).doubleValue(), 2);
                sum2 += Math.pow(((Number) value2).doubleValue(), 2);
            }
        }
        return sum / (Math.sqrt(sum1) * Math.sqrt(sum2));
    }
    public static Double[][] jaccardCoefficient(List<NodeList2> nodeList,List<String> removeList){
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {

                    double distance = calculateJaccardCoefficient(nodeList.get(i), nodeList.get(j),removeList);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance;
            }
        }
        return distanceMatrix;
    }
    private static double calculateJaccardCoefficient(NodeList2 node1, NodeList2 node2,List<String> removeList) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

//        removeList.forEach(properties1.keySet()::remove);
//        removeList.forEach(properties2.keySet()::remove);

        double sum = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;

        for (String propertyKey : properties1.keySet()) {
            if (removeList.contains(propertyKey)){
                continue;
            }
            Object value1 = properties1.get(propertyKey);
            Object value2 = properties2.get(propertyKey);

            if (value1 instanceof Number && value2 instanceof Number) {
                sum += Math.min(((Number) value1).doubleValue(), ((Number) value2).doubleValue());
                sum1 += ((Number) value1).doubleValue();
                sum2 += ((Number) value2).doubleValue();
            }
        }
        return sum / (sum1 + sum2 - sum);
    }
    public static Double[][] brayCurtisDistance(List<NodeList2> nodeList,List<String> removeList){
        int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {

                    double distance = calculateBrayCurtisDistance(nodeList.get(i), nodeList.get(j),removeList);
                    distanceMatrix[i][j] = distance;
                    distanceMatrix[j][i] = distance;
            }
        }
        return distanceMatrix;
        
    }
    private static double calculateBrayCurtisDistance(NodeList2 node1, NodeList2 node2,List<String> removeList) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

//        removeList.forEach(properties1.keySet()::remove);
//        removeList.forEach(properties2.keySet()::remove);

        double sum1 = 0.0;
        double sum2 = 0.0;

        for (String propertyKey : properties1.keySet()) {
            if (removeList.contains(propertyKey)){
                continue;
            }
            Object value1 = properties1.get(propertyKey);
            Object value2 = properties2.get(propertyKey);

            if (value1 instanceof Number && value2 instanceof Number) {
                sum1 += Math.abs(((Number) value1).doubleValue() - ((Number) value2).doubleValue());
                sum2 += Math.abs(((Number) value1).doubleValue() + ((Number) value2).doubleValue());
            }
        }
        return sum1 / sum2;

    }
    
    
    
    
    


}
