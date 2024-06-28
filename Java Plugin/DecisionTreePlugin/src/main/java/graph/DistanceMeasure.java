package graph;

import java.util.ArrayList;

import static graph.ReadCsvTestData.convertToDoubleArray;

public class DistanceMeasure {

    private Double [][] DistanceMatrix;

    public DistanceMeasure(Double[][] distanceMatrix) {
        this.DistanceMatrix = distanceMatrix;
    }
    /*
     calculateManhattanDistance
     */
    public static Double[][] calculateManhattanDistance(ArrayList<ArrayList<String>> testData){
        Double[][] distanceMatrix = convertToDoubleArray(testData);
        Double[][] distanceMatrix_new = new Double[distanceMatrix.length][distanceMatrix.length];

        for (int i = 0; i < distanceMatrix.length; i++) {
            for (int j = 0; j < distanceMatrix.length; j++) {
                if (i == j) {
                    distanceMatrix_new[i][j] = 0.0; // Distance to itself is 0
                } else {
                    distanceMatrix_new[i][j] = manhattanDistance(distanceMatrix[i], distanceMatrix[j]);
                }
            }
        }

        return distanceMatrix_new;
    }

    private static Double manhattanDistance(Double[] distanceMatrix, Double[] distanceMatrix1) {
        Double distance = 0.0;
        for (int i = 0; i < distanceMatrix.length; i++) {
            distance += Math.abs(distanceMatrix[i] - distanceMatrix1[i]);
        }
        return distance;
    }
        /*
     calculateCanberraDistance
     */

    public static Double[][] calculateCanberraDistance(ArrayList<ArrayList<String>> testData){
        Double[][] distanceMatrix = convertToDoubleArray(testData);
        Double[][] distanceMatrix_new = new Double[distanceMatrix.length][distanceMatrix.length];
        for (int i = 0; i < distanceMatrix.length; i++) {
            for (int j = 0; j < distanceMatrix.length; j++) {
                if (i == j) {
                    distanceMatrix_new[i][j] = 0.0; // Distance to itself is 0
                } else {
                    distanceMatrix_new[i][j] = canberraDistance(distanceMatrix[i], distanceMatrix[j]);
                }
            }
        }

        return distanceMatrix_new;
    }

    private static Double canberraDistance(Double[] distanceMatrix, Double[] distanceMatrix1) {
        Double distance = 0.0;
        for (int i = 0; i < distanceMatrix.length; i++) {
            distance += Math.abs(distanceMatrix[i] - distanceMatrix1[i]) / (Math.abs(distanceMatrix[i]) + Math.abs(distanceMatrix1[i]));
        }
        return distance;
    }
     /*
     calculateCosineSimilarity
     */

    public static Double[][] calculateCosineSimilarity(ArrayList<ArrayList<String>> testData){
        Double[][] distanceMatrix = convertToDoubleArray(testData);
        Double[][] distanceMatrix_new = new Double[distanceMatrix.length][distanceMatrix.length];
        for (int i = 0; i < distanceMatrix.length; i++) {
            for (int j = 0; j < distanceMatrix.length; j++) {
                if (i == j) {
                    distanceMatrix_new[i][j] = 0.0; // Distance to itself is 0
                } else {
                    distanceMatrix_new[i][j] = cosineSimilarity(distanceMatrix[i], distanceMatrix[j]);
                }
            }
        }

        return distanceMatrix_new;
    }

    private static Double cosineSimilarity(Double[] distanceMatrix, Double[] distanceMatrix1) {
        Double distance = 0.0;
        Double dotProduct = 0.0;
        Double normA = 0.0;
        Double normB = 0.0;
        for (int i = 0; i < distanceMatrix.length; i++) {
            dotProduct += distanceMatrix[i] * distanceMatrix1[i];
            normA += Math.pow(distanceMatrix[i], 2);
            normB += Math.pow(distanceMatrix1[i], 2);
        }
        distance = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        return distance;
    }
     /*
     calculateJaccardCoefficientDistance
     */

    public static Double[][] calculateJaccardCoefficient(ArrayList<ArrayList<String>> testData){
        Double[][] distanceMatrix = convertToDoubleArray(testData);
        Double[][] distanceMatrix_new = new Double[distanceMatrix.length][distanceMatrix.length];
        for (int i = 0; i < distanceMatrix.length; i++) {
            for (int j = 0; j < distanceMatrix.length; j++) {
                if (i == j) {
                    distanceMatrix_new[i][j] = 0.0; // Distance to itself is 0
                } else {
                    distanceMatrix_new[i][j] = jaccardCoefficient(distanceMatrix[i], distanceMatrix[j]);
                }
            }
        }

        return distanceMatrix_new;
    }

    private static Double jaccardCoefficient(Double[] distanceMatrix, Double[] distanceMatrix1) {
        Double distance = 0.0;
        Double dotProduct = 0.0;
        Double normA = 0.0;
        Double normB = 0.0;
        for (int i = 0; i < distanceMatrix.length; i++) {
            dotProduct += distanceMatrix[i] * distanceMatrix1[i];
            normA += Math.pow(distanceMatrix[i], 2);
            normB += Math.pow(distanceMatrix1[i], 2);
        }
        distance = dotProduct / (normA + normB - dotProduct);
        return distance;
    }
        /*
        calculateBrayCurtisDistance
        */
    public static Double[][] calculateBrayCurtisDistance(ArrayList<ArrayList<String>> testData){
        Double[][] distanceMatrix = convertToDoubleArray(testData);
        Double[][] distanceMatrix_new = new Double[distanceMatrix.length][distanceMatrix.length];
        for (int i = 0; i < distanceMatrix.length; i++) {
            for (int j = 0; j < distanceMatrix.length; j++) {
                if (i == j) {
                    distanceMatrix_new[i][j] = 0.0; // Distance to itself is 0
                } else {
                    distanceMatrix_new[i][j] = brayCurtisDistance(distanceMatrix[i], distanceMatrix[j]);
                }
            }
        }

        return distanceMatrix_new;
    }

    private static Double brayCurtisDistance(Double[] distanceMatrix, Double[] distanceMatrix1) {
        Double distance = 0.0;
        Double dotProduct = 0.0;
        Double normA = 0.0;
        Double normB = 0.0;
        for (int i = 0; i < distanceMatrix.length; i++) {
            dotProduct += Math.abs(distanceMatrix[i] - distanceMatrix1[i]);
            normA += Math.abs(distanceMatrix[i]);
            normB += Math.abs(distanceMatrix1[i]);
        }
        distance = dotProduct / (normA + normB);
        return distance;
    }
}
