package eigendecomposed;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math4.legacy.linear.BlockRealMatrix;
import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;

import graph.EdgeList2;

public class MatrixCalculation {
	
    public static class MatrixResult {
        private static RealMatrix adjacencyMatrix;
        private static RealMatrix degreeMatrix;
        private static RealMatrix laplacianMatrix;

        public MatrixResult(RealMatrix adjacencyMatrix, RealMatrix degreeMatrix, RealMatrix laplacianMatrix) {
            MatrixResult.adjacencyMatrix = adjacencyMatrix;
            MatrixResult.degreeMatrix = degreeMatrix;
            MatrixResult.laplacianMatrix = laplacianMatrix;
        }

        public RealMatrix getAdjacencyMatrix() {
            return adjacencyMatrix;
        }

        public static RealMatrix getDegreeMatrix() {
            return degreeMatrix;
        }

        public static RealMatrix getLaplacianMatrix() {
            return laplacianMatrix;
        }
    }
    
    public static double[][] convertToAdjacencyMatrix2(ArrayList<EdgeList2> edgeList) {
        Set<String> uniqueIndices = new HashSet<>();
        for (EdgeList2 edge : edgeList) {
            uniqueIndices.add(edge.getSource());
            uniqueIndices.add(edge.getTarget());
        }

        int dimension = uniqueIndices.size();
        List<String> sortedIndices = new ArrayList<>(uniqueIndices);
        Collections.sort(sortedIndices);

        Map<String, Integer> indexMapping = new HashMap<>();
        for (int i = 0; i < dimension; i++) {
            indexMapping.put(sortedIndices.get(i), i);
        }

        double[][] adjacencyMatrixData = new double[dimension][dimension];

        for (EdgeList2 edge : edgeList) {
            int i = indexMapping.get(edge.getSource());
            int j = indexMapping.get(edge.getTarget());
            double weight = edge.getWeight();
            adjacencyMatrixData[i][j] = weight;
            adjacencyMatrixData[j][i] = weight;
        }

        return adjacencyMatrixData;
    }

    
    
//    public static double[][] convertToAdjacencyMatrix(ArrayList<EdgeList2> edgeList) {
//        // Find the maximum index value to determine the dimension of the adjacency matrix
//        int maxIndex = 0;
//        for (EdgeList2 edge : edgeList) {
//            int sourceIndex = extractIndexFromId(edge.getSource());
//            int targetIndex = extractIndexFromId(edge.getTarget());
//
//            maxIndex = Math.max(maxIndex, sourceIndex);
//            maxIndex = Math.max(maxIndex, targetIndex);
//        }
//
//        // Dimension of the adjacency matrix
//        int dimension = maxIndex + 1;
//        double[][] adjacencyMatrixData = new double[dimension][dimension];
//
//        // Initialize adjacency matrix with appropriate weights
//        for (EdgeList2 edge : edgeList) {
//            int i = extractIndexFromId(edge.getSource());
//            int j = extractIndexFromId(edge.getTarget());
//            double weight = edge.getWeight();
//            adjacencyMatrixData[i][j] = weight;
//            adjacencyMatrixData[j][i] = weight;
//        }
//
//        return adjacencyMatrixData;
//    }
//
//    private static int extractIndexFromId(String id) {
//        return Integer.parseInt(id.replaceAll("\\D", ""));
//    }
   
    public static RealMatrix calculateLaplacianMatrix(RealMatrix degreeMatrix, RealMatrix adjacencyMatrix, String algorithm) {
        try {
            RealMatrix laplacianMatrix;

            switch (algorithm) {
                case "SYMMETRIC":
                    laplacianMatrix = calculateSymmetricLaplacianMatrix(degreeMatrix, adjacencyMatrix);
                    break;
                case "RANDOM_WALK":
                    laplacianMatrix = calculateRandomWalkLaplacianMatrix(degreeMatrix, adjacencyMatrix);
                    break;
                default:
                    throw new IllegalArgumentException("Invalid Laplacian algorithm choice.");
            }

            return laplacianMatrix;
        } catch (Exception e) {
            throw new RuntimeException("Error calculating Laplacian matrix: " + e.getMessage());
        }
    }

    
    public static RealMatrix calculateDegreeMatrix(double[][] adj_mat) {
        RealMatrix adjacencyMatrix = new BlockRealMatrix(adj_mat);
        int dimension = adjacencyMatrix.getColumnDimension();

        double[] columnSum = new double[dimension];
        for (int col = 0; col < dimension; col++) {
            columnSum[col] = adjacencyMatrix.getColumnVector(col).getL1Norm();
        }

        return MatrixUtils.createRealDiagonalMatrix(columnSum);
    }

    
    public static RealMatrix calculateSymmetricLaplacianMatrix(RealMatrix degreeMatrix, RealMatrix adjacencyMatrix) {
        int dimension = degreeMatrix.getColumnDimension();
        RealMatrix dHalf = MatrixUtils.createRealMatrix(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            double dHalfValue = 1.0 / Math.sqrt(degreeMatrix.getEntry(i, i));
            dHalf.setEntry(i, i, dHalfValue);
        }

        RealMatrix laplacianMatrixNormalized = dHalf.multiply(adjacencyMatrix).multiply(dHalf);
        return laplacianMatrixNormalized;
    }

    public static RealMatrix calculateRandomWalkLaplacianMatrix(RealMatrix degreeMatrix, RealMatrix adjacencyMatrix) {
        RealMatrix inverseDegreeMatrix = MatrixUtils.inverse(degreeMatrix);
        RealMatrix randomWalkLaplacianMatrix = inverseDegreeMatrix.multiply(adjacencyMatrix);

        return randomWalkLaplacianMatrix;
    }

    
}
