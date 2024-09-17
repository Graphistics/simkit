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

import definition.EdgeList2;

public class MatrixCalculation {
	
    public static class MatrixResult {
        private static RealMatrix adjacency_matrix;
        private static RealMatrix degree_matrix;
        private static RealMatrix laplacian_matrix;

        public MatrixResult(RealMatrix adjacency_matrix, RealMatrix degree_matrix, RealMatrix laplacian_matrix) {
            MatrixResult.adjacency_matrix = adjacency_matrix;
            MatrixResult.degree_matrix = degree_matrix;
            MatrixResult.laplacian_matrix = laplacian_matrix;
        }

        public RealMatrix getAdjacencyMatrix() {
            return adjacency_matrix;
        }

        public static RealMatrix getDegreeMatrix() {
            return degree_matrix;
        }

        public static RealMatrix getLaplacianMatrix() {
            return laplacian_matrix;
        }
    }
    
    /**
     * Converts an edge list into an adjacency matrix.
     *
     * @param edge_list The edge list to convert.
     * @return 2D array representing the adjacency matrix.
     */
    public static RealMatrix convertToAdjacencyMatrix(ArrayList<EdgeList2> edge_list) {
        Set<String> unique_indices = new HashSet<>();
        for (EdgeList2 edge : edge_list) {
            unique_indices.add(edge.getSource());
            unique_indices.add(edge.getTarget());
        }

        int dimension = unique_indices.size();
        List<String> sorted_indices = new ArrayList<>(unique_indices);
        Collections.sort(sorted_indices);

        Map<String, Integer> index_mapping = new HashMap<>();
        for (int i = 0; i < dimension; i++) {
            index_mapping.put(sorted_indices.get(i), i);
        }

        double[][] adjacency_matrix_data = new double[dimension][dimension];

        for (EdgeList2 edge : edge_list) {
            int i = index_mapping.get(edge.getSource());
            int j = index_mapping.get(edge.getTarget());
            double weight = edge.getWeight();
            adjacency_matrix_data[i][j] = weight;
            adjacency_matrix_data[j][i] = weight;
        }
        
        RealMatrix adjacency_matrix = new BlockRealMatrix(adjacency_matrix_data);
        return adjacency_matrix;
    }
   
    /**
     * Calculates the Laplacian matrix based on the provided degree matrix, adjacency matrix and desired type of Laplacian matrix.
     *
     * @param degree_matrix    The degree matrix of the graph.
     * @param adjacency_matrix The adjacency matrix of the graph.
     * @param algorithm       The algorithm to use for Laplacian matrix calculation ("sym" or "rw").
     * @return The Laplacian matrix.
     */
    public static RealMatrix calculateLaplacianMatrix(RealMatrix degree_matrix, RealMatrix adjacency_matrix, String algorithm) {
        try {
            RealMatrix laplacian_matrix;

            switch (algorithm) {
                case "sym":
                    laplacian_matrix = calculateSymmetricLaplacianMatrix(degree_matrix, adjacency_matrix);
                    break;
                case "rw":
                    laplacian_matrix = calculateRandomWalkLaplacianMatrix(degree_matrix, adjacency_matrix);
                    break;
                default:
                    throw new IllegalArgumentException("Invalid Laplacian algorithm choice.");
            }

            return laplacian_matrix;
        } catch (Exception e) {
            throw new RuntimeException("Error calculating Laplacian matrix: " + e.getMessage());
        }
    }

    /**
     * Calculates the degree matrix from an adjacency matrix.
     *
     * @param adj_mat The adjacency matrix.
     * @return The degree matrix.
     */
    public static RealMatrix calculateDegreeMatrix(RealMatrix adjacency_matrix) {
        int dimension = adjacency_matrix.getColumnDimension();

        double[] columnSum = new double[dimension];
        for (int col = 0; col < dimension; col++) {
            columnSum[col] = adjacency_matrix.getColumnVector(col).getL1Norm();
        }

        return MatrixUtils.createRealDiagonalMatrix(columnSum);
    }

    /**
     * Calculates the symmetric Laplacian matrix based on the degree matrix and adjacency matrix.
     *
     * @param degree_matrix    The degree matrix of the graph.
     * @param adjacency_matrix The adjacency matrix of the graph.
     * @return The symmetric Laplacian matrix.
     */
    public static RealMatrix calculateSymmetricLaplacianMatrix(RealMatrix degree_matrix, RealMatrix adjacency_matrix) {
        int dimension = degree_matrix.getColumnDimension();
        RealMatrix dHalf = MatrixUtils.createRealMatrix(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            double dHalfValue = 1.0 / Math.sqrt(degree_matrix.getEntry(i, i));
            dHalf.setEntry(i, i, dHalfValue);
        }

        RealMatrix laplacian_matrix_normalized = dHalf.multiply(adjacency_matrix).multiply(dHalf);
        return laplacian_matrix_normalized;
    }

    /**
     * Calculates the random walk Laplacian matrix based on the degree matrix and adjacency matrix.
     *
     * @param degree_matrix    The degree matrix of the graph.
     * @param adjacency_matrix The adjacency matrix of the graph.
     * @return The random walk Laplacian matrix.
     */
    public static RealMatrix calculateRandomWalkLaplacianMatrix(RealMatrix degree_matrix, RealMatrix adjacency_matrix) {
        RealMatrix inverse_degree_matrix = MatrixUtils.inverse(degree_matrix);
        RealMatrix random_walk_laplacian_matrix = inverse_degree_matrix.multiply(adjacency_matrix);

        return random_walk_laplacian_matrix;
    }

    
}
