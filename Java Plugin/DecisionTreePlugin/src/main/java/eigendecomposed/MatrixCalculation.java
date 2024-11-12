package eigendecomposed;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Precision;

import definition.EdgeList2;
import definition.NodeList2;

/**
 * Utility class for matrix calculations related to graph Laplacians.
 */
public class MatrixCalculation {

    /**
     * Inner class to hold the results of matrix computations.
     */
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
     * @return RealMatrix representing the adjacency matrix.
     */
    public static RealMatrix convertToAdjacencyMatrix(ArrayList<EdgeList2> edge_list) {
        // Collect unique node indices from the edge list
        Set<String> unique_indices = new HashSet<>();
        for (EdgeList2 edge : edge_list) {
            unique_indices.add(edge.getSource());
            unique_indices.add(edge.getTarget());
        }

        // Determine the dimension of the adjacency matrix
        int dimension = unique_indices.size();

        // Initialize the adjacency matrix data array
        double[][] adjacency_matrix_data = new double[dimension][dimension];

        // Populate the adjacency matrix with edge weights
        for (EdgeList2 edge : edge_list) {
            int i = Integer.parseInt(edge.getSource());
            int j = Integer.parseInt(edge.getTarget());

            double weight = edge.getWeight();

            adjacency_matrix_data[i][j] = weight;
            adjacency_matrix_data[j][i] = weight; // For undirected graphs
        }

        // Create the adjacency matrix from the data array
        RealMatrix adjacency_matrix = new BlockRealMatrix(adjacency_matrix_data);
        return adjacency_matrix;
    }

    /**
     * Converts a list of nodes into an X matrix containing eigenvector properties.
     *
     * @param node_list The list of nodes.
     * @return RealMatrix representing the X matrix.
     */
    public static RealMatrix convertToXMatrix(ArrayList<NodeList2> node_list) {
        int dimension = node_list.size();

        // Determine the actual number of eigenvector properties
        int vector_dimension = (int) node_list.get(0).getProperties().keySet().stream()
                .filter(key -> key.startsWith("eigenvector_"))
                .count();

        // Initialize the X matrix data array
        double[][] x_matrix_data = new double[dimension][vector_dimension];

        // Populate the X matrix with eigenvector values
        for (int i = 0; i < dimension; i++) {
            Map<String, Object> properties = node_list.get(i).getProperties();
            for (int j = 0; j < vector_dimension; j++) {
                String key = "eigenvector_" + j;
                if (properties.containsKey(key)) {
                    x_matrix_data[i][j] = (double) properties.get(key);
                }
            }
        }

        // Create the X matrix from the data array
        return new BlockRealMatrix(x_matrix_data);
    }

    /**
     * Calculates the Laplacian matrix based on the provided degree matrix, adjacency matrix, and desired type of Laplacian matrix.
     *
     * @param degree_matrix    The degree matrix of the graph.
     * @param adjacency_matrix The adjacency matrix of the graph.
     * @param algorithm        The algorithm to use for Laplacian matrix calculation ("sym", "rw", or "ad").
     * @return The Laplacian matrix.
     */
    public static RealMatrix calculateLaplacianMatrix(RealMatrix degree_matrix, RealMatrix adjacency_matrix, String algorithm) {
        try {
            RealMatrix laplacian_matrix;

            // Choose the Laplacian matrix calculation method based on the algorithm parameter
            switch (algorithm) {
                case "sym":
                    laplacian_matrix = calculateSymmetricLaplacianMatrix(degree_matrix, adjacency_matrix);
                    break;
                case "rw":
                    laplacian_matrix = calculateRandomWalkLaplacianMatrix(degree_matrix, adjacency_matrix);
                    break;
                case "ad":
                    laplacian_matrix = calculateAdaptiveLaplacianMatrix(degree_matrix, adjacency_matrix);
                    break;
                default:
                    laplacian_matrix = calculateSymmetricLaplacianMatrix(degree_matrix, adjacency_matrix);
            }

            return laplacian_matrix;
        } catch (Exception e) {
            throw new RuntimeException("Error calculating Laplacian matrix: " + e.getMessage());
        }
    }

    /**
     * Calculates the degree matrix from an adjacency matrix.
     *
     * @param adjacency_matrix The adjacency matrix.
     * @return The degree matrix.
     */
    public static RealMatrix calculateDegreeMatrix(RealMatrix adjacency_matrix) {
        int dimension = adjacency_matrix.getColumnDimension();

        // Initialize the degree vector
        double[] column_sum = new double[dimension];
        for (int col = 0; col < dimension; col++) {
            // Sum the weights in each column (or row, since the matrix is symmetric)
            column_sum[col] = adjacency_matrix.getColumnVector(col).getL1Norm();
        }

        // Create a diagonal matrix from the degree vector
        return MatrixUtils.createRealDiagonalMatrix(column_sum);
    }

    /**
     * Calculates the symmetric Laplacian matrix based on the degree matrix and adjacency matrix.
     *
     * @param degree_matrix    The degree matrix of the graph.
     * @param adjacency_matrix The adjacency matrix of the graph.
     * @return The symmetric normalized Laplacian matrix.
     */
    public static RealMatrix calculateSymmetricLaplacianMatrix(RealMatrix degree_matrix, RealMatrix adjacency_matrix) {
        int dimension = degree_matrix.getColumnDimension();
        // Initialize D^(-1/2)
        RealMatrix d_half_inverse = MatrixUtils.createRealMatrix(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            double degree = degree_matrix.getEntry(i, i);
            double d_half_value = degree > 0 ? 1.0 / Math.sqrt(degree) : 0;  // Set to zero for isolated nodes
            d_half_inverse.setEntry(i, i, d_half_value);
        }

        // Calculate L = D^(-1/2) * A * D^(-1/2)
        RealMatrix laplacian_matrix_normalized = d_half_inverse.multiply(adjacency_matrix).multiply(d_half_inverse);

        return laplacian_matrix_normalized;
    }

    /**
     * Calculates the random walk Laplacian matrix based on the degree matrix and adjacency matrix.
     *
     * @param degree_matrix    The degree matrix of the graph.
     * @param adjacency_matrix The adjacency matrix of the graph.
     * @return The random walk normalized Laplacian matrix.
     */
    public static RealMatrix calculateRandomWalkLaplacianMatrix(RealMatrix degree_matrix, RealMatrix adjacency_matrix) {
        int dimension = degree_matrix.getRowDimension();
        // Initialize D^(-1)
        RealMatrix d_inverse = MatrixUtils.createRealMatrix(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            double degree = degree_matrix.getEntry(i, i);
            double d_inverse_value = degree > 0 ? 1.0 / degree : 0;  // Set inverse to zero for isolated nodes
            d_inverse.setEntry(i, i, d_inverse_value);
        }

        // Calculate L = D^(-1) * A
        RealMatrix random_walk_laplacian_matrix = d_inverse.multiply(adjacency_matrix);

        return random_walk_laplacian_matrix;
    }

    /**
     * Calculates the adaptive Laplacian matrix based on the degree matrix and adjacency matrix.
     *
     * @param degree_matrix    The degree matrix of the graph.
     * @param adjacency_matrix The adjacency matrix of the graph.
     * @return The adaptive Laplacian matrix.
     */
    public static RealMatrix calculateAdaptiveLaplacianMatrix(RealMatrix degree_matrix, RealMatrix adjacency_matrix) {
        int dimension = degree_matrix.getRowDimension();
        double[] degree_vector = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            degree_vector[i] = degree_matrix.getEntry(i, i);
        }

        double[] D_local = new double[dimension];

        // Compute local degrees
        for (int i = 0; i < dimension; i++) {
            double sum_neighbors = 0.0;
            int neighbor_count = 0;

            for (int j = 0; j < adjacency_matrix.getColumnDimension(); j++) {
                if (adjacency_matrix.getEntry(i, j) > 0) {
                    sum_neighbors += degree_vector[j];
                    neighbor_count++;
                }
            }

            D_local[i] = neighbor_count > 0 ? sum_neighbors / degree_vector[i] : 0;  // Zero for isolated nodes
        }

        // Compute D_local^(-1/2)
        double[] D_local_inv_sqrt = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            D_local_inv_sqrt[i] = D_local[i] > 0 ? 1.0 / Math.sqrt(D_local[i]) : 0;  // Zero for isolated nodes
        }

        // Create diagonal matrix D_local^(-1/2)
        RealMatrix D_local_inv_sqrt_matrix = MatrixUtils.createRealDiagonalMatrix(D_local_inv_sqrt);

        // Calculate L = D_local^(-1/2) * A * D_local^(-1/2)
        return D_local_inv_sqrt_matrix.multiply(adjacency_matrix).multiply(D_local_inv_sqrt_matrix);
    }
}
