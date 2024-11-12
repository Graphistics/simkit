package eigendecomposed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.MatrixFeatures_DDRM;
import org.ejml.dense.row.decomposition.eig.SwitchingEigenDecomposition_DDRM;
import org.ejml.dense.row.decomposition.qr.QRDecompositionHouseholder_DDRM;
import org.ejml.interfaces.decomposition.EigenDecomposition_F64;
import org.ejml.interfaces.decomposition.QRDecomposition;
import org.ejml.simple.SimpleMatrix;

import definition.EdgeList2;
import definition.NodeList2;

/**
 * Utility class for eigenvalue and eigenvector calculations related to graph Laplacians.
 */
public class EigenCalculation {

    /**
     * Inner class to hold the results of eigenvalue decomposition.
     */
    public static class EigenResult {
        public double[] eigenvalues;      // Array of eigenvalues
        public SimpleMatrix eigenvectors; // Matrix of eigenvectors
        public SimpleMatrix X;            // Matrix X used in further calculations

        EigenResult(double[] eigenvalues, SimpleMatrix eigenvectors, SimpleMatrix X) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
            this.X = X;
        }
    }

    /**
     * Calculates the eigenvalues, eigenvectors, and X matrix from the provided Laplacian matrix.
     *
     * @param laplacian_matrix      The Laplacian matrix of the graph.
     * @param number_of_eigenvectors The number of desired eigenvectors to compute during eigen decomposition.
     * @return EigenResult containing eigenvalues, eigenvectors, and X matrix.
     */
    public static Object calculateEigen(RealMatrix laplacian_matrix, double number_of_eigenvectors) {
        // Convert RealMatrix to DMatrixRMaj (EJML data structure)
        int rows = laplacian_matrix.getRowDimension();
        int cols = laplacian_matrix.getColumnDimension();
        DMatrixRMaj laplacianDMatrix = new DMatrixRMaj(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                laplacianDMatrix.set(i, j, laplacian_matrix.getEntry(i, j));
            }
        }

        // Check if the Laplacian matrix is symmetric
        boolean is_symmetric;
        try {
            is_symmetric = isSymmetric(laplacianDMatrix);
        } catch (IllegalArgumentException e) {
            return e.getMessage(); // Return error if matrix is not square
        }

        // Attempt primary decomposition
        EigenResult result = null;
        if (is_symmetric) {
            // Try standard eigen decomposition for symmetric matrices
            result = standardEigenDecomposition(laplacianDMatrix);
        }

        // If the matrix is not symmetric or if the standard decomposition failed, attempt QR decomposition as fallback
        if (!is_symmetric || result == null) {
            try {
                // Fallback to QR decomposition
                result = qrEigenDecomposition(laplacianDMatrix, 1000, 1e-10);
            } catch (Exception e) {
                return "Error: QR decomposition fallback failed. " + e.getMessage();
            }
        }

        // Sort eigenvalues and eigenvectors and return the result together with X matrix
        return sortEigenvaluesAndVectors(result.eigenvalues, result.eigenvectors, number_of_eigenvectors);
    }

    /**
     * Performs standard eigen decomposition on a symmetric matrix.
     *
     * @param matrix The Laplacian matrix in DMatrixRMaj format.
     * @return EigenResult containing eigenvalues and eigenvectors.
     */
    private static EigenResult standardEigenDecomposition(DMatrixRMaj matrix) {
        try {
            // Initialize the eigen decomposition
            EigenDecomposition_F64<DMatrixRMaj> eigen_decomposition = new SwitchingEigenDecomposition_DDRM(matrix.getNumRows());

            // Perform the decomposition
            if (!eigen_decomposition.decompose(matrix)) {
                return null;  // Return null if decomposition fails
            }

            // Extract eigenvalues and eigenvectors
            int num_eigenvalues = eigen_decomposition.getNumberOfEigenvalues();
            double[] eigenvalues = new double[num_eigenvalues];
            SimpleMatrix eigenvectors = new SimpleMatrix(matrix.getNumRows(), num_eigenvalues);

            for (int i = 0; i < num_eigenvalues; i++) {
                if (!eigen_decomposition.getEigenvalue(i).isReal()) {
                    continue;  // Skip complex eigenvalues
                }
                eigenvalues[i] = eigen_decomposition.getEigenvalue(i).getReal();
                DMatrixRMaj eig_vector_matrix = eigen_decomposition.getEigenVector(i);
                if (eig_vector_matrix != null) {
                    eigenvectors.setColumn(i, 0, eig_vector_matrix.getData());
                }
            }

            return new EigenResult(eigenvalues, eigenvectors, null);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Performs QR decomposition as a fallback method for eigen decomposition.
     *
     * @param matrix         The input matrix.
     * @param max_iterations Maximum number of iterations.
     * @param tolerance      Convergence tolerance.
     * @return EigenResult containing eigenvalues and eigenvectors.
     */
    private static EigenResult qrEigenDecomposition(DMatrixRMaj matrix, int max_iterations, double tolerance) {
        int n = matrix.numRows;
        DMatrixRMaj Ak = matrix.copy();
        DMatrixRMaj Q = new DMatrixRMaj(n, n);
        DMatrixRMaj R = new DMatrixRMaj(n, n);
        DMatrixRMaj Q_accumulated = CommonOps_DDRM.identity(n);
        QRDecomposition<DMatrixRMaj> qr = new QRDecompositionHouseholder_DDRM();

        // Iterate for a maximum number of iterations
        for (int iter = 0; iter < max_iterations; iter++) {
            // Perform QR decomposition
            if (!qr.decompose(Ak)) {
                throw new RuntimeException("QR Decomposition failed at iteration " + iter);
            }
            qr.getQ(Q, false);
            qr.getR(R, false);

            // Update Ak = R * Q
            DMatrixRMaj Ak_new = new DMatrixRMaj(n, n);
            CommonOps_DDRM.mult(R, Q, Ak_new);
            Ak.setTo(Ak_new);

            // Accumulate Q matrices
            DMatrixRMaj Q_new = new DMatrixRMaj(n, n);
            CommonOps_DDRM.mult(Q_accumulated, Q, Q_new);
            Q_accumulated.setTo(Q_new);

            // Check for convergence
            if (hasConverged(Ak, tolerance)) {
                break;
            }
        }

        // Extract eigenvalues from the diagonal of Ak
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = Ak.get(i, i);
        }

        // Eigenvectors are the accumulated Q matrix
        SimpleMatrix eigenvectors = SimpleMatrix.wrap(Q_accumulated);
        return new EigenResult(eigenvalues, eigenvectors, null);
    }

    /**
     * Sorts the eigenvalues and eigenvectors in ascending order and selects top-k eigenvectors for matrix X.
     *
     * @param eigenvalues           Array of eigenvalues.
     * @param eigenvectors          Matrix of eigenvectors.
     * @param number_of_eigenvectors Number of eigenvectors to select for matrix X.
     * @return EigenResult containing sorted eigenvalues, eigenvectors, and matrix X.
     */
    private static EigenResult sortEigenvaluesAndVectors(double[] eigenvalues, SimpleMatrix eigenvectors, double number_of_eigenvectors) {
        // Create an array of indices for sorting
        Integer[] sorted_indices = new Integer[eigenvalues.length];
        for (int i = 0; i < eigenvalues.length; i++) {
            sorted_indices[i] = i;
        }

        // Sort the indices based on eigenvalues
        Arrays.sort(sorted_indices, Comparator.comparingDouble(index -> eigenvalues[index]));

        // Initialize sorted arrays
        double[] sorted_eigenvalues = new double[eigenvalues.length];
        SimpleMatrix sorted_eigenvectors = new SimpleMatrix(eigenvectors.getMatrix().getNumRows(), eigenvectors.getMatrix().getNumCols());

        // Populate the sorted eigenvalues and eigenvectors
        for (int i = 0; i < eigenvalues.length; i++) {
            int original_index = sorted_indices[i];
            sorted_eigenvalues[i] = eigenvalues[original_index];
            DMatrixRMaj column_vector = eigenvectors.extractVector(false, original_index).getDDRM();
            for (int row = 0; row < sorted_eigenvectors.getNumRows(); row++) {
                sorted_eigenvectors.set(row, i, column_vector.get(row, 0));
            }
        }

        // Determine the number of top eigenvectors to select
        int k = (int) ((number_of_eigenvectors > 0) ? number_of_eigenvectors : calculateOptimalK(sorted_eigenvalues));

        // Select top-k eigenvectors for matrix X
        SimpleMatrix X = sorted_eigenvectors.extractMatrix(0, sorted_eigenvectors.getMatrix().getNumRows(),
                sorted_eigenvectors.getMatrix().getNumCols() - k, sorted_eigenvectors.getMatrix().getNumCols());

        return new EigenResult(sorted_eigenvalues, sorted_eigenvectors, X);
    }

    /**
     * Checks if the matrix has converged based on the sub-diagonal elements.
     *
     * @param matrix        The matrix at the current iteration.
     * @param tolerance Convergence tolerance.
     * @return True if the matrix has converged, false otherwise.
     */
    private static boolean hasConverged(DMatrixRMaj matrix, double tolerance) {
        int n = matrix.numRows;
        for (int i = 1; i < n; i++) {
            if (Math.abs(matrix.get(i, i - 1)) > tolerance) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculates the optimal number of eigenvectors (k) based on the largest eigen gap.
     *
     * @param eigenvalues Array of sorted eigenvalues.
     * @return Optimal number of eigenvectors.
     */
    private static int calculateOptimalK(double[] eigenvalues) {
        int index = EigenGap.findLargestEigenGap(eigenvalues);
        return index + 1;
    }

    /**
     * Checks if a matrix is symmetric.
     *
     * @param matrix The matrix to check.
     * @return True if the matrix is symmetric, false otherwise.
     * @throws IllegalArgumentException if the matrix is not square.
     */
    public static boolean isSymmetric(DMatrixRMaj matrix) {
        if (matrix.numRows != matrix.numCols) {
            throw new IllegalArgumentException("Error: Laplacian matrix is not square, so eigen decomposition cannot be applied.");
        }
        DMatrixRMaj transpose = new DMatrixRMaj(matrix.numCols, matrix.numRows);
        CommonOps_DDRM.transpose(matrix, transpose);

        double tolerance = 1e-10;
        return MatrixFeatures_DDRM.isIdentical(matrix, transpose, tolerance);
    }

    /**
     * Creates an edge list based on the provided node properties, eigenvectors, and original edge list.
     *
     * @param node_properties_list List of node properties.
     * @param X                    The matrix of eigenvectors.
     * @param original_edge_list   The original edge list.
     * @return ArrayList of EdgeList2 representing the new edge list.
     */
    public static ArrayList<EdgeList2> createEdgeList(List<NodeList2> node_properties_list, SimpleMatrix X, ArrayList<EdgeList2> original_edge_list) {
        ArrayList<EdgeList2> edge_list = new ArrayList<>();

        int num_rows = X.getNumRows();
        // Compute the pairwise Euclidean distance matrix
        Double[][] distance_matrix = euclideanDistance(X);

        // Iterate over all pairs of nodes
        for (int i = 0; i < num_rows; i++) {
            for (int j = i + 1; j < num_rows; j++) {
                double distance = distance_matrix[i][j];

                String source_id = node_properties_list.get(i).getIndex();
                String target_id = node_properties_list.get(j).getIndex();

                // Check if there is an original edge between the nodes
                boolean has_original_edge = original_edge_list.stream()
                        .anyMatch(edge -> (edge.getSource().equals(source_id) && edge.getTarget().equals(target_id))
                                || (edge.getSource().equals(target_id) && edge.getTarget().equals(source_id)));

                if (has_original_edge) {
                    // Add the edge to the list with the computed distance
                    edge_list.add(new EdgeList2(source_id, target_id, distance, i, null));
                }
            }
        }

        return edge_list;
    }

    /**
     * Computes the pairwise Euclidean distance matrix from the given matrix X.
     *
     * @param X The matrix of eigenvectors.
     * @return A 2D array containing pairwise distances.
     */
    public static Double[][] euclideanDistance(SimpleMatrix X) {
        int size = X.getNumRows();
        Double[][] distance_matrix = new Double[size][size];

        // Compute distances between all pairs of rows in X
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double distance = calculateEuclideanDistance(X.extractVector(true, i), X.extractVector(true, j));
                distance_matrix[i][j] = distance;
            }
        }
        return distance_matrix;
    }

    /**
     * Calculates the Euclidean distance between two vectors.
     *
     * @param v1 First vector.
     * @param v2 Second vector.
     * @return The Euclidean distance between v1 and v2.
     */
    private static double calculateEuclideanDistance(SimpleMatrix v1, SimpleMatrix v2) {
        return v1.minus(v2).normF();
    }

}