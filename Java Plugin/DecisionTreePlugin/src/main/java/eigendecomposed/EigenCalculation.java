package eigendecomposed;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.math4.legacy.linear.BlockRealMatrix;
import org.apache.commons.math4.legacy.linear.EigenDecomposition;
import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;

import definition.EdgeList2;
import definition.NodeList2;

public class EigenCalculation {

    public static class EigenResult {
        public double[] eigenvalues;
        public RealMatrix eigenvectors;
        public RealMatrix X;

        EigenResult(double[] eigenvalues, RealMatrix eigenvectors, RealMatrix X) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
            this.X = X;
        }
    }
    
    /**
     * Calculates the eigenvalues, eigenvectors, and X matrix from the provided Laplacian matrix.
     *
     * @param laplacianMatrix The Laplacian matrix of the graph.
     * @param number_of_eigenvectors The number of desired eigenvectors to compute during eigen decomposition.
     * @return EigenResult containing eigenvalues, eigenvectors, and X matrix.
     */
    public static EigenResult calculateEigen(RealMatrix laplacian_matrix, double number_of_eigenvectors) {
        try {
            EigenDecomposition eigen_decomposition = new EigenDecomposition(laplacian_matrix);
            RealMatrix v = eigen_decomposition.getV();
            double[] e = eigen_decomposition.getRealEigenvalues();

            // Sort eigenvalues and eigenvectors in ascending order
            Integer[] sorted_indices = new Integer[e.length];
            for (int i = 0; i < e.length; i++) {
                sorted_indices[i] = i;
            }

            Arrays.sort(sorted_indices, Comparator.comparingDouble(index -> e[index]));

            double[] sorted_eigenvalues = new double[e.length];
            RealMatrix sorted_eigenvectors = new BlockRealMatrix(v.getRowDimension(), v.getColumnDimension());

            for (int i = 0; i < e.length; i++) {
                int original_index = sorted_indices[i];
                sorted_eigenvalues[i] = e[original_index];
                sorted_eigenvectors.setColumnVector(i, v.getColumnVector(original_index));
            }


            int dimension = laplacian_matrix.getColumnDimension();
            int k = (int) ((number_of_eigenvectors > 0) ? number_of_eigenvectors : calculateOptimalK(sorted_eigenvalues));
            
//            // Round eigenvalues to 7 decimal places
//            DecimalFormat decimal_format = new DecimalFormat("#.#######");
//            for (int i = 0; i < sorted_eigenvalues.length; i++) {
//                sorted_eigenvalues[i] = Double.parseDouble(decimal_format.format(sorted_eigenvalues[i]));
//            }
//
//            // Round eigenvectors to 7 decimal places
//            for (int i = 0; i < sorted_eigenvectors.getRowDimension(); i++) {
//                for (int j = 0; j < sorted_eigenvectors.getColumnDimension(); j++) {
//                    sorted_eigenvectors.setEntry(i, j, Double.parseDouble(decimal_format.format(sorted_eigenvectors.getEntry(i, j))));
//                }
//            }
            
            RealMatrix X = sorted_eigenvectors.getSubMatrix(0, dimension - 1, dimension - k, dimension - 1);
            return new EigenResult(sorted_eigenvalues, sorted_eigenvectors, X);
        } catch (Exception e) {
            e.printStackTrace();
            return new EigenResult(new double[0], MatrixUtils.createRealMatrix(0, 0), MatrixUtils.createRealMatrix(0, 0));
        }
    }
    

    private static int calculateOptimalK(double[] eigenvalues) {
        double largest_eigen_gap = EigenGap.findLargestEigenGap(eigenvalues);
        return (int) Math.round(largest_eigen_gap);
    }


    public static void displayEigenResult(EigenResult eigenResult) {
        displayArray(eigenResult.eigenvalues, "eigenvalues");
        displayMatrix(eigenResult.eigenvectors, "eigenvectors");
        displayMatrix(eigenResult.X, "X matrix");
    }

    private static void displayArray(double[] array, String arrayName) {
        System.out.println(arrayName + ": " + Arrays.toString(array));
        System.out.println();
    }

    public static void displayMatrix(RealMatrix matrix, String matrixName) {
        System.out.println(matrixName + ": ");
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                System.out.printf("%.8f ", matrix.getEntry(row, col));
            }
            System.out.println();
        }
        System.out.println();
    }
    
    /**
     * Creates an edge list based on the provided node properties, eigenvectors, and original edge list.
     *
     * @param node_properties_list List of node properties.
     * @param X                  The matrix of eigenvectors.
     * @param original_edge_list   The original edge list.
     * @return ArrayList of EdgeList2 representing the new edge list.
     */
    public static ArrayList<EdgeList2> createEdgeList(List<NodeList2> node_properties_list, RealMatrix X, ArrayList<EdgeList2> original_edge_list) {
        ArrayList<EdgeList2> edge_list = new ArrayList<>();

        int num_rows = X.getRowDimension();
        Double[][] distance_matrix = euclideanDistance(X);

        for (int i = 0; i < num_rows; i++) {
            for (int j = i + 1; j < num_rows; j++) {
                double distance = distance_matrix[i][j];

                    String source_id = node_properties_list.get(i).getIndex();
                    String target_id = node_properties_list.get(j).getIndex();
                    
                    boolean has_original_edge = original_edge_list.stream()
                            .anyMatch(edge -> (edge.getSource().equals(source_id) && edge.getTarget().equals(target_id))
                                    || (edge.getSource().equals(target_id) && edge.getTarget().equals(source_id)));

                    if (has_original_edge) {
                    edge_list.add(new EdgeList2(source_id, target_id, distance, i, null));
                    }
            }
        }

        return edge_list;
    }

    public static Double[][] euclideanDistance(RealMatrix X) {
        int size = X.getRowDimension();
        Double[][] distance_matrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double distance = calculateEuclideanDistance(X.getRowVector(i), X.getRowVector(j));
                distance_matrix[i][j] = distance;
            }
        }
        return distance_matrix;
    }

    private static double calculateEuclideanDistance(RealVector v1, RealVector v2) {
        return v1.getDistance(v2);
    }
}