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
    
    public static EigenResult calculateEigen(RealMatrix laplacianMatrix, double userDefinedK) {
        try {
            EigenDecomposition eigenDecomposition = new EigenDecomposition(laplacianMatrix);
            RealMatrix v = eigenDecomposition.getV();
            double[] e = eigenDecomposition.getRealEigenvalues();

            // Sort eigenvalues and eigenvectors in ascending order
            Integer[] sortedIndices = new Integer[e.length];
            for (int i = 0; i < e.length; i++) {
                sortedIndices[i] = i;
            }

            Arrays.sort(sortedIndices, Comparator.comparingDouble(index -> e[index]));

            double[] sortedEigenvalues = new double[e.length];
            RealMatrix sortedEigenvectors = new BlockRealMatrix(v.getRowDimension(), v.getColumnDimension());

            for (int i = 0; i < e.length; i++) {
                int originalIndex = sortedIndices[i];
                sortedEigenvalues[i] = e[originalIndex];
                sortedEigenvectors.setColumnVector(i, v.getColumnVector(originalIndex));
            }


            int dimension = laplacianMatrix.getColumnDimension();
            int k = (int) ((userDefinedK > 0) ? userDefinedK : calculateOptimalK(sortedEigenvalues));
            
            // Round eigenvalues to 7 decimal places
            DecimalFormat decimalFormat = new DecimalFormat("#.#######");
            for (int i = 0; i < sortedEigenvalues.length; i++) {
                sortedEigenvalues[i] = Double.parseDouble(decimalFormat.format(sortedEigenvalues[i]));
            }

            // Round eigenvectors to 7 decimal places
            for (int i = 0; i < sortedEigenvectors.getRowDimension(); i++) {
                for (int j = 0; j < sortedEigenvectors.getColumnDimension(); j++) {
                    sortedEigenvectors.setEntry(i, j, Double.parseDouble(decimalFormat.format(sortedEigenvectors.getEntry(i, j))));
                }
            }
            RealMatrix X = sortedEigenvectors.getSubMatrix(0, dimension - 1, dimension - k, dimension - 1);
            return new EigenResult(sortedEigenvalues, sortedEigenvectors, X);
        } catch (Exception e) {
            e.printStackTrace();
            return new EigenResult(new double[0], MatrixUtils.createRealMatrix(0, 0), MatrixUtils.createRealMatrix(0, 0));
        }
    }
    

    private static int calculateOptimalK(double[] eigenvalues) {
        double largestEigenGap = EigenGap.findLargestEigenGap(eigenvalues);
        return (int) Math.round(largestEigenGap);
    }


    private static void displayEigenResult(EigenResult eigenResult) {
        displayArray(eigenResult.eigenvalues, "eigenvalues");
        displayMatrix(eigenResult.eigenvectors, "eigenvectors");
        displayMatrix(eigenResult.X, "X matrix");
    }

    private static void displayArray(double[] array, String arrayName) {
        System.out.println(arrayName + ": " + Arrays.toString(array));
        System.out.println();
    }

    private static void displayMatrix(RealMatrix matrix, String matrixName) {
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
    
    public static ArrayList<EdgeList2> createEdgeList(List<NodeList2> nodePropertiesList, RealMatrix X, ArrayList<EdgeList2> originalEdgeList) {
        ArrayList<EdgeList2> edgeList = new ArrayList<>();

        int numRows = X.getRowDimension();
        Double[][] distanceMatrix = euclideanDistance(X);

        for (int i = 0; i < numRows; i++) {
            for (int j = i + 1; j < numRows; j++) {
                double distance = distanceMatrix[i][j];

                    String sourceId = nodePropertiesList.get(i).getIndex();
                    String targetId = nodePropertiesList.get(j).getIndex();
                    
                    boolean hasOriginalEdge = originalEdgeList.stream()
                            .anyMatch(edge -> (edge.getSource().equals(sourceId) && edge.getTarget().equals(targetId))
                                    || (edge.getSource().equals(targetId) && edge.getTarget().equals(sourceId)));

                    if (hasOriginalEdge) {
                    edgeList.add(new EdgeList2(sourceId, targetId, distance, i, null));
                    }
            }
        }

        return edgeList;
    }

    public static Double[][] euclideanDistance(RealMatrix X) {
        int size = X.getRowDimension();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double distance = calculateEuclideanDistance(X.getRowVector(i), X.getRowVector(j));
                distanceMatrix[i][j] = distance;
            }
        }
        return distanceMatrix;
    }

    private static double calculateEuclideanDistance(RealVector v1, RealVector v2) {
        return v1.getDistance(v2);
    }



    
//    public static ArrayList<EdgeList2> createEdgeList(List<NodeList2> nodePropertiesList, RealMatrix eigenvectors, double threshold) {
//        ArrayList<EdgeList2> edgeList = new ArrayList<>();
//
//        int numRows = eigenvectors.getRowDimension();
//
//        for (int i = 0; i < numRows; i++) {
//            for (int j = i + 1; j < numRows; j++) {
//                double value = eigenvectors.getEntry(i, j);
//
//                if (Math.abs(value) >= threshold) {
//                    String sourceId = nodePropertiesList.get(i).getIndex();
//                    String targetId = nodePropertiesList.get(j).getIndex();
//                    edgeList.add(new EdgeList2(sourceId, targetId, value, i, null));
//                }
//            }
//        }
//
//        return edgeList;
//    }


    public static ArrayList<NodeList2> createNodeList(RealMatrix eigenvectors, double threshold) {
        ArrayList<NodeList2> nodeList = new ArrayList<>();

        int numRows = eigenvectors.getRowDimension();

        for (int i = 0; i < numRows; i++) {
            double value = eigenvectors.getEntry(i, i);

            if (Math.abs(value) >= threshold) {
                nodeList.add(new NodeList2(Integer.toString(i), null));
            }
        }

        return nodeList;
    }
}