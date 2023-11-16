package eigendecomposed;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.math4.legacy.linear.BlockRealMatrix;
import org.apache.commons.math4.legacy.linear.EigenDecomposition;
import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;

import definition.EdgeList2;
import definition.NodeList2;

public class EigenCalculation {

    public static class EigenResult {
        public double[] eigenvalues;
        public RealMatrix eigenvectors;
        RealMatrix X;

        EigenResult(double[] eigenvalues, RealMatrix eigenvectors, RealMatrix X) {
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
            this.X = X;
        }
    }

    public static EigenResult calculateEigen(RealMatrix laplacianMatrix) {
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
            int k = 2;
            RealMatrix X = sortedEigenvectors.getSubMatrix(0, dimension - 1, dimension - k, dimension - 1);
            return new EigenResult(sortedEigenvalues, sortedEigenvectors, X);
        } catch (Exception e) {
            e.printStackTrace();
            return new EigenResult(new double[0], MatrixUtils.createRealMatrix(0, 0), MatrixUtils.createRealMatrix(0, 0));
        }
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
    
    public static ArrayList<EdgeList2> createEdgeList(List<NodeList2> nodePropertiesList, RealMatrix eigenvectors, double threshold) {
        ArrayList<EdgeList2> edgeList = new ArrayList<>();

        int numRows = eigenvectors.getRowDimension();

        for (int i = 0; i < numRows; i++) {
            for (int j = i + 1; j < numRows; j++) {
                double value = eigenvectors.getEntry(i, j);

                if (Math.abs(value) >= threshold) {
                    String sourceId = nodePropertiesList.get(i).getIndex();
                    String targetId = nodePropertiesList.get(j).getIndex();
                    edgeList.add(new EdgeList2(sourceId, targetId, value, i, null));
                }
            }
        }

        return edgeList;
    }


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