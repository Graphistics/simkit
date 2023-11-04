package eigendecomposed;

import java.util.ArrayList;

import org.apache.commons.math4.legacy.linear.BlockRealMatrix;
import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;

import graph.EdgeList;

public class MatrixCalculation {

    public enum LaplacianAlgorithm {
        NJW, RANDOM_WALK
    }

    private static LaplacianAlgorithm algorithm;

    public MatrixCalculation(LaplacianAlgorithm algorithm) {
        MatrixCalculation.algorithm = algorithm;
    }

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
   

    public static double[][] convertToAdjacencyMatrix(ArrayList<EdgeList> edgeList) {
        int dimension = edgeList.size();
        double[][] adjacencyMatrixData = new double[dimension][dimension];

        for (EdgeList edge : edgeList) {
            int i = (int) edge.getSource();
            int j = (int) edge.getTarget();
            double weight = edge.getWeight();
            adjacencyMatrixData[i][j] = weight;
            adjacencyMatrixData[j][i] = weight;
        }

        return adjacencyMatrixData;
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
        int dimension = degreeMatrix.getColumnDimension();
        RealMatrix inverseDegreeMatrix = MatrixUtils.createRealMatrix(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            double inverseDegreeValue = 1.0 / degreeMatrix.getEntry(i, i);
            inverseDegreeMatrix.setEntry(i, i, inverseDegreeValue);
        }

        RealMatrix randomWalkLaplacianMatrix = MatrixUtils.createRealIdentityMatrix(dimension).subtract(
                inverseDegreeMatrix.multiply(adjacencyMatrix)
        );

        return randomWalkLaplacianMatrix;
    }

    
}
