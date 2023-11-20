package eigendecomposed;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import javax.swing.JFrame;

import org.apache.commons.math4.legacy.linear.BlockRealMatrix;
import org.apache.commons.math4.legacy.linear.EigenDecomposition;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import graph.EdgeList;

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

    static RealMatrix laplacianMatrix = null;

    public void performEigenCalculation(RealMatrix laplacianMatrix) throws Exception {
        try {
            EigenResult eigenResult = calculateEigen(laplacianMatrix);
            displayEigenResult(eigenResult);
            drawScatterPlot(eigenResult.eigenvalues);

            double threshold = 0.1;
            ArrayList<EdgeList> edgeList = createEdgeList(eigenResult.eigenvectors, threshold);

            System.out.println("Edge List:");
            for (EdgeList edge : edgeList) {
                System.out.println("Source: " + edge.getSource() + ", Target: " + edge.getTarget() + ", Weight: " + edge.getWeight());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static EigenResult calculateEigen(RealMatrix laplacianMatrix) {
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

    public static void drawScatterPlot(double[] eigenvalues) {
        XYSeries series = new XYSeries("Scatter Plot");

        int numEigenvalues = eigenvalues.length;
        for (int i = 0; i < numEigenvalues; i++) {
            series.add(i, eigenvalues[i]);
        }

        XYSeriesCollection dataset = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory.createScatterPlot(
                "Eigenvalues Scatter Plot",
                "Index",
                "Eigenvalues",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setDomainPannable(true);
        plot.setRangePannable(true);

        XYDotRenderer renderer = new XYDotRenderer();
        renderer.setDotWidth(10);
        renderer.setDotHeight(10);
        renderer.setSeriesShape(0, new Rectangle2D.Double(-5, -5, 10, 10));
        renderer.setSeriesPaint(0, Color.BLACK);
        plot.setRenderer(renderer);

        // Set the x-axis range from 0 to 4
        NumberAxis xAxis = (NumberAxis) plot.getDomainAxis();
        xAxis.setRange(0, 4);

        JFrame frame = new JFrame("Eigenvalues Scatter Plot");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new ChartPanel(chart));
        frame.setPreferredSize(new Dimension(800, 600));
        frame.pack();
        frame.setVisible(true);
    }

    public static ArrayList<EdgeList> createEdgeList(RealMatrix eigenvectors, double threshold) {
        ArrayList<EdgeList> edgeList = new ArrayList<>();

        int numRows = eigenvectors.getRowDimension();

        for (int i = 0; i < numRows; i++) {
            for (int j = i + 1; j < numRows; j++) {
                double value = eigenvectors.getEntry(i, j);

               
                if (Math.abs(value) >= threshold) {
                    edgeList.add(new EdgeList(i, j, value, i));
                }
            }
        }

        return edgeList;
    }
                
}
