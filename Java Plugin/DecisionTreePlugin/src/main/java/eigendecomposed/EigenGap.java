package eigendecomposed;

import java.awt.Color;
import java.awt.Dimension;

import javax.swing.JFrame;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import eigendecomposed.EigenCalculation.EigenResult;


public class EigenGap {

    public static DefaultCategoryDataset calculateEigenGapSpectrum(double[] eigenvalues) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        int num_eigenvalues = eigenvalues.length - 1;

        for (int i = 0; i < num_eigenvalues; i++) {
            dataset.addValue(Math.abs(eigenvalues[i + 1] - eigenvalues[i]), "Eigen Gap", String.valueOf(i + 1));
        }

        return dataset;
    }

    public static void visualizeEigenGapSpectrum(DefaultCategoryDataset dataset) {
        JFreeChart chart = ChartFactory.createBarChart(
                "Eigen Gap Spectrum",
                "Index",
                "Eigen Gap",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false
        );

        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setPaint(Color.black);

        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setRangeGridlinePaint(Color.white);

        JFrame frame = new JFrame("Eigen Gap Spectrum");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(new ChartPanel(chart));
        frame.setPreferredSize(new Dimension(800, 600));
        frame.pack();
        frame.setVisible(true);
    }

    public static void printEigenGapValues(double[] eigenvalues) {
        System.out.println("Eigen Gap Values:");
        for (int i = 0; i < eigenvalues.length - 1; i++) {
            double eigen_gap = Math.abs(eigenvalues[i + 1] - eigenvalues[i]);
            System.out.println("Index: " + (i + 1) + ", Eigen Gap: " + eigen_gap);
        }
        System.out.println();
    }

    /**
     * Finds the index of the largest eigenvalue gap in the array of eigenvalues.
     *
     * @param eigenvalues The array of eigenvalues.
     * @return The index of the largest eigenvalue gap.
     */
    public static double findLargestEigenGap(double[] eigenvalues) {
        double largest_eigen_gap = 0;
        int index_of_largest_gap = 0;
        
        for (int i = 0; i < eigenvalues.length - 1; i++) {
            double eigen_gap = Math.abs(eigenvalues[i + 1] - eigenvalues[i]);
            if (eigen_gap > largest_eigen_gap) {
                largest_eigen_gap = eigen_gap;
                index_of_largest_gap = i;
            }
        }
//        return largest_eigen_gap;
        return index_of_largest_gap;
    }
}
