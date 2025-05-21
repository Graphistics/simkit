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
     * Finds the index of the largest eigen gap among the first 10 eigenvalues.
     *
     * @param eigenvalues Array of eigenvalues sorted in ascending order.
     * @return Index of the largest eigen gap where the index is less than 10.
     */
    public static int findLargestEigenGap(double[] eigenvalues) {
        double largest_eigen_gap = 0;
        int index_of_largest_gap = 0;
        int max_index = Math.min(10, eigenvalues.length - 1); // Limit index to be under 10 or within array bounds

        for (int i = 0; i < max_index; i++) {
            double eigen_gap = Math.abs(eigenvalues[i + 1] - eigenvalues[i]);
            if (eigen_gap > largest_eigen_gap) {
                largest_eigen_gap = eigen_gap;
                index_of_largest_gap = i;
            }
        }
        return index_of_largest_gap;
    }


}
