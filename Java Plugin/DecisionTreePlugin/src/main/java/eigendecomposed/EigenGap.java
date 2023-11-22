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

    public static void main(String[] args) {
        EigenGap eigenGap = new EigenGap();

        EigenResult eigenResult = EigenCalculation.calculateEigen(null);
		double[] eigenvalues = eigenResult.eigenvalues;
		DefaultCategoryDataset dataset = eigenGap.calculateEigenGapSpectrum(eigenvalues);
		eigenGap.visualizeEigenGapSpectrum(dataset);

		// Print eigen gap values
		printEigenGapValues(eigenvalues);

		// Print and identify the largest eigen gap
		double largestEigenGap = findLargestEigenGap(eigenvalues);
		System.out.println("Largest Eigen Gap: " + largestEigenGap);
    }

    public DefaultCategoryDataset calculateEigenGapSpectrum(double[] eigenvalues) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        int numEigenvalues = eigenvalues.length - 1;

        for (int i = 0; i < numEigenvalues; i++) {
            dataset.addValue(Math.abs(eigenvalues[i + 1] - eigenvalues[i]), "Eigen Gap", String.valueOf(i + 1));
        }

        return dataset;
    }

    public void visualizeEigenGapSpectrum(DefaultCategoryDataset dataset) {
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
            double eigenGap = Math.abs(eigenvalues[i + 1] - eigenvalues[i]);
            System.out.println("Index: " + (i + 1) + ", Eigen Gap: " + eigenGap);
        }
        System.out.println();
    }

    public static double findLargestEigenGap(double[] eigenvalues) {
        double largestEigenGap = 0;
        for (int i = 0; i < eigenvalues.length - 1; i++) {
            double eigenGap = Math.abs(eigenvalues[i + 1] - eigenvalues[i]);
            if (eigenGap > largestEigenGap) {
                largestEigenGap = eigenGap;
            }
        }
        return largestEigenGap;
    }
}
