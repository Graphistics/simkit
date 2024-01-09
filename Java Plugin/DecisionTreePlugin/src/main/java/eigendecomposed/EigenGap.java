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

//    public static void main(String[] args) {
//        EigenGap eigenGap = new EigenGap();
//
//        EigenResult eigenResult = EigenCalculation.calculateEigen(null);
//		double[] eigenvalues = {-2.42514341e-01, -2.26311061e-01, -2.23027564e-01, -2.01539742e-01,
//			       -1.96689228e-01, -1.93091582e-01, -1.79766012e-01, -1.74297388e-01,
//			       -1.70254639e-01, -1.69393091e-01, -1.63446146e-01, -1.63160132e-01,
//			       -1.57833023e-01, -1.57454231e-01, -1.56492731e-01, -1.55658915e-01,
//			       -1.54691624e-01, -1.49925291e-01, -1.48399258e-01, -1.47607474e-01,
//			       -1.45847299e-01, -1.45750796e-01, -1.44641038e-01, -1.43453432e-01,
//			       -1.43299456e-01, -1.42085810e-01, -1.40245760e-01, -1.38637917e-01,
//			       -1.36336411e-01, -1.36299318e-01, -1.35716646e-01, -1.34790400e-01,
//			       -1.33585607e-01, -1.33239968e-01, -1.33137154e-01, -1.32881248e-01,
//			       -1.32316216e-01, -1.31737266e-01, -1.31039260e-01, -1.30689521e-01,
//			       -1.29943653e-01, -1.29418808e-01, -1.28784771e-01, -1.28670060e-01,
//			       -1.27997837e-01, -1.26905543e-01, -1.26725500e-01, -1.26485914e-01,
//			       -1.25833195e-01, -1.25812644e-01, -1.25310171e-01, -1.24844449e-01,
//			       -1.24844449e-01, -1.24100096e-01, -1.23380798e-01, -1.22770192e-01,
//			       -1.21548168e-01, -1.21397795e-01, -1.21029220e-01, -1.20374338e-01,
//			       -1.19903767e-01, -1.18448335e-01, -1.17728100e-01, -1.17505095e-01,
//			       -1.17414958e-01, -1.17092899e-01, -1.16632523e-01, -1.15843208e-01,
//			       -1.15216442e-01, -1.14805303e-01, -1.14274313e-01, -1.14051338e-01,
//			       -1.13449905e-01, -1.12402703e-01, -1.12190743e-01, -1.11867296e-01,
//			       -1.11575593e-01, -1.10543116e-01, -1.10100215e-01, -1.09526059e-01,
//			       -1.08882094e-01, -1.08511492e-01, -1.07953964e-01, -1.07287669e-01,
//			       -1.06967310e-01, -1.06331613e-01, -1.05633946e-01, -1.04647060e-01,
//			       -1.03909389e-01, -1.03179191e-01, -1.02772508e-01, -1.01662651e-01,
//			       -1.00028809e-01, -9.99056407e-02, -9.95871952e-02, -9.89772004e-02,
//			       -9.87394455e-02, -9.76121035e-02, -9.52346485e-02, -9.49201811e-02,
//			       -9.48116453e-02, -9.37099848e-02, -9.32471451e-02, -9.27140570e-02,
//			       -9.21953759e-02, -9.15958110e-02, -8.92181601e-02, -8.89811274e-02,
//			       -8.80712713e-02, -8.67426983e-02, -8.58071494e-02, -8.44270521e-02,
//			       -8.31063973e-02, -8.17306882e-02, -8.11038365e-02, -7.87108545e-02,
//			       -7.66890648e-02, -7.63346882e-02, -7.55729847e-02, -7.34220831e-02,
//			       -7.28199551e-02, -6.97052788e-02, -6.90321871e-02, -6.64567433e-02,
//			       -6.37642499e-02, -6.13337359e-02, -5.88746395e-02, -5.64537313e-02,
//			       -5.50518035e-02, -5.38188713e-02, -5.17545862e-02, -4.79853441e-02,
//			       -4.67808621e-02, -4.43532029e-02, -4.35247012e-02, -4.17481227e-02,
//			       -3.99994360e-02, -3.43606420e-02, -3.18852192e-02, -3.03313370e-02,
//			       -2.65218299e-02, -2.42332971e-02, -2.17997851e-02, -1.73566047e-02,
//			       -1.71919654e-02, -6.77230431e-03, -4.73800035e-03,  2.06469722e-04,
//			        2.63898184e-03,  1.25095170e-02,  1.40847379e-02,  1.89813039e-02,
//			        2.66846793e-02,  2.80901296e-02,  3.41438342e-02,  3.82308764e-02,
//			        4.31018873e-02,  4.41374350e-02,  4.89351254e-02,  6.35574493e-02,
//			        6.39882030e-02,  6.84401241e-02,  7.04146640e-02,  7.91960737e-02,
//			        8.95572965e-02,  9.64156692e-02,  1.04726369e-01,  1.11012505e-01,
//			        1.22086777e-01,  1.54520093e-01,  1.65274286e-01,  1.74201350e-01,
//			        1.95393749e-01,  2.01071965e-01,  2.05743429e-01,  2.18978917e-01,
//			        2.22122778e-01,  2.47536338e-01,  2.54155418e-01,  2.71248757e-01,
//			        2.72332024e-01,  2.89424020e-01,  2.92001428e-01,  3.33261905e-01,
//			        3.59108491e-01,  3.92842424e-01,  4.24825496e-01,  5.02855150e-01,
//			        5.90058846e-01,  6.09243153e-01,  6.25177190e-01,  6.27361411e-01,
//			        7.48690308e-01,  8.15298074e-01,  8.44775008e-01,  8.54530411e-01,
//			        9.36529036e-01,  9.86148680e-01,  9.97767885e-01,  1.00000000e+00};
//		DefaultCategoryDataset dataset = EigenGap.calculateEigenGapSpectrum(eigenvalues);
////		EigenGap.visualizeEigenGapSpectrum(dataset);
//
//		// Print eigen gap values
//		printEigenGapValues(eigenvalues);
//
//		// Print and identify the largest eigen gap
//		double largestEigenGap = findLargestEigenGap(eigenvalues) +1;
//		System.out.println("Largest Eigen Gap: " + largestEigenGap);
//    }

    public static DefaultCategoryDataset calculateEigenGapSpectrum(double[] eigenvalues) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        int numEigenvalues = eigenvalues.length - 1;

        for (int i = 0; i < numEigenvalues; i++) {
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
            double eigenGap = Math.abs(eigenvalues[i + 1] - eigenvalues[i]);
            System.out.println("Index: " + (i + 1) + ", Eigen Gap: " + eigenGap);
        }
        System.out.println();
    }

    public static double findLargestEigenGap(double[] eigenvalues) {
        double largestEigenGap = 0;
        int indexOfLargestGap = 0;
        
        for (int i = 0; i < eigenvalues.length - 1; i++) {
            double eigenGap = Math.abs(eigenvalues[i + 1] - eigenvalues[i]);
            if (eigenGap > largestEigenGap) {
                largestEigenGap = eigenGap;
                indexOfLargestGap = i;
            }
        }
//        return largestEigenGap;
        return indexOfLargestGap;
    }
}
