package simkit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import cv.CrossValidation;

/**
 * This class is used to run cross validation on the entire sample 
 * Dataset1 - heart_failure_original.csv . Target - 'DEATH_EVENT'
 * Dataset2 - flu_classification.csv. Target - 'Diagnosis'
 * Dataset3 - metaprotein_50.csv. Target - 'Patient_Type'
 * 
 * AlgorithmType - 'InfoGain', 'GiniIndex', 'GainRatio'
 *
 * @author Nasim
 *
 */

public class SimKit {
    public static void simkit(String[] args) throws IOException {

		Scanner in = new Scanner(System.in);
		System.out.println("===============================================================");

		CrossValidation cv = new CrossValidation("data/metaprotein_50.csv", "Diagnosis");

		ArrayList<Double> final_score = cv.validate(10, "GiniIndex");

		ArrayList<Double> totalGenerationTime = cv.getCvGenerationTime();

		ArrayList<Double> totalRepeatsAccuracy = new ArrayList<>();
		ArrayList<Double> totalRepeatsTime = new ArrayList<>();

		for (int i = 0; i < 30; i++) {

			double accuracy = calculateAverage(final_score);
			double gTime = calculateAverage(totalGenerationTime);

			totalRepeatsAccuracy.add(accuracy);
			totalRepeatsTime.add(gTime);

		}

		System.out.println("Accuracy:" + calculateAverage(totalRepeatsAccuracy) * 100 + "%");
		System.out.println("GenerationTime:" + calculateAverage(totalRepeatsTime) + "sec");

		in.close();
    }
    
    private static double calculateAverage(ArrayList<Double> final_score) {
        return final_score.stream()
                    .mapToDouble(d -> d)
                    .average()
                    .orElse(0.0);
    }
    
    
    
    
}