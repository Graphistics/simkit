package graph;

import definition.NodeList2;
import simkit.SimKitProcedures;
import org.junit.jupiter.api.Test;
import java.io.IOException;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

class DistanceMeasureNodesTest {

    String dataPath = "././././data/test.csv";
    ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
    ArrayList<NodeList2> nodePropertiesList = readCsvTestData.readCsvFileToMap(dataPath);
    ArrayList<String> removeListNew = new ArrayList<>();

    DistanceMeasureNodesTest() throws IOException {
        removeListNew.add("points");
        removeListNew.add("class");
    }


    @Test
    void euclideanDistance(){
        String distance_measure = "euclidean";
        Double[][] DistanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);

        // round the values to 2 decimal places
        roundMatrixTwoDecimalPlaces(DistanceMatrix);

        Double[][] expected = new Double[][]{
                {0.00, 1.00, 7.07, 9.22, 10.30},
                {1.00, 0.00, 6.40, 8.60, 9.85},
                {7.07, 6.40, 0.00, 2.24, 4.00},
                {9.22, 8.60, 2.24, 0.00, 2.24},
                {10.30, 9.85, 4.00, 2.24, 0.00}};

        assertArrayEquals(expected, DistanceMatrix);


    }

    @Test
    void manhattanDistance(){
        String distance_measure = "manhattan";
        Double[][] DistanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);
        
        // round the values to 2 decimal places
        roundMatrixTwoDecimalPlaces(DistanceMatrix);

        Double[][] expected = new Double[][]{
                {0.00, 1.00, 10.00, 13.00, 14.00},
                {1.00, 0.00, 9.00, 12.00, 13.00},
                {10.00, 9.00, 0.00, 3.00, 4.00},
                {13.00, 12.00, 3.00, 0.00, 3.00},
                {14.00, 13.00, 4.00, 3.00, 0.00}};

    assertArrayEquals(expected, DistanceMatrix);

    }
    @Test
    void canberraDistance(){
        String distance_measure = "canberra";
        Double[][] DistanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);

        // round the values to 2 decimal places
        roundMatrixTwoDecimalPlaces(DistanceMatrix);

        Double[][] expected = new Double[][]{
                {0.00, 0.08, 1.27, 1.53, 1.37},
                {0.08, 0.00, 1.21, 1.49, 1.32} ,
                {1.27, 1.21, 0.00, 0.48, 0.25} ,
                {1.53, 1.49, 0.48, 0.00, 0.44} ,
                {1.37, 1.32, 0.25, 0.44, 0.00}};

        assertArrayEquals(expected, DistanceMatrix);


    }
    @Test
    void cosineSimilarity() {
        String distance_measure = "cosine";
        Double[][] DistanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes(distance_measure, nodePropertiesList, removeListNew);

        // round the values to 2 decimal places
        roundMatrixTwoDecimalPlaces(DistanceMatrix);

        Double[][] expected = new Double[][]{
                {0.00, 1.0, 0.45, 0.26, 0.33},
                {1.0, 0.00, 0.47, 0.29, 0.35},
                {0.45, 0.47, 0.00, 0.98, 0.99},
                {0.26, 0.29, 0.98, 0.00, 1.0},
                {0.33, 0.35, 0.99, 1.0, 0.00}};
        assertArrayEquals(expected, DistanceMatrix);
    }

    @Test
    void brayCurtisDistance(){
        String distance_measure = "bray_curtis";
        Double[][] DistanceMatrix = SimKitProcedures.getDistanceMatrixFromNodes(distance_measure, nodePropertiesList, removeListNew);

        // round the values to 2 decimal places
        roundMatrixTwoDecimalPlaces(DistanceMatrix);

        Double[][] expected = new Double[][]{
                {0.00, 0.07, 0.63, 0.76, 0.7},
                {0.07, 0.00, 0.60, 0.75, 0.68},
                {0.63, 0.60, 0.00, 0.18, 0.20},
                {0.76, 0.75, 0.18, 0.00, 0.14},
                {0.70, 0.68, 0.20, 0.14, 0.00}};
        assertArrayEquals(expected, DistanceMatrix);
    }

    private static void roundMatrixTwoDecimalPlaces(Double[][] DistanceMatrix) {
        for (int i = 0; i < DistanceMatrix.length; i++) {
            for (int j = 0; j < DistanceMatrix.length; j++) {
                DistanceMatrix[i][j] = Math.round(DistanceMatrix[i][j] * 100.0) / 100.0;
            }
        }
    }
}