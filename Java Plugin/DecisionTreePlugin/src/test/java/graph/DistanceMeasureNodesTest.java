package graph;

import definition.NodeList2;
import main.OutputDecisionTreeNeo4j;
import org.junit.jupiter.api.Test;
import java.io.IOException;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

class DistanceMeasureNodesTest {

    String dataPath = "D:/de/MASTER_THESIS/Decision-Tree-Neo4j/Java Plugin/DecisionTreePlugin/src/main/resources/test.csv";
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
        Double[][] DistanceMatrix = OutputDecisionTreeNeo4j.getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);

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
        Double[][] DistanceMatrix = OutputDecisionTreeNeo4j.getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);
        
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


    private static void roundMatrixTwoDecimalPlaces(Double[][] DistanceMatrix) {
        for (int i = 0; i < DistanceMatrix.length; i++) {
            for (int j = 0; j < DistanceMatrix.length; j++) {
                DistanceMatrix[i][j] = Math.round(DistanceMatrix[i][j] * 100.0) / 100.0;
            }
        }
    }
}