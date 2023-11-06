package graph;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;

public class ReadCsvTestData {
    public String dataPath;

    public ReadCsvTestData(String dataPath){
        this.dataPath = dataPath;
    }

    String[] HEADERS = { "points","x_coordinate","y_coordinate","class"};

    public ArrayList<String> readCSVHeader(String dataPath) throws IOException {
        Reader in = new FileReader(dataPath);

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder().build();
        Iterable<CSVRecord> records = csvFormat.parse(in);
        ArrayList<String> headers = new ArrayList<>();
        for (CSVRecord record : records) {
            for (int i = 0; i < record.size(); i++) {
                headers.add(record.get(i));
            }
            break;
        }
        return headers;
    }
    public ArrayList<String> readCSVFirstLine(String dataPath) throws IOException {
        Reader in = new FileReader(dataPath);

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder().setSkipHeaderRecord(true).build();
        CSVParser records = csvFormat.parse(in);
        ArrayList<String> First = new ArrayList<>();
        records.iterator().next();
        for (CSVRecord record : records) {
            for (int i = 0; i < record.size(); i++) {
                First.add(record.get(i));
            }
            break;
        }
        return First;
    }


    public ArrayList<TestData> readCsvFile(String dataPath) throws IOException {
        ArrayList<TestData> TestDataArrayList = new ArrayList<>();
        Reader in = new FileReader(dataPath);

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader(HEADERS)
                .setSkipHeaderRecord(true)
                .build();


        Iterable<CSVRecord> records = csvFormat.parse(in);

        for (CSVRecord record : records) {
            String points = record.get("points");
            String x_coordinate = record.get("x_coordinate");
            String y_coordinate = record.get("y_coordinate");
            String Class = record.get("class");
            TestData testData = new TestData(Double.parseDouble(x_coordinate),Double.parseDouble(y_coordinate),Double.parseDouble(Class),points);
            TestDataArrayList.add(testData);
        }
        return TestDataArrayList;
    }
    public  Double[][] euclidianDistance(ArrayList<TestData> TestDataArrayList){
        Double[][] distanceMatrix = new Double[TestDataArrayList.size()][TestDataArrayList.size()];
        for (int i = 0; i < TestDataArrayList.size(); i++) {
            for (int j = 0; j < TestDataArrayList.size(); j++) {
                double distance = Math.sqrt(Math.pow(TestDataArrayList.get(i).getX_coordinate()-TestDataArrayList.get(j).getX_coordinate(),2)+Math.pow(TestDataArrayList.get(i).getY_coordinate()-TestDataArrayList.get(j).getY_coordinate(),2));
                distanceMatrix[i][j] = distance;
//                System.out.println(distance);
            }
        }
        return distanceMatrix;

    }
    public static Double[] calculateKNN(Double[][] pdist) {
        Double[] sigmas = new Double[5];

        for (int i = 0; i < pdist.length; i++) {
            Double[] sortedDistances = Arrays.copyOf(pdist[i], pdist[i].length);
            Arrays.sort(sortedDistances);
            sigmas[i] = sortedDistances[1];
        }

        return sigmas;
    }
    public static Double[] calculateLocalSigmas(Double[][] pdist) {
        Double[] sigmas = new Double[5];

        for (int i = 0; i < pdist.length; i++) {
            Double[] sortedDistances = Arrays.copyOf(pdist[i], pdist[i].length);
            Arrays.sort(sortedDistances, Collections.reverseOrder());
            sigmas[i] = sortedDistances[2];
        }

        return sigmas;
    }
    public static Double[][] calculateAdjacencyMatrix(Double[][] dist_, Double[] sigmas) {
        Double[][] adj = new Double[5][5];

        for (int i = 0; i < dist_.length; i++) {
            for (int j = 0; j < dist_[i].length; j++) {
                if (i == j) {
                    adj[i][j] = 0.0;
                    continue;
                }
                //adj[i][j] = Math.exp( (-Math.pow(dist_[i][j], 2)) / ((2 * Math.pow(sigmas[i], 2))) );
                adj[i][j] = Math.exp( (-1* Math.pow(dist_[i][j], 2)) / ((sigmas[i] * sigmas[j])) );
                //adj[j][i] = Math.exp( (-1 * Math.pow(dist_[i][j], 2)) / ((sigmas[i] * sigmas[j])) );
            }
        }

        return adj;
    }
    public static ArrayList<EdgeList> calulateEdgeList(Double[][] adj_mat) {
        ArrayList<EdgeList> edgeList = new ArrayList<>();

        for (int i = 0; i < adj_mat.length; i++) {
            for (int j = i + 1; j < adj_mat[i].length; j++) {
                edgeList.add(new EdgeList(i, j, adj_mat[i][j],i));
            }
        }
            return edgeList;
    }
    public static ArrayList<Nodelist> getNodeList(ArrayList<TestData> TestDataArrayList) {
        ArrayList<Nodelist> nodeList = new ArrayList<>();
        for (int i = 0; i < TestDataArrayList.size(); i++) {
            nodeList.add(new Nodelist(TestDataArrayList.get(i).getPoints(),i,TestDataArrayList.get(i).getX_coordinate(),TestDataArrayList.get(i).getY_coordinate(),TestDataArrayList.get(i).getClass_type()));
        }
        return nodeList;
    }
    public static Double [][] calculateEpsilonNeighbourhoodGraph (Double[][] dist_,Double epsilon){
        Double[][] adj = new Double[5][5];
        for (int i = 0; i < dist_.length; i++) {
            for (int j = 0; j < dist_[i].length; j++) {
                if (i == j) {
                    adj[i][j] = 0.0;
                    continue;
                }
                if (dist_[i][j] <= epsilon){
                    adj[i][j] = 1.0;
                }
                else {
                    adj[i][j] = 0.0;
                }
            }
        }
        return adj;
    }
    public static Double[][] calculateKNNGraph(Double[][] dist_,Double [] knn){

        Double[][] adj = new Double[dist_.length][dist_[0].length];

        for (int i = 0; i < dist_.length; i++) {
            for (int j = 0; j < dist_[i].length; j++) {
                if (i == j) {
                    adj[i][j] = 0.0;
                    continue;
                }
                if (dist_[i][j] == knn[i]){
                    adj[i][j] = 1.0;
                }
                else {
                    adj[i][j] = 0.0;
                }
            }
        }

        return adj;
    }



    public static void main(String[] args) throws IOException {

        String dataPath = "D:/de/MASTER_THESIS/Decision-Tree-Neo4j/Java Plugin/DecisionTreePlugin/src/main/resources/test.csv";
        ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
        ArrayList<TestData> testData = readCsvTestData.readCsvFile(dataPath);
        Double[][] DistanceMatrix = readCsvTestData.euclidianDistance(testData);
        Double[] sigmas = calculateLocalSigmas(DistanceMatrix);
        Double[][] adj_mat = calculateAdjacencyMatrix(DistanceMatrix,sigmas);
        ArrayList<EdgeList> edgeLists = calulateEdgeList(adj_mat);
        Double [][] EpsilonMatrix = calculateEpsilonNeighbourhoodGraph(DistanceMatrix,2.0);
        Double [] knn = calculateKNN(DistanceMatrix);
        Double [][] knnMatrix = calculateKNNGraph(DistanceMatrix,knn);
        getNodeList(testData);
        //calulateEdgeList(adj_mat);

    }

}
