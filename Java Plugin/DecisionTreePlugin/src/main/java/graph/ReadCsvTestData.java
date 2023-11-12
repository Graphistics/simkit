package graph;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

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

    public ArrayList<ArrayList<String>> readCsvFileNew(String dataPath,Boolean indexColumn) throws IOException {
        Reader in = new FileReader(dataPath);
        ArrayList<String> header = readCSVHeader(dataPath);
        CSVFormat csvFormat = CSVFormat.DEFAULT.builder().setHeader(header.toString()).setSkipHeaderRecord(true).build();
        Iterable<CSVRecord> records = csvFormat.parse(in);

        ArrayList<ArrayList<String>>  csvFilerow =  new ArrayList<ArrayList<String>>();

        // read csv file without header
        for (CSVRecord record : records) {
            ArrayList<String> TestDataArrayList = new ArrayList<>();
            for(int i = 0; i < header.size(); i++){
                if(indexColumn && i==0){
                    continue;
                }
                if(record.get(i).matches(".*[a-zA-Z].*")){
                    continue;
                }
                TestDataArrayList.add(record.get(i));
            }
            csvFilerow.add(TestDataArrayList);
        }

        return csvFilerow;
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
    public static Double[][] euclidianDistance(ArrayList<ArrayList<String>> TestDataArrayList){


        double[][] doubleList = convertToDoubleArray(TestDataArrayList);

        Double[][] distanceMatrix = new Double[doubleList.length][doubleList.length];

        for (int i = 0; i < doubleList.length; i++) {
            for (int j = 0; j < doubleList.length; j++) {
                if (i == j) {
                    distanceMatrix[i][j] = 0.0; // Distance to itself is 0
                } else {
                    distanceMatrix[i][j] = euclideanDistance(doubleList[i], doubleList[j]);
                }
            }
            }

        return distanceMatrix;

    }

    private static double euclideanDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }
    public static double[][] convertToDoubleArray(ArrayList<ArrayList<String>> arrayList) {
        int numRows = arrayList.size();
        int numCols = arrayList.get(0).size(); // Assuming all inner lists have the same size

        double[][] doubleArray = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            ArrayList<String> row = arrayList.get(i);
            for (int j = 0; j < numCols; j++) {
                doubleArray[i][j] = Double.parseDouble(row.get(j));
            }
        }

        return doubleArray;
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
        Double[] sigmas = new Double[pdist.length];

        for (int i = 0; i < pdist.length; i++) {
            Double[] sortedDistances = Arrays.copyOf(pdist[i], pdist[i].length);
            Arrays.sort(sortedDistances, Collections.reverseOrder());
            sigmas[i] = sortedDistances[2];
        }
        return sigmas;
    }
    public static Double[][] calculateAdjacencyMatrix(Double[][] dist_, Double[] sigmas) {
        Double[][] adj = new Double[dist_.length][dist_.length];

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
    public static ArrayList<String> getNodeList(ArrayList<ArrayList<String>> TestDataArrayList) {
        ArrayList<String> nodeList = new ArrayList<>();
        // add all the nodes to the nodeList all entries
        for (int i = 0; i < TestDataArrayList.size(); i++) {
            nodeList.add(i + "");
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
        String dataPath1 = "D:/de/MASTER_THESIS/SimKit/simkit/dataset_1_iris_numeric/Iris_Mythica_Data_Set.csv";

        ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
        ArrayList<ArrayList<String>> arrayLists = readCsvTestData.readCsvFileNew(dataPath,true);
        Double[][] dist_array = euclidianDistance(arrayLists);
        Double[] sigmas = calculateLocalSigmas(dist_array);
        calculateAdjacencyMatrix(dist_array,sigmas);






    }

}
