package graph;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class ReadCSvIris {

     String[] HEADERS = { "Index","sepal.length","sepal.width","petal.length","petal.width","class"};

    public ArrayList<IrisData> readCsvFile() throws IOException {
        ArrayList<IrisData> irisDataArrayList = new ArrayList<>();
        Reader in = new FileReader("src/main/resources/Iris _Mythica_Data_Set.csv");

        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader(HEADERS)
                .setSkipHeaderRecord(true)
                .build();


        Iterable<CSVRecord> records = csvFormat.parse(in);

        for (CSVRecord record : records) {
            String Index = record.get("Index");
            String SepelLength = record.get("sepal.length");
            String SepelWidth = record.get("sepal.width");
            String PetalLength = record.get("petal.length");
            String PetalWidth = record.get("petal.width");
            String Class = record.get("class");
            IrisData irisData = new IrisData(Integer.parseInt(Index),Double.parseDouble(SepelLength),Double.parseDouble(SepelWidth),Double.parseDouble(PetalLength),Double.parseDouble(PetalWidth),Class);
            irisDataArrayList.add(irisData);
        }
        return irisDataArrayList;
    }
    public  Double[][] euclidianDistance(ArrayList<IrisData> irisDataArrayList){
       Double[][] distanceMatrix = new Double[irisDataArrayList.size()][irisDataArrayList.size()];
        for (int i = 0; i < irisDataArrayList.size(); i++) {
            for (int j = 0; j < irisDataArrayList.size(); j++) {
                double distance = Math.sqrt(Math.pow(irisDataArrayList.get(i).getSepalLength()-irisDataArrayList.get(j).getSepalLength(),2)+Math.pow(irisDataArrayList.get(i).getSepalWidth()-irisDataArrayList.get(j).getSepalWidth(),2)+Math.pow(irisDataArrayList.get(i).getPetalLength()-irisDataArrayList.get(j).getPetalLength(),2)+Math.pow(irisDataArrayList.get(i).getPetalWidth()-irisDataArrayList.get(j).getPetalWidth(),2));
                distanceMatrix[i][j] = distance;
                System.out.println(distance);
            }
        }
        return distanceMatrix;
    }
    public static Double[] calculateLocalSigmas(Double[][] pdist) {
        Double[] sigmas = new Double[200];

        for (int i = 0; i < pdist.length; i++) {
            Double[] sortedDistances = Arrays.copyOf(pdist[i], pdist[i].length);
            Arrays.sort(sortedDistances, Collections.reverseOrder());
            sigmas[i] = sortedDistances[2];
        }

        return sigmas;
    }
    public static Double[][] calculateAdjacencyMatrix(Double[][] dist_, Double[] sigmas) {
        Double[][] adj = new Double[200][200];

        for (int i = 0; i < dist_.length; i++) {
            for (int j = 0; j < dist_[i].length; j++) {
//                adj[i][j] = Math.exp(-Math.pow(dist_[i][j], 2) / (2 * Math.pow(sigmas[i], 2)));
                adj[i][j] = Math.exp( (-1* Math.pow(dist_[i][j], 2)) / ((sigmas[i] * sigmas[j])) );
//                adj[j][i] = Math.exp( (-1 * Math.pow(dist_[i][j], 2)) / ((sigmas[i] * sigmas[j])) );
            }
        }

        return adj;
    }


public static void main(String[] args) throws IOException {
//    Path path = Path.of("src/main/resources/Iris _Mythica_Data_Set.csv");
    ReadCSvIris readCSv = new ReadCSvIris();
    ArrayList<IrisData> irisData = readCSv.readCsvFile();
    Double[][] pdist = readCSv.euclidianDistance(irisData);
    Double[] sigmas = calculateLocalSigmas(pdist);
    Double[][] adj = calculateAdjacencyMatrix(pdist,sigmas);




    }

}
