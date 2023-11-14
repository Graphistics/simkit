package graph;

import static org.neo4j.driver.Values.parameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.neo4j.driver.Driver;
import org.neo4j.driver.Record;
import org.neo4j.driver.Result;
import org.neo4j.driver.Session;
import org.neo4j.driver.Transaction;
import org.neo4j.driver.TransactionWork;
import org.neo4j.driver.Value;
import org.neo4j.driver.exceptions.Neo4jException;
import org.neo4j.driver.types.Node;
import org.neo4j.driver.types.Relationship;

public class Neo4jGraphHandler {
    private static Driver driver;

    public static ArrayList<EdgeList2> retrieveEdgeListFromNeo4j(final String nodeType) {
        ArrayList<EdgeList2> edgeList = new ArrayList<>();

        try (Session session = driver.session()) {
            String cypherQuery = "MATCH (n:" + nodeType + ")-[r]->(m:" + nodeType + ") RETURN n, m, r, n.id AS source, m.id AS target, toFloat(type(r)) AS weight, id(r) AS index";
            Result result = session.run(cypherQuery);

            while (result.hasNext()) {
                Record record = result.next();
//                Node sourceNode = record.get("n").asNode();
                Relationship relationship = record.get("r").asRelationship();
//                Node targetNode = record.get("m").asNode();

                String source = record.get("source").asString();
                String target = record.get("target").asString();
                double weight = record.get("weight").asDouble();
                long index = record.get("index").asLong();

                Map<String, Object> relationshipProperties = extractPropertiesFromRelationship(relationship);

                EdgeList2 edge = new EdgeList2(source, target, weight, index, relationshipProperties);
                edgeList.add(edge);
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error retrieving edge data from Neo4j: " + e.getMessage());
        }
        return edgeList;
    }

    public static ArrayList<NodeList2> retrieveNodeListFromNeo4j(final String nodeType) {
        ArrayList<NodeList2> nodeList = new ArrayList<>();

        try (Session session = driver.session()) {
            String cypherQuery = "MATCH (n:" + nodeType + ") RETURN n, n.id AS index";
            Result result = session.run(cypherQuery);

            while (result.hasNext()) {
                Record record = result.next();
                Node node = record.get("n").asNode();

                String index = record.get("index").asString();

                Map<String, Object> nodeProperties = extractPropertiesFromNode(node);

                NodeList2 nodeObject = new NodeList2(index, nodeProperties);
                nodeList.add(nodeObject);
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error retrieving node data from Neo4j for label: " + nodeType + ", Error: " + e.getMessage());
        }
        return nodeList;
    }

    
    public static void createNodeGraph(String graphType, String message, NodeList2 nodeDetail) {
        final String id = nodeDetail.getIndex();
        final Map<String, Object> properties = nodeDetail.getProperties();

        try (Session session = driver.session()) {
            session.writeTransaction(new TransactionWork<String>() {
                @Override
                public String execute(Transaction tx) {
                    String cypherQuery = "CREATE (:" + graphType + " {id: $id";
                    
                    for (Map.Entry<String, Object> entry : properties.entrySet()) {
                        cypherQuery += ", " + entry.getKey() + ": $" + entry.getKey();
                    }
                    
                    cypherQuery += "})";

                    Map<String, Object> parameters = new HashMap<>();
                    parameters.put("id", id);
                    for (Map.Entry<String, Object> entry : properties.entrySet()) {
                        parameters.put(entry.getKey(), entry.getValue());
                    }

                    Result result = tx.run(cypherQuery, parameters);
                    return message;
                }
            });
        }
    }

    
    public static void createRelationshipGraph(String graphType, String message, EdgeList2 edgeListDetail) {
        final String source = edgeListDetail.getSource();
        final String target = edgeListDetail.getTarget();
        double weightValue = (double) Math.round(edgeListDetail.getWeight() * 100000d) / 100000d;

        try (Session session = driver.session()) {
            session.writeTransaction(new TransactionWork<Void>() {
                @Override
                public Void execute(Transaction tx) {
                    Result result = tx.run(
                            "MATCH (n:" + graphType + " {id: $source}), (m:" + graphType + " {id: $target}) " +
                                    "CREATE (n)-[r:`" + weightValue + "` {weight: $weightValue}]->(m)",
                            parameters("source", source, "target", target, "weightValue", weightValue)
                    );
                    return null;
                }
            });
        } catch (Neo4jException e) {
            throw new RuntimeException("Error creating relationship in Neo4j: " + e.getMessage());
        }
    }


    private static Map<String, Object> extractPropertiesFromNode(Node node) {
        Map<String, Object> properties = new HashMap<>();

        for (String key : node.keys()) {
            Value value = node.get(key);
            properties.put(key, convertToJavaType(value));
        }

        return properties;
    }

    private static Object convertToJavaType(Value value) {
        if (value.type().name().equals("String")) {
            return value.asString();
        } else if (value.type().name().equals("Integer")) {
            return value.asLong();
        } else if (value.type().name().equals("Float")) {
            return value.asDouble();
        } else if (value.type().name().equals("Boolean")) {
            return value.asBoolean();
        } else {
            return value.asObject();
        }
    }


    private static Map<String, Object> extractPropertiesFromRelationship(Relationship relationship) {
        Map<String, Object> properties = new HashMap<>();

        for (String key : relationship.keys()) {
            Value value = relationship.get(key);
            properties.put(key, convertToJavaType(value));
        }

        return properties;
    }

    
    public static Double[][] euclideanDistance(List<NodeList2> nodeList) {

    	int size = nodeList.size();
        Double[][] distanceMatrix = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double distance = calculateEuclideanDistance(nodeList.get(i), nodeList.get(j));
                distanceMatrix[i][j] = distance;
            }
        }
        return distanceMatrix;
    }

    private static double calculateEuclideanDistance(NodeList2 node1, NodeList2 node2) {
        Map<String, Object> properties1 = node1.getProperties();
        Map<String, Object> properties2 = node2.getProperties();

        double sum = 0.0;

        for (String propertyKey : properties1.keySet()) {
            Object value1 = properties1.get(propertyKey);
            Object value2 = properties2.get(propertyKey);

            if (value1 instanceof Number && value2 instanceof Number) {
                double diff = ((Number) value1).doubleValue() - ((Number) value2).doubleValue();
                sum += Math.pow(diff, 2);
            }
        }
        return Math.sqrt(sum);
    }


    public static Double[] calculateKNN(Double[][] pdist) {
        int size = pdist.length;
        Double[] sigmas = new Double[size];

        for (int i = 0; i < size; i++) {
            Double[] sortedDistances = Arrays.copyOf(pdist[i], pdist[i].length);
            Arrays.sort(sortedDistances);
            sigmas[i] = sortedDistances[1];
        }

        return sigmas;
    }

    public static Double[] calculateLocalSigmas(Double[][] pdist) {
        int size = pdist.length;
        Double[] sigmas = new Double[size];

        for (int i = 0; i < size; i++) {
            Double[] sortedDistances = Arrays.copyOf(pdist[i], pdist[i].length);
            Arrays.sort(sortedDistances, Collections.reverseOrder());
            sigmas[i] = sortedDistances[2];
        }

        return sigmas;
    }

    public static Double[][] calculateAdjacencyMatrix(Double[][] dist_, Double[] sigmas) {
        int size = dist_.length;
        Double[][] adj = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    adj[i][j] = 0.0;
                    continue;
                }
                adj[i][j] = Math.exp((-1 * Math.pow(dist_[i][j], 2)) / ((sigmas[i] * sigmas[j])));
            }
        }

        return adj;
    }

    public static Double[][] calculateEpsilonNeighbourhoodGraph(Double[][] dist_, Double epsilon) {
        int size = dist_.length;
        Double[][] adj = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    adj[i][j] = 0.0;
                    continue;
                }
                adj[i][j] = (dist_[i][j] <= epsilon) ? 1.0 : 0.0;
            }
        }
        return adj;
    }

    public static Double[][] calculateKNNGraph(Double[][] dist_, Double[] knn) {
        int size = dist_.length;
        Double[][] adj = new Double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    adj[i][j] = 0.0;
                    continue;
                }
                adj[i][j] = (dist_[i][j] == knn[i]) ? 1.0 : 0.0;
            }
        }

        return adj;
    }
    
    public void close() {
        if (driver != null) {
            driver.close();
        }
    }
    
    
    public static ArrayList<EdgeList2> calculateEdgeList(List<NodeList2> nodePropertiesList, Double[][] adj_mat) {
        ArrayList<EdgeList2> edgeList = new ArrayList<>();

        for (int i = 0; i < adj_mat.length; i++) {
            for (int j = i + 1; j < adj_mat[i].length; j++) {
                String sourceId = nodePropertiesList.get(i).getIndex();
                String targetId = nodePropertiesList.get(j).getIndex(); 
                edgeList.add(new EdgeList2(sourceId, targetId, adj_mat[i][j], i, null));
            }
        }
        return edgeList;
    }

    
}
