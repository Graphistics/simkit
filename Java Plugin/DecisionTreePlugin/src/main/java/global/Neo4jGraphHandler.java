package global;

import static org.neo4j.driver.Values.parameters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import graph.EdgeList;
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

import definition.EdgeList2;
import definition.NodeList2;

public class Neo4jGraphHandler {

    public static ArrayList<EdgeList2> retrieveEdgeListFromNeo4j(final String nodeType, Driver driver) {
        ArrayList<EdgeList2> edgeList = new ArrayList<>();

        try (Session session = driver.session()) {
            String cypherQuery = "MATCH (n:" + nodeType + ")-[r]->(m:" + nodeType + ") RETURN n, m, r, n.id AS source, m.id AS target, r.weight AS weight, id(r) AS index";
            Result result = session.run(cypherQuery);

            while (result.hasNext()) {
                Record record = result.next();
//              Node sourceNode = record.get("n").asNode();
                Relationship relationship = record.get("r").asRelationship();
//              Node targetNode = record.get("m").asNode();

                String source = record.get("source").asString();
                String target = record.get("target").asString();
                double weight = record.get("weight").asDouble();  // Correctly extract the weight property
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


    public static ArrayList<NodeList2> retrieveNodeListFromNeo4j(final String nodeType, Driver driver) {
        ArrayList<NodeList2> nodeList = new ArrayList<>();

        try (Session session = driver.session()) {
            String cypherQuery = "MATCH (n:" + nodeType + ") RETURN n, n.id AS index";
            Result result = session.run(cypherQuery);
            String index = "";
            int count = 0;

            while (result.hasNext()) {
                Record record = result.next();
                Node node = record.get("n").asNode();

                index = String.valueOf(count);
                count++;

                Map<String, Object> nodeProperties = extractPropertiesFromNode(node);

                NodeList2 nodeObject = new NodeList2(index, nodeProperties);
                nodeList.add(nodeObject);
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error retrieving node data from Neo4j for label: " + nodeType + ", Error: " + e.getMessage());
        }
        return nodeList;
    }


    public static void createNodeGraph(String graphType, String message, NodeList2 nodeDetail, Driver driver) {
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

    public static void createRelationshipGraph(String graphType, String message, EdgeList2 edgeListDetail, Driver driver) {
        final String source = String.valueOf(edgeListDetail.getSource());
        final String target = String.valueOf(edgeListDetail.getTarget());
        double weightValue = (double) Math.round(edgeListDetail.getWeight() * 100000d) / 100000d;

        try (Session session = driver.session()) {
            session.writeTransaction(new TransactionWork<Void>() {
                @Override
                public Void execute(Transaction tx) {
                    Result result = tx.run(
                            "MATCH (n:" + graphType + " {id: $source}), (m:" + graphType + " {id: $target}) " +
                                    "CREATE (n)-[r:`link` {weight: $weightValue}]->(m)",
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


}
