package global;

import static org.neo4j.driver.Values.parameters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math4.legacy.linear.RealMatrix;
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
	
    /**
     * Retrieves the edge list from Neo4j for a specified node type.
     *
     * @param node_label The label of the nodes in the graph.
     * @param driver   The Neo4j Driver instance.
     * @return ArrayList of EdgeList2 representing the edges in the graph.
     */
    public static ArrayList<EdgeList2> retrieveEdgeListFromNeo4j(final String node_label, Driver driver) {
        ArrayList<EdgeList2> edge_list = new ArrayList<>();

        try (Session session = driver.session()) {
        	String cypher_query = "MATCH (n:" + node_label + ")-[r]->(m:" + node_label + ") WHERE r.value IS NOT NULL RETURN n, m, r, toString(n.id) AS source, toString(m.id) AS target, r.value AS weight, id(r) AS index";
            Result result = session.run(cypher_query);

            while (result.hasNext()) {
                Record record = result.next();
//              Node sourceNode = record.get("n").asNode();
                Relationship relationship = record.get("r").asRelationship();
//              Node targetNode = record.get("m").asNode();

                String source = record.get("source").asString();
                String target = record.get("target").asString();
                double weight = record.get("weight").asDouble(); 
                long index = record.get("index").asLong();

                Map<String, Object> relationship_properties = extractPropertiesFromRelationship(relationship);

                EdgeList2 edge = new EdgeList2(source, target, weight, index, relationship_properties);
                edge_list.add(edge);
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error retrieving edge data from Neo4j: " + e.getMessage());
        }
        return edge_list;
    }
    
    // correct - generate and export csv
    public static void exportCSVFile(final String node_label, final String csvFileName, Driver driver) {
        try (Session session = driver.session()) {
            String cypher_query = "CALL apoc.export.csv.query("
                + "'MATCH (n:" + node_label + ")-[r]->(m:" + node_label + ") "
                + "WHERE r.value IS NOT NULL "
                + "RETURN toString(n.id) AS source, toString(m.id) AS target, r.value AS weight', "
                + "'" + csvFileName + "', "
                + "{}"
                + ")";
            session.run(cypher_query);
        } catch (Neo4jException e) {
            throw new RuntimeException("Error exporting CSV from Neo4j: " + e.getMessage());
        }
    }


    
    public static Pair<ArrayList<NodeList2>, String> retrieveNodeListFromNeo4jSimilarityGraph(final String nodeType, Driver driver) {
        ArrayList<NodeList2> nodeList = new ArrayList<>();
        String propertyKeys = "";

        try (Session session = driver.session()) {
            String cypherQuery = "MATCH (n:" + nodeType + ") RETURN n, n.id AS index";
            Result result = session.run(cypherQuery);
            String index;
            int count = 0;


            while (result.hasNext()) {
                Record record = result.next();
                Node node = record.get("n").asNode();
                index = String.valueOf(count);
                Map<String, Object> nodeProperties = extractPropertiesFromNode(node);
                NodeList2 nodeObject = new NodeList2(index, nodeProperties);
                nodeList.add(nodeObject);
                
                if (count == 0) {
                    propertyKeys = String.join(",", node.keys());
                }
                
                count++;
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error retrieving node data from Neo4j for label: " + nodeType + ", Error: " + e.getMessage());
        }
        return Pair.of(nodeList, propertyKeys);
    }

    public static ArrayList<NodeList2> retrieveNodeListFromNeo4jSimilarityGraph2(final String nodeType, Driver driver) {
        ArrayList<NodeList2> nodeList = new ArrayList<>();

        try (Session session = driver.session()) {
            String cypherQuery = "MATCH (n:" + nodeType + ") RETURN n, n.id AS index";
            Result result = session.run(cypherQuery);
            String index;
            int count = 0;


            while (result.hasNext()) {
                Record record = result.next();
                Node node = record.get("n").asNode();
                index = String.valueOf(count);
                Map<String, Object> nodeProperties = extractPropertiesFromNode(node);
                NodeList2 nodeObject = new NodeList2(index, nodeProperties);
                nodeList.add(nodeObject);
                count++;
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error retrieving node data from Neo4j for label: " + nodeType + ", Error: " + e.getMessage());
        }
        return nodeList;
    }
    
    

    /**
     * Retrieves the node list from Neo4j for a specified node type.
     *
     * @param node_label The label of the nodes in the graph.
     * @param driver   The Neo4j Driver instance.
     * @return ArrayList of NodeList2 representing the nodes in the graph.
     */
    public static ArrayList<NodeList2> retrieveNodeListFromNeo4j(final String node_label, Driver driver) {
        ArrayList<NodeList2> nodeList = new ArrayList<>();

        try (Session session = driver.session()) {
            String cypher_query = "MATCH (n:" + node_label + ") RETURN n, toString(n.id) AS index";
            Result result = session.run(cypher_query);

            while (result.hasNext()) {
                Record record = result.next();
                Node node = record.get("n").asNode();

                String index = record.get("index").asString();

                Map<String, Object> node_properties = extractPropertiesFromNode(node);

                NodeList2 node_object = new NodeList2(index, node_properties);
                nodeList.add(node_object);
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error retrieving node data from Neo4j for label: " + node_label + ", Error: " + e.getMessage());
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

    /**
     * Creates nodes in Neo4j for the transformed graph after Laplacian Eigen Transform.
     *
     * @param graph_type   The type of the transformed graph.
     * @param message     A message indicating the operation.
     * @param nodeDetail  Details of the node to be created.
     * @param X           The X matrix obtained from eigen decomposition.
     * @param driver      The Neo4j Driver instance.
     */
    public static void createNodeGraphEigenTransform(String graph_type, String message, NodeList2 nodeDetail, RealMatrix X, Driver driver) {
        final String id = nodeDetail.getIndex();
        final Map<String, Object> properties = nodeDetail.getProperties();

        for (int i = 0; i < X.getColumnDimension(); i++) {
            properties.put("eigenvector_" + i, X.getEntry(Integer.parseInt(id), i));
        }

        try (Session session = driver.session()) {
            session.writeTransaction(new TransactionWork<String>() {
                @Override
                public String execute(Transaction tx) {
                    String cypher_query = "CREATE (:" + graph_type + " {id: $id";
                    
                    for (Map.Entry<String, Object> entry : properties.entrySet()) {
                        cypher_query += ", " + entry.getKey() + ": $" + entry.getKey();
                    }
                    
                    cypher_query += "})";

                    Map<String, Object> parameters = new HashMap<>();
                    parameters.put("id", id);
                    for (Map.Entry<String, Object> entry : properties.entrySet()) {
                        parameters.put(entry.getKey(), entry.getValue());
                    }

                    Result result = tx.run(cypher_query, parameters);
                    return message;
                }
            });
        }
    }
    
    /**
     * Creates relationships in Neo4j for the transformed graph after Laplacian Eigen Transform.
     *
     * @param graph_type        The type of the transformed graph.
     * @param message          A message indicating the operation.
     * @param edge_list_detail   Details of the edge to be created.
     * @param driver           The Neo4j Driver instance.
     */
    public static void createRelationshipGraph(String graph_type, String message, EdgeList2 edge_list_detail, Driver driver) {
        final String source = edge_list_detail.getSource();
        final String target = edge_list_detail.getTarget();
//        double weightValue = (double) Math.round(edge_list_detail.getWeight() * 10000000d) / 10000000d;
        double weightValue = edge_list_detail.getWeight();
        
        try (Session session = driver.session()) {
            session.writeTransaction(new TransactionWork<Void>() {
                @Override
                public Void execute(Transaction tx) {
                    String deleteQuery = "MATCH (n:" + graph_type + " {id: $source})-[r:`link`]->(m:" + graph_type + " {id: $target}) " + "DELETE r";
                    tx.run(deleteQuery, parameters("source", source, "target", target));

                    Result result = tx.run(
                            "MATCH (n:" + graph_type + " {id: $source}), (m:" + graph_type + " {id: $target}) " +
                                    "CREATE (n)-[r:`link` {value: $weightValue}]->(m)",
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
