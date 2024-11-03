package simkit;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Record;
import org.neo4j.driver.Result;
import org.neo4j.driver.Session;
import org.neo4j.driver.Transaction;
import org.neo4j.driver.TransactionWork;
import org.neo4j.driver.Value;
import org.neo4j.driver.exceptions.AuthenticationException;
import org.neo4j.driver.exceptions.Neo4jException;
import org.neo4j.driver.exceptions.ServiceUnavailableException;
import org.neo4j.driver.util.Pair;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import definition.EdgeList2;
import definition.NodeList2;
import eigendecomposed.EigenCalculation;
import eigendecomposed.MatrixCalculation;
import global.Neo4jGraphHandler;
import graph.DistanceMeasureNodes;
import graph.GraphTransform;
import graph.ReadCsvTestData;
import simkit.SimKitProcedures;

import org.neo4j.logging.Log;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Procedure;

/**
 *
 * This class is used to fetch nodes from graph database or from csv and call the functions to generate decision tree
 * with confusion matrix, generation time and prediction time for the output
 *
 * @author minh dung
 *
 */

public class SimKitProcedures implements AutoCloseable{

	private static Driver driver;
	private static List<Record> dataKey = new ArrayList<>();
	private static ArrayList<String> mapNodeList =  new ArrayList<String>();
	private static List<Double> trueNodeLabels =  new ArrayList<Double>();
	private static List<Double> predictedNodeLabels =  new ArrayList<Double>();

	private static String uri;
    private static String username;
    private static String password;

	@Context
    public Log log;

	/**
	 * Creation of driver object using bolt protocol
	 * @param uri Uniform resource identifier for bolto
	 * @param user Username
	 * @param password Password
	 */
	public SimKitProcedures( String uri, String user, String password )
	{
		driver = GraphDatabase.driver( uri, AuthTokens.basic( user, password ) );

	}

	/**
	 * Empty constructor
	 */
	public SimKitProcedures()
	{
		driver = null;
	}
	
	/**
	 * This function is used to initialize the SimKit
	 * @param uri The URI of DB
	 * @param user The Username for DB
	 * @param pass The Password for DB
	 * @throws Exception if connection to Neo4j fails
	 * @author Jonas Heinzmann
	 */
	@UserFunction
	public String initSimKit(@Name("URI") String uri, @Name("User") String user, @Name("Password") String pass) {
        String output;
        try {
            driver = GraphDatabase.driver(uri, AuthTokens.basic(user, pass));
            // Attempt to establish a connection to verify credentials and availability
            driver.verifyConnectivity();
            output = "Initialization succesfull";
            // Store credentials if needed
            SimKitProcedures.uri = uri;
            SimKitProcedures.username = user;
            SimKitProcedures.password = pass;
        } catch (ServiceUnavailableException e) {
            output = "Connection error: Service unavailable";
        } catch (AuthenticationException e) {
            output = "Connection error: Authentication failed";
        } catch (Neo4jException e) {
            output = "Connection error: " + e.getMessage();
        } finally {
            if (driver != null) {
                driver.close();
            }
        }
        return output;
    }

    public Driver getDriver() {
        return driver;
    }

	/**
	 * Close the driver object
	 */
	@Override
	public void close() throws Exception
	{
		driver.close();
	}

	@UserFunction
	public String createGraphFromCsv(@Name("data_path") String data_path, @Name("distance_measure") String distance_measure, @Name("graph_type") String graph_type, @Name("parameter") String parameter,@Name("remove_column") String remove_columns) throws Exception {


		String confusionMatrix = "";
		try ( SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password) )
		{

			if(data_path == null && distance_measure == null) {
				return "Missing data_path or distance measure type";
			}else {
				String graphName = null;
				Double[][] adj_mat = null;
				String[] removeList = remove_columns.split(",");
				List<String> removeListNew = Arrays.stream(removeList).collect(Collectors.toList());
				ReadCsvTestData readCsvTestData = new ReadCsvTestData(data_path);
//				//ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNew(data_path,IndexColumn);
				ArrayList<NodeList2> nodePropertiesList = readCsvTestData.readCsvFileToMap(data_path);
//                ArrayList<NodeList2> nodePropertiesList_copy = readCsvTestData.readCsvFileToMap(data_path);
				Double[][] DistanceMatrix = getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);

				if(graph_type.equals("full")) {
					Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
					graphName = graph_type.concat("_"+parameter);
				}
				if(graph_type.equals("eps")) {
					Double espilonValue = Double.parseDouble(parameter);
					adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,espilonValue);
					graphName = graph_type.concat("_"+parameter);
				}
				if(graph_type.equals("knn")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateKNNGraph(DistanceMatrix,knn);
					graphName = graph_type.concat("_"+parameter);
				}
				if(graph_type.equals("mknn")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateMutualKNNGraph(DistanceMatrix,knn);
					graphName = graph_type.concat("_"+parameter);
				}
				ArrayList<EdgeList2> edgeList = GraphTransform.calculateEdgeList(nodePropertiesList,adj_mat);

				for (NodeList2 node : nodePropertiesList) {
					Neo4jGraphHandler.createNodeGraph(graphName.concat("new"), "created nodes in neo4j", node, connector.getDriver());
				}

				for (int i = 0; i < edgeList.size(); i++) {
					EdgeList2 edgeListDetail = edgeList.get(i);
					if(edgeListDetail.getWeight()==0.0){
						continue;
					}
					Neo4jGraphHandler.createRelationshipGraph(graphName.concat("new"), "created relationship in neo4j \n", edgeListDetail, connector.getDriver());

				}

			}
			return "Create fully connected graph successful, " + confusionMatrix;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}
	
	@UserFunction
	public String createGraphFromNodes(@Name("label") String label,@Name("distance_measure") String distanceMeasure,@Name("graph_type") String graphType,@Name("parameter") String parameter,@Name("remove_column") String remove_columns) throws Exception {

		String confusionMatrix = "";
		try ( SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password) )
		{


			if(label == null && distanceMeasure == null) {
				return "Missing dataPath or distance measure type";
			}else {
				String graphName = null;
				//ArrayList<NodeList2> nodePropertiesList = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(label, connector.getDriver());
				org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodeData = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(label, connector.getDriver());
				ArrayList<NodeList2> nodePropertiesList = nodeData.getLeft();
				//				ArrayList<NodeList2> nodePropertiesList_copy = Neo4jGraphHandler.retrieveNodeListFromNeo4j(label, connector.getDriver());
				String[] removeList = remove_columns.split(",");
				List<String> removeListNew = Arrays.stream(removeList).collect(Collectors.toList());
//				Double[][] DistanceMatrix = GraphTransform.euclideanDistance(nodePropertiesList);
				Double[][] DistanceMatrix = getDistanceMatrixFromNodes(distanceMeasure,nodePropertiesList,removeListNew);
				Double[][] adj_mat = null;

				if(graphType.equals("full")) {
					Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
					graphName = graphType.concat("_"+parameter);
				}
				if(graphType.equals("eps")) {
					Double espilonValue = Double.parseDouble(parameter);
					adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,espilonValue);
					graphName = graphType.concat("_"+parameter);

				}
				if(graphType.equals("knn")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateKNNGraph(DistanceMatrix,knn);
					graphName = graphType.concat("_"+parameter);

				}
				if(graphType.equals("mknn")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateMutualKNNGraph(DistanceMatrix,knn);
					graphName = graphType.concat("_"+parameter);
				}
				ArrayList<EdgeList2> edgeList = GraphTransform.calculateEdgeList(nodePropertiesList,adj_mat);



				//for (EdgeList edgeListDetail : edgeList) {
				for (NodeList2 node : nodePropertiesList) {
					Neo4jGraphHandler.createNodeGraph(graphName.concat("new"), "created nodes in neo4j", node, connector.getDriver());
				}

				for (int i = 0; i < edgeList.size(); i++) {
					EdgeList2 edgeListDetail = edgeList.get(i);
					if(edgeListDetail.getWeight()==0.0){
						continue;
					}
					Neo4jGraphHandler.createRelationshipGraph(graphName.concat("new"), "created relationship in neo4j \n", edgeListDetail, connector.getDriver());

				}

			}
			return "Create fully connected graph successful, " + confusionMatrix;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	public static Double[][] getDistanceMatrixFromNodes(String distanceMeasure, ArrayList<NodeList2> nodePropertiesList, List<String> removeList) {
		Double[][] DistanceMatrix = null;

		switch (distanceMeasure) {
			case "euclidean":
				DistanceMatrix = DistanceMeasureNodes.euclideanDistance(nodePropertiesList,removeList);
				break;
			case "manhattan":
				DistanceMatrix = DistanceMeasureNodes.manhattanDistance(nodePropertiesList,removeList);
				break;
			case "canberra":
				DistanceMatrix = DistanceMeasureNodes.canberraDistance(nodePropertiesList,removeList);
				break;
			case "cosine":
				DistanceMatrix = DistanceMeasureNodes.cosineSimilarity(nodePropertiesList,removeList);
				break;
			case "jaccard":
				DistanceMatrix = DistanceMeasureNodes.jaccardCoefficient(nodePropertiesList,removeList);
				break;
			case "bray_curtis":
				DistanceMatrix = DistanceMeasureNodes.brayCurtisDistance(nodePropertiesList,removeList);
				break;
			default:
				System.out.println("Invalid distance measure type");
		}

		return DistanceMatrix;

	}

	/**
	 * Creates a Laplacian Eigen Transform graph based on Laplacian matrix type and specified number of eigenvectors.
	 *
	 * @param node_label            The label of the nodes in the graph.
	 * @param laplacian_type        The type of Laplacian matrix to be used.
	 * @param number_of_eigenvectors The number of desired eigenvectors to compute during eigen decomposition.
	 * @return String indicating the success of the graph creation and creates Graph with Nodes and Relationships in Neo4j.
	 * @throws RuntimeException If an error occurs while creating the Laplacian graph in Neo4j.
	 */
	@UserFunction
	public String createEigenGraph(@Name("node_label") String node_label,  @Name("laplacian_type") String laplacian_type, @Name("number_of_eigenvectors") Double number_of_eigenvectors) throws Exception {

		try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password) ) {
	        if (node_label == null) {
	            return "Missing node label";
	        } else {
	            ArrayList<EdgeList2> edge_list = new ArrayList<>();
	            edge_list = Neo4jGraphHandler.retrieveEdgeListFromNeo4j(node_label, connector.getDriver());

	            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list);
	            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
	            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);

	            EigenCalculation.EigenResult eigen_result = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);
	            ArrayList<NodeList2> node_list_eigen = Neo4jGraphHandler.retrieveNodeListFromNeo4j(node_label, connector.getDriver());
	            ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_list_eigen, eigen_result.X, edge_list);

	            
	            String graph_name = "eigenGraph_" + laplacian_type + "_" + node_label + "_" + Math.round(number_of_eigenvectors);
	            
	            for (NodeList2 node : node_list_eigen) {
	            	Neo4jGraphHandler.createNodeGraphEigenTransform(graph_name, "created nodes in neo4j", node, eigen_result.X, connector.getDriver());
	            }
	            for (int i = 0; i < edge_list_eigen.size(); i++) {
					EdgeList2 edge_list_detail = edge_list_eigen.get(i);
	            	Neo4jGraphHandler.createRelationshipGraph(graph_name, "created relationship in neo4j \n", edge_list_detail, connector.getDriver());
	            }
	        }
	        return "Create eigendecomposed graph successful!";
	    } catch (Neo4jException e) {
	        throw new RuntimeException("Error creating laplacian graph in Neo4j: " + e.getMessage());
	    }
	}
	
	
	@UserFunction
	public String mapNodes(@Name("nodeSet") String nodeSet, @Name("overlook") String overLook) throws Exception {
	    String listOfData = "";
	    String[] overLookArray = new String[0];
	    mapNodeList.clear();
	    try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
	        if (!overLook.isEmpty()) {
	            overLookArray = overLook.split(",");
	        }
	        queryData(nodeSet);
	        for (Record key : dataKey) {
	            List<Pair<String, Value>> values = key.fields();
	            for (Pair<String, Value> nodeValues : values) {
	                if ("n".equals(nodeValues.key())) {
	                    Value value = nodeValues.value();
	                    String valueOfNode = getNodeValues(value, overLookArray);
	                    mapNodeList.add(valueOfNode);
//	                    listOfData = listOfData + valueOfNode + " | ";
	                    listOfData = mapNodeList.toString();
	                }
	            }
	        }
	    }
	    return "Map all node data: " + listOfData;
	}
	
	 /**
 * Procedure for k-means clustering and visualization in neo4j
 * @param params Type of node
 * @return Cluster result and visualization
 * @throws Exception
 */
@UserFunction
@Description("K-means clustering function")
public String kmean(@Name("params") Map<String, Object> params) throws Exception {

    predictedNodeLabels.clear();

	String nodeSet = (String) params.getOrDefault("nodeSet", "eigenGraph_sym_full_7new_3");
	String numberOfCentroid = (String) params.getOrDefault("numberOfCentroid", "3");
	String numberOfInteration = (String) params.getOrDefault("numberOfInteration", "100");
	String distanceMeasure = (String) params.getOrDefault("distanceMeasure", "euclidean");
	String originalNodeSet = (String) params.getOrDefault("originalSet", "Iris");
	String overLook = (String) params.getOrDefault("overlook", "target,sepal_length,sepal_width,petal_length,petal_width");
	String overlookOriginal = (String) params.getOrDefault("overlookOriginal", "target");
	boolean kmeanBool = (Boolean) params.getOrDefault("useKmeanForSilhouette", false);


    try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
        int numCentroids = Integer.parseInt(numberOfCentroid);
        int numIterations = Integer.parseInt(numberOfInteration);
        double centroidNumber = 1.0;

        ArrayList<String> mapNodeList = parseNodeValues(nodeSet, overLook.split(","));
        ArrayList<String> mapNodeOriginalList = parseNodeValues(originalNodeSet, overlookOriginal.split(","));

        HashMap<String, ArrayList<String>> kmeanAssign = Unsupervised.KmeanClust(
            mapNodeList, numCentroids, numIterations, distanceMeasure, false, new ArrayList<>()
        );

        double averageSilhouetteCoefficientValue;
        if (kmeanBool) {
			// Remove index and id from kmeanAssign
			HashMap<String, ArrayList<String>> cleanedKmeanAssign = Unsupervised.removeIndexAndId(kmeanAssign);
            averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(cleanedKmeanAssign, distanceMeasure);
        } else {
            HashMap<String, ArrayList<String>> mappedNodes = Unsupervised.replaceValuesWithOriginalSet(kmeanAssign, mapNodeOriginalList, log);
            averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(mappedNodes, distanceMeasure);
        }

        processClusters(connector, nodeSet, kmeanAssign, centroidNumber, distanceMeasure);

        return "The average Silhouette Coefficient value is: " + averageSilhouetteCoefficientValue +
               " predicted labels: " + predictedNodeLabels;
    }
}

/**
 * Parses node values from the dataset based on overlook fields.
 */
private ArrayList<String> parseNodeValues(String nodeSet, String[] overlookFields) throws Exception {
    ArrayList<String> nodeList = new ArrayList<>();
    queryData(nodeSet);

    for (Record record : dataKey) {
        for (Pair<String, Value> pair : record.fields()) {
            if ("n".equals(pair.key())) {
                String value = getNodeValues(pair.value(), overlookFields);
                nodeList.add(value);
            }
        }
    }
    return nodeList;
}

/**
 * Processes clusters and connects nodes based on the k-means result.
 */
private void processClusters(SimKitProcedures connector, String nodeSet,
                             HashMap<String, ArrayList<String>> kmeanAssign,
                             double centroidNumber, String distanceMeasure) throws Exception {
    DecimalFormat decimalFormat = new DecimalFormat("#.###");

    for (String centroid : kmeanAssign.keySet()) {
        ArrayList<String> clusterNodes = kmeanAssign.get(centroid);

        for (String clusterNode : clusterNodes) {
            predictedNodeLabels.add(centroidNumber);

            double distance = Unsupervised.calculateDistance(clusterNode, centroid, distanceMeasure);
            String formattedDistance = decimalFormat.format(distance);

            connector.connectNodes(nodeSet, "create relationship in kmean node", centroid, clusterNode, Double.parseDouble(formattedDistance));
        }
        centroidNumber += 1.0;
    }
}
    
    @UserFunction
	public String adjustedRandIndex(@Name("nodeSet") String nodeSet, @Name("trueLabels") String trueLabel) throws Exception {
	    if(predictedNodeLabels.size()==0)
	    {
	    	return " predicted Labels is null, please run kmean clustering to add the predicted labels";
	    }
	    else {
	    	String listOfData = "";
	    	Double adjustedRandIndexValue = 0.0;
		    trueNodeLabels.clear();
		    List<String> stringTrueNodeLabelsList = new ArrayList<String>();
		    try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
		        queryData(nodeSet);
		        for (Record key : dataKey) {
		            List<Pair<String, Value>> values = key.fields();
		            for (Pair<String, Value> nodeValues : values) {
		                if ("n".equals(nodeValues.key())) {
		                    Value value = nodeValues.value();
		                    StringBuilder nodeLabel = new StringBuilder();
		                    for (String nodeKey : value.keys()) {
		                    	if(nodeKey.equals(trueLabel))
		                    	{
		                    		try {
		                	            double num = Double.parseDouble(String.valueOf(value.get(nodeKey)));
	                	            	trueNodeLabels.add(num);
	                	            	listOfData = listOfData + num;
	//	                	            	nodeLabel.append(getStringValue(nodeLabel)).append(nodeKey).append(":").append(value.get(nodeKey));
		                	        } catch (NumberFormatException e) {
		                	            System.out.println(value.get(nodeKey) + " is not a number.");
		                	            stringTrueNodeLabelsList.add(String.valueOf(value.get(nodeKey)));
		                	        }
		                    	}
		                    }
		                }
		            }
		        }
		        if(stringTrueNodeLabelsList.size() != 0 )
		        {
		        	trueNodeLabels =  convertStringLabels(stringTrueNodeLabelsList);
		        }
		        
		        if(trueNodeLabels.size() != predictedNodeLabels.size())
		        {
		        	return "true labels size: " + trueNodeLabels +" and predicted labels:" + predictedNodeLabels + " does not have the same size";
		        }
		        else {
		        	adjustedRandIndexValue = calculateAdjustedRandIndex(trueNodeLabels, predictedNodeLabels);
				}
		    }
		    return "ajusted rand index of " + nodeSet + " is: " + adjustedRandIndexValue ;
	    }
	}
    

    @UserFunction
    @Description("Calculate the mean of the Silhouette Coefficients for all point")
	public String averageSilhouetteCoefficient(@Name("nodeSet") String nodeSet, @Name("numberOfCentroid") String numberOfCentroid, @Name("numberOfInteration") String numberOfInteration, @Name("distanceMeasure") String distanceMeasure) throws Exception
	{
    	if(nodeSet != null)
    	{
			String averageSilhouetteCoefficientString = "The average Silhouette Coefficient value is: ";
			HashMap<String, ArrayList<String>> kmeanAssign = new HashMap<String, ArrayList<String>>();
			int numberOfCentroidInt = Integer.parseInt(numberOfCentroid);
			int numberOfInterationInt = Integer.parseInt(numberOfInteration);
			ArrayList<String> debug = new ArrayList<>();
			kmeanAssign = Unsupervised.KmeanClust(mapNodeList, numberOfCentroidInt, numberOfInterationInt, distanceMeasure, false, debug);
			double averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(kmeanAssign, distanceMeasure);
	        return averageSilhouetteCoefficientString + averageSilhouetteCoefficientValue ;
		}
    	else
    	{
    		return null;
    	}
	}
    
    
	/**
	 * This function is used to query the data from graph database
	 * @param nodeType type of node
	 */
	public void queryData( final String nodeType)
	{
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					Result result = tx.run( "MATCH (n:" + nodeType + ") RETURN n");
					dataKey = result.list();
					return "Query Successful";
				}
			} );
		}
	}
    
	public void connectNodes(final String nodeType, final String message, final String nodeCentroid, final String nodeCluster, final double distance)
    {
    	final String name = "kmean";
        try ( Session session = driver.session() )
        {
            String greeting = session.writeTransaction( new TransactionWork<String>()
            {
                @Override
                public String execute( Transaction tx )
                {
					// First, find and delete the existing node if it exists
					String deleteQuery = "MATCH (a:Clustering_" + nodeType + " {" + nodeCentroid + "}) " +
										 "DETACH DELETE a";
					tx.run(deleteQuery);
					// Then create the new nodes and relationship
					String createQuery = "MERGE (a:Clustering_" + nodeType + " {" + nodeCentroid + "}) " +
										 "MERGE (b:Clustering_" + nodeType + " {" + nodeCluster + "}) " +
										 "MERGE (a)-[r:link]->(b) " +
										 "SET r.distance = " + distance + " " +
										 "RETURN a.message";

					Result result = tx.run(createQuery);
					return result.single().get(0).asString();
                }
            } );
		}
	}
	
	private String getNodeValues(Value value, String[] overLookArray) {
	    StringBuilder valueOfNode = new StringBuilder();
	    for (String nodeKey : value.keys()) {
	        if (overLookArray.length > 0 && Arrays.asList(overLookArray).contains(nodeKey)) {
	            continue;
	        }
	        try {
	            double num = Double.parseDouble(String.valueOf(value.get(nodeKey)));
	            if (value.get(nodeKey).getClass().equals(String.class)) {
	                valueOfNode.append(getStringValue(valueOfNode)).append(nodeKey).append(":").append(value.get(nodeKey));
	            } else {
	                valueOfNode.append(getStringValue(valueOfNode)).append(nodeKey).append(":").append(value.get(nodeKey));
	            }
	        } catch (NumberFormatException e) {
	            System.out.println(value.get(nodeKey) + " is not a number.");
	        }
	    }
	    return valueOfNode.toString();
	}
	
	private String getStringValue(StringBuilder valueOfNode) {
	    return valueOfNode.length() > 0 ? ", " : "";
	}

    public static List<Double> convertStringLabels(List<String> strings) {
        Map<String, Double> labelMap = new HashMap<>();
        List<Double> labels = new ArrayList<>();

        double currentLabel = 0.0;
        for (String s : strings) {
            if (!labelMap.containsKey(s)) {
                labelMap.put(s, currentLabel++);
            }
            labels.add(labelMap.get(s));
        }

        return labels;
    }
    
    public static double calculateAdjustedRandIndex(List<Double> trueLabels, List<Double> predictedLabels) {
        if (trueLabels.size() != predictedLabels.size()) {
            throw new IllegalArgumentException("Input lists must have the same length");
        }

        int n = trueLabels.size();
        Map<Double, Map<Double, Double>> contingencyTable = new HashMap<>();
        Map<Double, Double> trueLabelCounts = new HashMap<>();
        Map<Double, Double> predictedLabelCounts = new HashMap<>();

        // Build the contingency table and label counts
        for (int i = 0; i < n; i++) {
            double trueLabel = trueLabels.get(i);
            double predictedLabel = predictedLabels.get(i);

            contingencyTable.computeIfAbsent(trueLabel, k -> new HashMap<>());
            contingencyTable.get(trueLabel).merge(predictedLabel, 1.0, Double::sum);

            trueLabelCounts.merge(trueLabel, 1.0, Double::sum);
            predictedLabelCounts.merge(predictedLabel, 1.0, Double::sum);
        }

        double a = 0.0; // Number of pairs in the same cluster in both true and predicted
        for (Map<Double, Double> row : contingencyTable.values()) {
            for (double count : row.values()) {
                a += count * (count - 1) / 2.0;
            }
        }

        double b = 0.0; // Number of pairs in the same cluster in trueLabels
        for (double count : trueLabelCounts.values()) {
            b += count * (count - 1) / 2.0;
        }

        double c = 0.0; // Number of pairs in the same cluster in predictedLabels
        for (double count : predictedLabelCounts.values()) {
            c += count * (count - 1) / 2.0;
        }

        double totalPairs = n * (n - 1) / 2.0;
        double expectedIndex = (b * c) / totalPairs;
        double maxIndex = 0.5 * (b + c);
        double adjustedRandIndex = (a - expectedIndex) / (maxIndex - expectedIndex);

        return adjustedRandIndex;
    }
	
    /**
     * This function performs spectral clustering on graph data stored in a Neo4j database. 
     * It integrates the steps of the spectral clustering algorithm, including constructing the similarity graph, 
     * eigendecomposition of the graph, k-means clustering, and evaluation using silhouette score and adjusted rand index. 
     * The function also generates CSV files for the similarity graph and eigendecomposed graph.
     * 
     * @param node_label Label of the nodes in the Neo4j graph to be used for clustering.
     * @param distance_measure Method to calculate the similarity between nodes (e.g., Euclidean, cosine).
     * @param graph_type Type of graph to be created for clustering (e.g., k-nearest neighbors, epsilon-neighborhood).
     * @param parameter Parameter value for the graph type (e.g., epsilon value or number of neighbors).
     * @param remove_columns Columns to be removed before performing clustering.
     * @param laplacian_type Type of Laplacian matrix to be used (e.g., normalized, random walk).
     * @param number_of_eigenvectors Number of eigenvectors to be considered in the eigendecomposition step.
     * @param number_of_iteration Number of iterations for the k-means clustering.
     * @param distance_measure_kmean Distance measure to be used in the k-means clustering step.
     * 
     * @return Generated graphs, CSVs, silhouette score and adjusted rand index.
     * 
     * @throws Exception if any error occurs during the spectral clustering process.
     */

	@UserFunction
	public String spectralClusteringFromNeo4j(@Name("node_label") String node_label, @Name("distance_measure") String distance_measure,@Name("graph_type") String graph_type,@Name("parameter") String parameter,@Name("remove_column") String remove_columns, @Name("laplacian_type") String laplacian_type, @Name("number_of_eigenvectors") Double number_of_eigenvectors, @Name("number_of_iteration") String number_of_iteration, @Name("distance_measure_kmean") String distance_measure_kmean, @Name("target Column on original Data") String target_column, @Name("Use kmean for Silhouette calculation") Boolean use_kmean_for_silhouette) throws Exception {
		predictedNodeLabels.clear();
		try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
			if(node_label == null && distance_measure == null) {
				return "Missing dataPath or distance measure type";
			}else {
				String graphName = null;
				org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodeData = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(node_label, connector.getDriver());
				ArrayList<NodeList2> nodePropertiesList = nodeData.getLeft();
	            String propertyNames = nodeData.getRight(); 
				//				ArrayList<NodeList2> nodePropertiesList_copy = Neo4jGraphHandler.retrieveNodeListFromNeo4j(label, connector.getDriver());
				String[] removeList = remove_columns.split(",");
				List<String> removeListNew = Arrays.stream(removeList).collect(Collectors.toList());
//				Double[][] DistanceMatrix = GraphTransform.euclideanDistance(nodePropertiesList);
				Double[][] DistanceMatrix = getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);
				Double[][] adj_mat = null;

				if(graph_type.equals("full")) {
					Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
					graphName = graph_type.concat("_"+parameter);
				}
				if(graph_type.equals("eps")) {
					Double espilonValue = Double.parseDouble(parameter);
					adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,espilonValue);
					graphName = graph_type.concat("_"+parameter);

				}
				if(graph_type.equals("knn")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateKNNGraph(DistanceMatrix,knn);
					graphName = graph_type.concat("_"+parameter);

				}
				if(graph_type.equals("mknn")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
					adj_mat = ReadCsvTestData.calculateMutualKNNGraph(DistanceMatrix,knn);
					graphName = graph_type.concat("_"+parameter);
				}
				ArrayList<EdgeList2> edgeList = GraphTransform.calculateEdgeList(nodePropertiesList,adj_mat);

				try (Session session = connector.getDriver().session();
			             Transaction tx = session.beginTransaction()) {

					//for (EdgeList edgeListDetail : edgeList) {
					for (NodeList2 node : nodePropertiesList) {
						Neo4jGraphHandler.createNodeGraph(graphName.concat("new"), "created nodes in neo4j", node, connector.getDriver());
					}
	
					for (int i = 0; i < edgeList.size(); i++) {
						EdgeList2 edgeListDetail = edgeList.get(i);
						if(edgeListDetail.getWeight()==0.0){
							continue;
						}
						Neo4jGraphHandler.createRelationshipGraph(graphName.concat("new"), "created relationship in neo4j \n", edgeListDetail, connector.getDriver());
	
					}
					
					Neo4jGraphHandler.exportCSVFile(graphName.concat("new"),graphName.concat("new"),connector.getDriver());
					
					tx.commit();
				}
				try (Session session = connector.getDriver().session()) {
		            ArrayList<EdgeList2> edge_list = new ArrayList<>();
		            edge_list = Neo4jGraphHandler.retrieveEdgeListFromNeo4j(graphName.concat("new"), connector.getDriver());
	
		            RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list);
		            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
		            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
	
		            
		            EigenCalculation.EigenResult eigen_result = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);
		            ArrayList<NodeList2> node_list_eigen = Neo4jGraphHandler.retrieveNodeListFromNeo4j(graphName.concat("new"), connector.getDriver());
		            ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_list_eigen, eigen_result.X, edge_list);
	
		            
		            String graph_name = "eigenGraph_" + laplacian_type + "_" + graphName.concat("new") + "_" + Math.round(number_of_eigenvectors);
		            
		            for (NodeList2 node : node_list_eigen) {
		            	Neo4jGraphHandler.createNodeGraphEigenTransform(graph_name, "created nodes in neo4j", node, eigen_result.X, connector.getDriver());
		            }
		            for (int i = 0; i < edge_list_eigen.size(); i++) {
						EdgeList2 edge_list_detail = edge_list_eigen.get(i);
		            	Neo4jGraphHandler.createRelationshipGraph(graph_name, "created relationship in neo4j \n", edge_list_detail, connector.getDriver());
		            }
		            
		            Neo4jGraphHandler.exportCSVFile(graph_name,graph_name,connector.getDriver());
		            
		            //mapNodes(graph_name,propertyNames);
		            
		            String number_of_clusters = Integer.toString(number_of_eigenvectors.intValue());
					// Split the string into an array
					String[] propertiesArray = propertyNames.split(",");

					// Use a StringBuilder to build the new string without 'index' and 'target'
					StringBuilder filteredProperties = new StringBuilder();

					for (String property : propertiesArray) {
						if (!property.equals("index") && !property.equals("target")) {
							if (filteredProperties.length() > 0) {
								filteredProperties.append(",");
							}
							filteredProperties.append(property);
						}
					}
		            String kmeanResult = kmean(Map.of(
						"nodeSet", graph_name,
						"numberOfCentroid", number_of_clusters,
						"numberOfInteration", number_of_iteration,
						"distanceMeasure", distance_measure_kmean,
						"originalSet", node_label,
						"overlook", target_column + filteredProperties.toString(),
						"overlookOriginal", target_column,
						"useKmeanForSilhouette", use_kmean_for_silhouette
					));
					//String kmeanResult = "test";

			        return "Created similarity graph, eigendecomposed graph successful!" + kmeanResult;
		            
				}catch (Neo4jException e) {
			        throw new RuntimeException("Error creating laplacian graph in Neo4j: " + e.getMessage());
			    }

	        }
	    } catch (Neo4jException e) {
	        throw new RuntimeException("Error creating laplacian graph in Neo4j: " + e.getMessage());
	    }
	}

	/**
	 * Displays information about a graph based on the specified node type and desired number of eigenvectors. Laplacian matrix set on "Symmetric".
	 *
	 * @param node_label           The label of the nodes in the graph.
	 * @param number_of_eigenvectors The number of desired eigenvectors to compute during eigen decomposition.
	 * @return String containing information about the graph, including edge list, adjacency matrix, degree matrix, Laplacian matrix, and eigen decomposition results.
	 * @throws Exception
	 */
    @UserFunction
    public String displayGraphList(@Name("node_label") String node_label, @Name("numberOfEigenvectors") Double number_of_eigenvectors, @Name("laplacian_type") String laplacian_type) throws Exception {
        try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
            if (node_label == null) {
                return "Missing nodeType";
            } else {
                ArrayList<EdgeList2> edge_list = Neo4jGraphHandler.retrieveEdgeListFromNeo4j(node_label, connector.getDriver());

                // Display edge list
                StringBuilder outputString = new StringBuilder("Edge List Data: ");
                for (EdgeList2 edge : edge_list) {
                    outputString.append(" | ").append(edge.toString());
                }

                // Display adjacency matrix
                RealMatrix adjacency_matrix = MatrixCalculation.convertToAdjacencyMatrix(edge_list);
                outputString.append("\n\nAdjacency Matrix:\n").append(matrixToString(adjacency_matrix));

                // Display degree matrix
                RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix);
                outputString.append("\n\nDegree Matrix:\n").append(matrixToString(degree_matrix));

                // Display Laplacian matrix
                RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);
                outputString.append("\n\nLaplacian Matrix:\n").append(matrixToString(laplacian_matrix));

                // Display Eigenvalues, Eigenvectors, and X
//                EigenCalculation.EigenResult eigen_result = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);
//                outputString.append("\n\nEigenvectors Matrix:\n").append(matrixToString(eigen_result.eigenvectors));
//				outputString.append("\n\nX Matrix:\n").append(matrixToString(eigen_result.X));
				
                return outputString.toString();
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error displaying edge list in Neo4j: " + e.getMessage());
		}
    }

	@UserFunction
	public String displayEdgeList(@Name("nodeType") String nodeType, @Name("dataPath") String dataPath, @Name("distance_measure") String distance_measure, @Name("graph_type") String graph_type, @Name("method") String method, @Name("parameter") String parameter,@Name("remove_column") String remove_columns) throws Exception {
		
		try (SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password)) {
		
		if(dataPath == null && distance_measure == null) {
			return "Missing data_path or distance measure type";
		}else {
			
            // Display edge list

			
			String graphName = null;
			Double[][] adj_mat = null;
			ArrayList<NodeList2> nodePropertiesList = null;
			String[] removeList = remove_columns.split(",");
			List<String> removeListNew = Arrays.stream(removeList).collect(Collectors.toList());
			if(nodeType.isEmpty()) {
				ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
//				//ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNew(data_path,IndexColumn);
				nodePropertiesList = readCsvTestData.readCsvFileToMap(dataPath);
			}
			org.apache.commons.lang3.tuple.Pair<ArrayList<NodeList2>, String> nodeData = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(nodeType, connector.getDriver());
			nodePropertiesList = nodeData.getLeft();
//			nodePropertiesList = Neo4jGraphHandler.retrieveNodeListFromNeo4jSimilarityGraph(nodeType, connector.getDriver());

//            ArrayList<NodeList2> nodePropertiesList_copy = readCsvTestData.readCsvFileToMap(data_path);
			Double[][] DistanceMatrix = getDistanceMatrixFromNodes(distance_measure,nodePropertiesList,removeListNew);

            StringBuilder outputString = new StringBuilder("Graph Data: ");
            outputString.append("\n\nDistance Matrix:\n").append(doubleToString(DistanceMatrix));
			
			
			if(graph_type.equals("full")) {
				Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix,parameter);
				adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
				graphName = graph_type.concat("_"+parameter);
			}
			if(graph_type.equals("eps")) {
				Double espilonValue = Double.parseDouble(parameter);
				adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,espilonValue);
				graphName = graph_type.concat("_"+parameter);
			}
			if(graph_type.equals("knn")) {
				Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
				adj_mat = ReadCsvTestData.calculateKNNGraph(DistanceMatrix,knn);
				graphName = graph_type.concat("_"+parameter);
			}
			if(graph_type.equals("mknn")) {
				Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,parameter);
				adj_mat = ReadCsvTestData.calculateMutualKNNGraph(DistanceMatrix,knn);
				graphName = graph_type.concat("_"+parameter);
			}
			
			outputString.append("\n\nAdjacency Matrix:\n").append(doubleToString(adj_mat));
			
			ArrayList<EdgeList2> edgeList = GraphTransform.calculateEdgeList(nodePropertiesList,adj_mat);
			
			outputString.append("\n\nEdge List:\n");
            for (EdgeList2 edge : edgeList) {
                outputString.append(" | ").append(edge.toString());
            }
			
			
			return outputString.toString() + '\n' + "nodes " + nodePropertiesList;
		}
		}
    }

	
	@UserFunction
	public String displayFinalResults(@Name("nodeSet") String nodeSet, @Name("numberOfCentroid") String numberOfCentroid, @Name("numberOfInteration") String numberOfInteration, @Name("distanceMeasure") String distanceMeasure) throws Exception {
		
    	predictedNodeLabels.clear();
		    	try ( SimKitProcedures connector = new SimKitProcedures(SimKitProcedures.uri, SimKitProcedures.username, SimKitProcedures.password))
        {
		    		
		    		if(nodeSet == null && distanceMeasure == null) {
		    			return "Missing data_path or distance measure type";
		    		}else {

		    StringBuilder outputString = new StringBuilder("The average Silhouette Coefficient value is: ");
			HashMap<String, ArrayList<String>> kmeanAssign = new HashMap<String, ArrayList<String>>();
			int numberOfCentroidInt = Integer.parseInt(numberOfCentroid);
			int numberOfInterationInt = Integer.parseInt(numberOfInteration);
			double centroidNumber = 1.0;

			ArrayList<String> debug = new ArrayList<>();
			kmeanAssign = Unsupervised.KmeanClust(mapNodeList, numberOfCentroidInt, numberOfInterationInt, distanceMeasure, false, debug);
			double averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(kmeanAssign, distanceMeasure);
			outputString.append(averageSilhouetteCoefficientValue);

            outputString.append("\n\nKmean assign:\n").append(hashMapToString(kmeanAssign));
            
			for (String centroid : kmeanAssign.keySet()) {
			    ArrayList<String> clusterNode = kmeanAssign.get(centroid);
			    for (int i = 0; i < clusterNode.size(); i++) {
			    	//Add predict labels
		    		predictedNodeLabels.add(centroidNumber);

			    	DecimalFormat decimalFormat = new DecimalFormat("#.###");
			        double distance = Unsupervised.calculateDistance(clusterNode.get(i), centroid, distanceMeasure);
			        String formattedDistance = decimalFormat.format(distance);
			        double roundedDistance = Double.parseDouble(formattedDistance);
			        connector.connectNodes(nodeSet, "create relationship in kmean node", centroid, clusterNode.get(i), distance);

//			        connector.connectNodes(nodeSet, "create relationship in kmean node", centroid, clusterNode.get(i), roundedDistance);
			    }
			    centroidNumber = centroidNumber + 1;
			}

			return outputString + " predicted labels: " + predictedNodeLabels;
		}
		}
    }
	
    public static String doubleToString(Double[][] matrix) {
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                // Format the number with proper alignment (e.g., 2 decimal places)
                sb.append(matrix[i][j]);
                if (j < matrix[i].length - 1) {
                    sb.append(" "); // Separate columns with a space
                }
            }
            sb.append("\n"); // Newline after each row
        }
        
        return sb.toString();
    }

    private String matrixToString(RealMatrix matrix) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                result.append(matrix.getEntry(i, j)).append("\t");
            }
            result.append("\n");
        }
        return result.toString();
    }
    
    public String hashMapToString(HashMap<String, ArrayList<String>> kmeanAssign) {
        StringBuilder sb = new StringBuilder();
        
        // Iterate through the entries of the HashMap
        for (HashMap.Entry<String, ArrayList<String>> entry : kmeanAssign.entrySet()) {
            String key = entry.getKey();
            ArrayList<String> values = entry.getValue();
            
            sb.append(key).append(": ");
            
            // Append the values in the ArrayList to the StringBuilder
            sb.append(values.toString());
            
            // Add a newline after each key-value pair
            sb.append("\n");
        }
        
        return sb.toString();
    }
}




