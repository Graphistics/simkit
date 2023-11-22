package main;
import java.util.Scanner;

import definition.EdgeList2;
import definition.NodeList2;
import global.Neo4jGraphHandler;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import cv.CrossValidation;
import evaluate.EvaluateTree;
import gainratio.EvaluateTreeGR;
import gini.EvaluateTreeGI;
import input.ProcessInputData;
import output.PrintTree;
import graph.*;

import static org.neo4j.driver.Values.parameters;

import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.regex.Pattern;

import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Record;
import org.neo4j.driver.Result;
import org.neo4j.driver.Session;
import org.neo4j.driver.Transaction;
import org.neo4j.driver.TransactionWork;
import org.neo4j.driver.Value;
import org.neo4j.driver.util.Pair;


/**
 *
 * This class is used to fetch nodes from graph database or from csv and call the functions to generate decision tree
 * with confusion matrix, generation time and prediction time for the output
 *
 * @author minh dung
 *
 */

public class OutputDecisionTreeNeo4j implements AutoCloseable{

	private static Driver driver;
	private static List<Record> dataKey = new ArrayList<>();
	private static ArrayList<String> testDataList =  new ArrayList<String>();
	private static ArrayList<String> trainDataList =  new ArrayList<String>();
	private static ArrayList<String> autoSplitDataList =  new ArrayList<String>();
	private static ArrayList<String> classificationDataList = new ArrayList<String>();

	/**
	 * Creation of driver object using bolt protocol
	 * @param uri Uniform resource identifier for bolt
	 * @param user Username
	 * @param password Password
	 */
	public OutputDecisionTreeNeo4j( String uri, String user, String password )
	{
		driver = GraphDatabase.driver( uri, AuthTokens.basic( user, password ) );

	}
	public Driver getDriver() {
		return driver;
	}

	/**
	 * Empty constructor
	 */
	public OutputDecisionTreeNeo4j()
	{
		driver = null;
	}

	/**
	 * Close the driver object
	 */
	@Override
	public void close() throws Exception
	{
		driver.close();
	}


	public void connectNodeToClassLabel(final String nodeType, final String classLabel, final String node)
	{
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					//a is present for the node
					Result result = tx.run( "MERGE (a:" + nodeType + "{" + node +"}) " +
							"MERGE (b {" + "predictedLabel:"+ classLabel +"}) " +
							"MERGE (a)-[:link]->(b) "
							+ "RETURN a.message");
					return "connected";
				}
			} );
		}
	}

	@UserFunction
	public String classifyOfNodes(@Name("nodeType") String nodeType, @Name("decisionTreeType") String decisionTreeType , @Name("classLabel") String targetAttribute ) throws Exception
	{
		String output = "";
		classificationDataList.clear();
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123" ) )
		{
			boolean isTrainListEmpty = trainDataList.isEmpty();
			boolean isTestListEmpty = testDataList.isEmpty();
			if(isTrainListEmpty && isTestListEmpty) {
				return targetAttribute + "False";
			}else {

				EvaluateTree mine;
				if(decisionTreeType == "IG")
				{
					mine = new EvaluateTree(trainDataList, testDataList, targetAttribute,"False",0);
				}
				else if (decisionTreeType == "GI")
				{
					mine = new EvaluateTreeGI(trainDataList, testDataList, targetAttribute,"False",0);
				}
				else
				{
					mine = new EvaluateTreeGR(trainDataList, testDataList, targetAttribute,"False",0);
				}

				mine.calculateAccuracy();
				HashMap<String, ArrayList<String>> hashMapClassify = mine.predictedResults;
				for (String classLabel: hashMapClassify.keySet()) {
					ArrayList<String> arrayNodes = hashMapClassify.get(classLabel);
					for (String node : arrayNodes)
					{
						connector.connectNodeToClassLabel(nodeType,classLabel,node);
					}
				}
				output = hashMapClassify.values().toString();

			}
		}

		return output;
	}
	public void createNodeConnectedGraph(final String GraphType, final String message, final String nodeDetail,final int index)
	{
		final String name = "Index" + index;
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					//CREATE (p:fullyConnected {id:"p2", a1: 2, a2: 8,a3: 2})
					//a is present for the node
					Result result = tx.run( "CREATE (connectedGraph:" +GraphType+
							"{id:" +"\""+name+"\""+
							",NodeProperties:" + "\""+nodeDetail+"\""
							+ "})", parameters( "name", name ) );
					return message;
				}
			} );
		}
	}

	/**
	 *
	 * Create nodes in Neo4j using Java
	 * @param dtType Type of decision tree
	 * @param message String The message that print to Console
	 * @param nodeDetail ArrayList<String> Detail of a node
	 */
	public void createNode(final String dtType, final String message, final ArrayList<String> nodeDetail)
	{
		final String name = nodeDetail.get(2);
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					//a is present for the node
					Result result = tx.run( "CREATE (" + nodeDetail.get(0) + dtType + nodeDetail.get(1) +") " +
									"SET a.name = $name" +
									" SET a.l = " + nodeDetail.get(3) +
									" SET a.i = " + nodeDetail.get(4) +
									" SET a.dupValue = " + nodeDetail.get(5) +
									" RETURN a.message + ', from node ' + id(a)",
							parameters( "name", name ) );
					return result.single().get( 0 ).asString();
				}
			} );
		}
	}
	public void createRelationshipConnectedGraph( final String dtType, final String message, final EdgeList edgeListDetail)
	{
		final long bid = edgeListDetail.getTarget();
		final String bindex = "Index" + bid;
		double weightValue = (double)Math.round(edgeListDetail.getWeight() * 100000d) / 100000d;
		final String weight = "`" + Double.toString(weightValue) + "` {value: " + weightValue + "}" ;
		//final String weight = "_" + Double.toString(edgeListDetail.getWeight()).replace(".","_") + "_";
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					Result result = tx.run( "MATCH (a:" + dtType + "), (b:" + dtType + ") " +
							"WHERE a.id = "+"\"" +"Index"+edgeListDetail.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
							"CREATE (a)-[r:`link` {weight:" + "$weight" +  "}]->(b)",parameters("weight",weight));
					return message;
				}
			} );
		}
	}




	/**
	 * Create relationship between nodes in Java
	 * @param dtType Type of decision tree
	 * @param message String the message that print to Console
	 * @param relationshipDetail ArrayList<String> Detail of a relationship
	 */
	public void createRelationship( final String dtType, final String message, final ArrayList<String> relationshipDetail)
	{
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					Result result = tx.run( "MATCH (a:" + dtType + "), (b:" + dtType + ") " +
							"Where a.name = '" + relationshipDetail.get(0) + "' And " +
							"a.l = " + relationshipDetail.get(1) + " And " +
							"a.dupValue = " + relationshipDetail.get(2) + " And " +
							"b.name = '" + relationshipDetail.get(3) + "' And " +
							"b.dupValue = " + relationshipDetail.get(5) + " And " +
							"b.l = " + relationshipDetail.get(4) +
							" Create (a)-[r:"+"_"+ relationshipDetail.get(6)+"_"+" {type: '" + relationshipDetail.get(7) +
							"' , value: '" +relationshipDetail.get(6) +
							"' , propname: '" + relationshipDetail.get(0) + "' }]->(b)" +
							" RETURN type(r)");
					return result.single().get( 0 ).asString();
				}
			} );
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
	@UserFunction
	public String loadCsvGraph(@Name("dataPath") String dataPath,@Name("Name") String Name)  throws Exception{

		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{
			if(dataPath == null && Name == null) {
				return "Missing dataPath or distance measure type";
			}else {
				ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
				ArrayList<String> arrayListHeaders = readCsvTestData.readCSVHeader(dataPath);
				ArrayList<String> arrayListFirst = readCsvTestData.readCSVFirstLine(dataPath);
				connector.loadCsvConnector(dataPath, Name ,arrayListHeaders,arrayListFirst);
			}
		}catch(Exception e) {
			throw new RuntimeException(e);
		}
		return dataPath;
	}
	private void loadCsvConnector(String dataPath, String Name,ArrayList<String> arrayListHeaders,ArrayList<String> arrayListFirst) throws Exception {

		// LOAD CSV with headers FROM 'file:///test.csv' AS row
		//merge (:csvdata8 {points: row.points,x_cordinate: toFloat(row.x_coordinate),y_cordinate: toFloat(row.y_coordinate),class: toFloat(row.class)})
		String proerties = OutputDecisionTreeNeo4j.getHeadersList(arrayListHeaders,arrayListFirst);

		String FileName = Name.substring(0,Name.indexOf("."));
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					Result result = tx.run( "LOAD CSV WITH HEADERS FROM 'file:///"+Name+"' AS row "+"merge (:"+FileName+"{"+ proerties+"})", parameters( "name", Name ) );
					return Name;
				}
			} );
		}

	}

	private static String getHeadersList(ArrayList<String> arrayListHeaders,ArrayList<String> arrayListFirst) {

		StringBuilder stringBuilder = new StringBuilder();
		Pattern pattern = Pattern.compile("-?\\d+(\\.\\d+)?");


		for (int i = 0; i < arrayListHeaders.size(); i++) {

			if(i == arrayListHeaders.size()-1) {

				if (pattern.matcher(arrayListFirst.get(i)).matches()) {
					stringBuilder.append(arrayListHeaders.get(i) + ": toFloat(row." + arrayListHeaders.get(i) + ")");
				} else
					stringBuilder.append(arrayListHeaders.get(i) + ": row." + arrayListHeaders.get(i));
			}else {
				if (pattern.matcher(arrayListFirst.get(i)).matches()) {
					stringBuilder.append(arrayListHeaders.get(i) + ": toFloat(row." + arrayListHeaders.get(i) + "),");
				} else
					stringBuilder.append(arrayListHeaders.get(i) + ": row." + arrayListHeaders.get(i) + ",");
			}


		}

		return stringBuilder.toString();
	}

	public Double [][] getDistanceMatrix(String distanceMeasure,ArrayList<ArrayList<String>> testData){
		Double[][] DistanceMatrix = null;


		switch (distanceMeasure){
			case "euclideanDistance":
				DistanceMatrix = ReadCsvTestData.euclidianDistance(testData);
				break;
			case "manhattanDistance" :
				DistanceMatrix = DistanceMeasure.calculateManhattanDistance(testData);
				break;
			case "canberraDistance" :
				DistanceMatrix = DistanceMeasure.calculateCanberraDistance(testData);
				break;
			case "cosineSimilarity" :
				DistanceMatrix = DistanceMeasure.calculateCosineSimilarity(testData);
				break;
			case "jaccardCoefficient" :
				DistanceMatrix = DistanceMeasure.calculateJaccardCoefficient(testData);
				break;
			case "brayCurtisDistance" :
				DistanceMatrix = DistanceMeasure.calculateBrayCurtisDistance(testData);
				break;
			default:
				System.out.println("give correct name");
		}


		return DistanceMatrix;



	}


	@UserFunction
	public String createGraph(@Name("dataPath") String dataPath,@Name("distanceMeasure") String distanceMeasure,@Name("IndexBoolean") Boolean IndexColumn,@Name("graphType") String graphType,@Name("epsilon") Double epsilon,@Name("sigma") String sigma,@Name("Knn") String knn_neighbour) throws Exception {

		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{



			if(dataPath == null && distanceMeasure == null) {
				return "Missing dataPath or distance measure type";
			}else {
				Double[][] adj_mat = null;
				ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
				ArrayList<String> fileHeader = readCsvTestData.readCSVHeader(dataPath);
				ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNew(dataPath,IndexColumn);
				Double[][] DistanceMatrix = getDistanceMatrix(distanceMeasure,testData);

				if(graphType.equals("ConnectedGraph")) {
					Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix,sigma);
					adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
				}
				if(graphType.equals("EpsilonGraph")) {
					adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,epsilon);

				}
				if(graphType.equals("knnGraph")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,knn_neighbour);
					adj_mat = ReadCsvTestData.calculateKNNGraph(DistanceMatrix,knn);

				}
				if(graphType.equals("MutualKnnGraph")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,knn_neighbour);
					adj_mat = ReadCsvTestData.calculateMutualKNNGraph(DistanceMatrix,knn);
				}

				ArrayList<String> nodeList = ReadCsvTestData.getNodeList(testData);
				ArrayList<EdgeList> edgeList = ReadCsvTestData.calulateEdgeList(adj_mat);

				for (int i = 0; i < nodeList.size(); i++) {

					connector.createNodeConnectedGraph(graphType, "created nodes in neo4j", nodeList.get(i),i);
				}



//				for (String nodeDetail : nodeList) {
//
//					connector.createNodeConnectedGraph("connectedGraph", "created nodes in neo4j", nodeDetail);
//				}

				//for (EdgeList edgeListDetail : edgeList) {
				for (int i = 0; i < edgeList.size(); i++) {
					EdgeList edgeListDetail = edgeList.get(i);
					if(edgeListDetail.getWeight()==0.0){
						continue;
					}
					connector.createRelationshipConnectedGraph(graphType, "created relationship in neo4j \n", edgeListDetail);
				}

			}
			return "Create fully connected graph successful, " + confusionMatrix;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}
	@UserFunction
	public String createGraphFromNodes(@Name("label") String label,@Name("distanceMeasure") String distanceMeasure,@Name("graphType") String graphType,@Name("epsilon") Double epsilon,@Name("sigma") String sigma,@Name("Knn") String knn_neighbour) throws Exception {

		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{


			if(label == null && distanceMeasure == null) {
				return "Missing dataPath or distance measure type";
			}else {
				String graphName = null;
				ArrayList<NodeList2> nodePropertiesList = Neo4jGraphHandler.retrieveNodeListFromNeo4j(label, connector.getDriver());
				//Double[][] DistanceMatrix = GraphTransform.euclideanDistance(nodePropertiesList);
				Double[][] DistanceMatrix = getDistanceMatrixFromNodes(distanceMeasure,nodePropertiesList);
				Double[][] adj_mat = null;
//				ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
//				ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNew(dataPath,IndexColumn);
				if(graphType.equals("ConnectedGraph")) {
					Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix,sigma);
					adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
					graphName = "connectedGraph";
				}
				if(graphType.equals("EpsilonGraph")) {
					adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,epsilon);
					graphName = "epsilonGraph";

				}
				if(graphType.equals("knnGraph")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,knn_neighbour);
					adj_mat = ReadCsvTestData.calculateKNNGraph(DistanceMatrix,knn);
					graphName = "knnGraph";

				}
				if(graphType.equals("MutualKnnGraph")) {
					Double[][] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,knn_neighbour);
					adj_mat = ReadCsvTestData.calculateMutualKNNGraph(DistanceMatrix,knn);
					graphName = "mutualKnnGraph";
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

	private Double[][] getDistanceMatrixFromNodes(String distanceMeasure, ArrayList<NodeList2> nodePropertiesList) {
		Double[][] DistanceMatrix = null;


		switch (distanceMeasure) {
			case "euclideanDistance":
				DistanceMatrix = GraphTransform.euclideanDistance(nodePropertiesList);
				break;
			case "manhattanDistance":
				DistanceMatrix = DistanceMeasureNodes.manhattanDistance(nodePropertiesList);
				break;
			case "canberraDistance":
				DistanceMatrix = DistanceMeasureNodes.canberraDistance(nodePropertiesList);
				break;
			case "cosineSimilarity":
				DistanceMatrix = DistanceMeasureNodes.cosineSimilarity(nodePropertiesList);
				break;
			case "jaccardCoefficient":
				DistanceMatrix = DistanceMeasureNodes.jaccardCoefficient(nodePropertiesList);
				break;
			case "brayCurtisDistance":
				DistanceMatrix = DistanceMeasureNodes.brayCurtisDistance(nodePropertiesList);
				break;
			default:
				System.out.println("Invalid distance measure type");
		}

		return DistanceMatrix;

	}

	public void createRelationshipConnectedGraphFromExistingNodes( final String dtType, final String message, final EdgeList edgeListDetail)
	{
		final long bid = edgeListDetail.getTarget();
		final String bindex = "Index" + bid;
		double weightValue = (double)Math.round(edgeListDetail.getWeight() * 100000d) / 100000d;
		final String weight = "`" + Double.toString(weightValue) + "` {value: " + weightValue + "}" ;
		//final String weight = "_" + Double.toString(edgeListDetail.getWeight()).replace(".","_") + "_";
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					Result result = tx.run( "MATCH (a:" + dtType + "), (b:" + dtType + ") " +
							"WHERE a.id = "+"\"" +"Index"+edgeListDetail.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
							"CREATE (a)-[r:" + weight +  "]->(b)");
					return message;
				}
			} );
		}
	}


//	@UserFunction
//	public String createGraphStringClass(@Name("dataPath") String dataPath,@Name("distanceMeasure") String distanceMeasure,@Name("IndexBoolean") Boolean IndexColumn,@Name("graphType") String graphType,@Name("epsilon") Double epsilon,@Name("sigma") int sigma,@Name("Knn") int knn_neighbour,@Name("class types") List<String> classTypes) throws Exception {
//
//		String confusionMatrix = "";
//		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
//		{
//
//
//			if(dataPath == null && distanceMeasure == null) {
//				return "Missing dataPath or distance measure type";
//			}else {
//				Double[][] adj_mat = null;
//				ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
//				ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNewString(dataPath,IndexColumn,classTypes);
//				if(graphType.equals("connectedGraph")) {
//					Double[][] DistanceMatrix = ReadCsvTestData.euclidianDistance(testData);
//					Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix,sigma);
//					adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
//				}
//				if(graphType.equals("epsilonGraph")) {
//					Double[][] DistanceMatrix = ReadCsvTestData.euclidianDistance(testData);
//					adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,epsilon);
//
//				}
//				if(graphType.equals("knnGraph")) {
//					Double[][] DistanceMatrix = ReadCsvTestData.euclidianDistance(testData);
//					Double[] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,knn_neighbour);
//					adj_mat = ReadCsvTestData.calculateKNNGraph(DistanceMatrix,knn);
//
//				}
//				if(graphType.equals("mutualKnnGraph")) {
//					Double[][] DistanceMatrix = ReadCsvTestData.euclidianDistance(testData);
//					Double[] knn = ReadCsvTestData.calculateKNN(DistanceMatrix,knn_neighbour);
//					adj_mat = ReadCsvTestData.calculateMutualKNNGraph(DistanceMatrix,knn);
//				}
//
//				ArrayList<String> nodeList = ReadCsvTestData.getNodeList(testData);
//				ArrayList<EdgeList> edgeList = ReadCsvTestData.calulateEdgeList(adj_mat);
//
//
//
//				for (String nodeDetail : nodeList) {
//
//					connector.createNodeConnectedGraph("connectedGraph", "created nodes in neo4j", nodeDetail);
//				}
//
//				//for (EdgeList edgeListDetail : edgeList) {
//				for (int i = 0; i < edgeList.size()-1; i++) {
//					EdgeList edgeListDetail = edgeList.get(i);
////					ArrayList<String> relationshipDetail = new ArrayList<>();
////					relationshipDetail.add(String.valueOf(edgeListDetail.getSource()));
////					relationshipDetail.add(String.valueOf(edgeListDetail.getTarget()));
////					relationshipDetail.add(String.valueOf(edgeListDetail.getWeight()));
////					String testCypherQuery = "MATCH (a:" + dtType + "), (b:" + dtType + ") " +
////							"WHERE a.id = "+"\"" +"Index"+edgeListDetail.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
////							"CREATE (a)-[r:" + edgeListDetail.getWeight() + "]->(b)";
//					connector.createRelationshipConnectedGraph("Index", "created relationship in neo4j \n", edgeListDetail);
//				}
//
//			}
//			return "Create fully connected graph successful, " + confusionMatrix;
//		} catch (Exception e) {
//			throw new RuntimeException(e);
//		}
//
//	}

//	@UserFunction
//	public String createEpsilonGraph(@Name("dataPath") String dataPath,@Name("distanceMeasure") String distanceMeasure,@Name("IndexBoolean") Boolean IndexColumn,@Name("epsilon") Double epsilon) throws Exception {
//
//		String confusionMatrix = "";
//		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
//		{
//
//
//			if(dataPath == null && distanceMeasure == null) {
//				return "Missing dataPath or distance measure type";
//			}else {
//				ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
//				ArrayList<ArrayList<String>> testData = readCsvTestData.readCsvFileNew(dataPath,IndexColumn);
//
//				Double[][] DistanceMatrix = ReadCsvTestData.euclidianDistance(testData);
//				//Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix);
//				//Double[][] adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
//				Double[][] adj_mat = ReadCsvTestData.calculateEpsilonNeighbourhoodGraph(DistanceMatrix,epsilon);
//
//				ArrayList<String> nodeList = ReadCsvTestData.getNodeList(testData);
//				ArrayList<EdgeList> edgeList = ReadCsvTestData.calulateEdgeList(adj_mat);
//
//
//
//				for (String nodeDetail : nodeList) {
//
//					connector.createNodeConnectedGraph("connectedGraph", "created nodes in neo4j", nodeDetail);
//				}
//
//				//for (EdgeList edgeListDetail : edgeList) {
//				for (int i = 0; i < edgeList.size()-1; i++) {
//					EdgeList edgeListDetail = edgeList.get(i);
//					if(edgeListDetail.getWeight()==0.0){
//						continue;
//					}
//					connector.createRelationshipConnectedGraph("Index", "created relationship in neo4j \n", edgeListDetail);
//				}
//
//			}
//			return "Create epsilon  graph successful, " + confusionMatrix;
//		} catch (Exception e) {
//			throw new RuntimeException(e);
//		}
//
//	}
	/**
	 * This function is used to split the nodes from database based on training ratio given
	 * @param nodeType
	 * @param trainRatio
	 * @return String with train ratio and test ratio
	 * @throws Exception
	 */
	@UserFunction
	public String queryAutoSplitData(@Name("nodeType") String nodeType, @Name("trainRatio") String trainRatio ) throws Exception
	{
		String listOfData = "";
		double testRatio = 0;
		autoSplitDataList.clear();
		testDataList.clear();
		trainDataList.clear();
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123" ) )
		{
			queryData(nodeType);
			for (Record key : dataKey) {
				List<Pair<String,Value>> values = key.fields();
				for (Pair<String,Value> nodeValues: values) {
					String valueOfNode = "";
					if ("n".equals(nodeValues.key())) {
						Value value = nodeValues.value();
						for (String nodeKey : value.keys())
						{
							if(value.get(nodeKey).getClass().equals(String.class))
							{
								if(valueOfNode != "")
								{
									valueOfNode = valueOfNode + ", " + nodeKey + ":" + value.get(nodeKey);
								}
								else
								{
									valueOfNode = nodeKey + ":" + value.get(nodeKey);
								}

							}
							else
							{
								if(valueOfNode != "")
								{
									String converValueToString = String.valueOf(value.get(nodeKey));
									valueOfNode = valueOfNode + ", " + nodeKey + ":" + converValueToString;
								}
								else
								{
									String converValueToString = String.valueOf(value.get(nodeKey));
									valueOfNode =  nodeKey + ":" + converValueToString;
								}
							}
						}
						autoSplitDataList.add(valueOfNode);
					}
				}
			}
			int size = autoSplitDataList.size();
			double sizeForTrain = Math.floor(size * Double.parseDouble(trainRatio));
			int startTestData =  (int) sizeForTrain;
			testRatio = 1 - Double.parseDouble(trainRatio);
			//Add data to trainDataList
			for (int i = 0; i < startTestData; i++)
			{
				trainDataList.add(autoSplitDataList.get(i));
			}
			//Add data to testDataList
			for (int i = startTestData; i < size; i++)
			{
				testDataList.add(autoSplitDataList.get(i));
			}
		}
		return "The Data has been split -  Train Ratio: " + trainRatio + " Test Ratio: " + testRatio;
	}



	/**
	 * This function is used to query the test dataset from Neo4j and populate the global arraylist of Java Code
	 * @param nodeType The name of the type of node.For example, P_test for Test
	 * @return String showing the data queried
	 * @throws Exception if connection to Neo4j fails
	 */
	@UserFunction
	public String queryTestData(@Name("nodeType") String nodeType) throws Exception
	{
		String listOfData = "";
		testDataList.clear();
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{
			queryData(nodeType);
			for (Record key : dataKey) {
				List<Pair<String,Value>> values = key.fields();
				for (Pair<String,Value> nodeValues: values) {
					String valueOfNode = "";
					if ("n".equals(nodeValues.key())) {
						Value value = nodeValues.value();
						for (String nodeKey : value.keys())
						{
							if(value.get(nodeKey).getClass().equals(String.class))
							{
								if(valueOfNode != "")
								{
									String valueKey = ":" + value.get(nodeKey);
									valueOfNode = valueOfNode + "," + nodeKey +  valueKey.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+value.get(nodeKey));
								}
								else
								{
									String valueKey = ":" + value.get(nodeKey);
									valueOfNode = nodeKey + valueKey.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+value.get(nodeKey));
								}

							}
							else
							{
								if(valueOfNode != "")
								{
									String converValueToString = String.valueOf(value.get(nodeKey));
									valueOfNode = valueOfNode + "," + nodeKey + ":" + converValueToString.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+converValueToString);
								}
								else
								{
									String converValueToString = String.valueOf(value.get(nodeKey));
									valueOfNode =  nodeKey + ":" + converValueToString.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+converValueToString);
								}

							}
						}
						testDataList.add(valueOfNode);
					}
					listOfData = listOfData + valueOfNode + " | ";
				}
			}
		}
		return "The Data: " + listOfData;
	}

	/**
	 * This function is used to query the training dataset from Neo4j and populate the global trainDataList of Java Code
	 *
	 * @param nodeType
	 * @return
	 * @throws Exception
	 */

	@UserFunction
	public String queryTrainData(@Name("nodeType") String nodeType) throws Exception
	{
		String listOfData = "";
		trainDataList.clear();
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{
			queryData(nodeType);
			for (Record key : dataKey) {
				List<Pair<String,Value>> values = key.fields();
				for (Pair<String,Value> nodeValues: values) {
					String valueOfNode = "";
					if ("n".equals(nodeValues.key())) {
						Value value = nodeValues.value();
						for (String nodeKey : value.keys())
						{
							if(value.get(nodeKey).getClass().equals(String.class))
							{
								if(valueOfNode != "")
								{
									String valueKey = ":" + value.get(nodeKey);
									valueOfNode = valueOfNode + "," + nodeKey +  valueKey.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+value.get(nodeKey));
								}
								else
								{
									String valueKey = ":" + value.get(nodeKey);
									valueOfNode = nodeKey + valueKey.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+value.get(nodeKey));
								}
							}
							else
							{
								if(valueOfNode != "")
								{
									String converValueToString = String.valueOf(value.get(nodeKey));
									valueOfNode = valueOfNode + "," + nodeKey + ":" + converValueToString.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+converValueToString);
								}
								else
								{
									String converValueToString = String.valueOf(value.get(nodeKey));
									valueOfNode =  nodeKey + ":" + converValueToString.replaceAll("^\"|\"$", "");
									//nodeData.add(nodeKey+":"+converValueToString);
								}
							}
						}
						trainDataList.add(valueOfNode);
					}
					listOfData = listOfData + valueOfNode + " | ";
				}
			}
		}
		return "The Data: " + listOfData;
	}
	/**
	 * This function is used to display the nodes which has been queried and populated already. Used for testing purpose.
	 * @param dataType
	 * @return String showing the data queried
	 * @throws Exception if connection to Neo4j fails
	 */

	@UserFunction
	public String displayData(@Name("dataType") String dataType) throws Exception
	{
		String listOfData = "";
		int countLine = 0;
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123" ) )
		{
			if(dataType.equals("train"))
			{

				listOfData = "train data: ";
				for(String node : trainDataList)
				{
					countLine++;
					listOfData = listOfData + node + "|";
				}
			}
			else if (dataType.equals("all"))
			{

				listOfData = "all data: ";
				for(String node : autoSplitDataList)
				{
					countLine++;
					listOfData = listOfData + node + "|";
				}
			}
			else
			{

				listOfData = "test data: ";
				for(String node : testDataList)
				{
					countLine++;
					listOfData = listOfData + node + "|";
				}

			}
		}
		return "Number of Lines: " + countLine + " The " + listOfData;

	}

	/**
	 * User defined function to create the decision tree with nodes and relationships in neo4j. This creates a tree based on information gain.
	 * @param target attribute
	 * @return
	 * @throws Exception
	 */
	@UserFunction
	public String createTreeIG(@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth) throws Exception {

		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{

			boolean isTrainListEmpty = trainDataList.isEmpty();
			boolean isTestListEmpty = testDataList.isEmpty();
			if(isTrainListEmpty && isTestListEmpty) {
				return target + "False";
			}else {
				int maxDepth = Integer.parseInt(max_depth);
				EvaluateTree mine = new EvaluateTree(trainDataList, testDataList, target, isPruned, maxDepth);

				confusionMatrix = mine.calculateAccuracy();

				PrintTree tree = new PrintTree();

				tree.createNodesForGraph(mine.getRoot());

				for (ArrayList<String> nodeDetail : tree.nodesBucket) {
					connector.createNode("DTInfoGain", "create nodes in neo4j", nodeDetail);
				}

				for (ArrayList<String> relationshipDetail : tree.relationshipsBucket) {
					System.out.println("Relationship " + relationshipDetail);
					connector.createRelationship("DTInfoGain", "create relationship in neo4j \n", relationshipDetail);
				}

			}
			return "Create the Information Gain Decision Tree successful, " + confusionMatrix;
		}

	}

	/**
	 * User defined function to create the decision tree with nodes and relationships in neo4j. This creates a tree based on gini index.
	 * @param target attribute
	 * @return
	 * @throws Exception
	 */
	@UserFunction
	public String createTreeGI(@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth) throws Exception {

		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123" ) )
		{

			boolean isTrainListEmpty = trainDataList.isEmpty();
			boolean isTestListEmpty = testDataList.isEmpty();
			if(isTrainListEmpty && isTestListEmpty) {
				return target + "False";
			}else {
				int maxDepth = Integer.parseInt(max_depth);
				EvaluateTreeGI mine = new EvaluateTreeGI(trainDataList, testDataList, target, isPruned, maxDepth);

				confusionMatrix = mine.calculateAccuracy();

				PrintTree tree = new PrintTree();

				tree.createNodesForGraph(mine.getRoot());

				for (ArrayList<String> nodeDetail : tree.nodesBucket) {
					connector.createNode("DTGini", "create nodes in neo4j", nodeDetail);
				}

				for (ArrayList<String> relationshipDetail : tree.relationshipsBucket) {
					System.out.println("Relationship " + relationshipDetail);
					connector.createRelationship("DTGini", "create relationship in neo4j \n", relationshipDetail);
				}

			}
			return "Create the Gini Index Decision Tree successful, " + confusionMatrix;
		}

	}


	/**
	 * User defined function to create the decision tree with nodes and relationships in neo4j. This creates a tree based on gain ratio.
	 * @param target attribute
	 * @return
	 * @throws Exception
	 */
	@UserFunction
	public String createTreeGR(@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth) throws Exception {

		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{

			boolean isTrainListEmpty = trainDataList.isEmpty();
			boolean isTestListEmpty = testDataList.isEmpty();
			if(isTrainListEmpty && isTestListEmpty) {
				return target + "False";
			}else {
				int maxDepth = Integer.parseInt(max_depth);
				EvaluateTreeGR mine = new EvaluateTreeGR(trainDataList, testDataList, target, isPruned, maxDepth);

				confusionMatrix = mine.calculateAccuracy();

				PrintTree tree = new PrintTree();

				tree.createNodesForGraph(mine.getRoot());

				for (ArrayList<String> nodeDetail : tree.nodesBucket) {
					connector.createNode("DTGainRatio", "create nodes in neo4j", nodeDetail);
				}

				for (ArrayList<String> relationshipDetail : tree.relationshipsBucket) {
					System.out.println("Relationship " + relationshipDetail);
					connector.createRelationship("DTGainRatio", "create relationship in neo4j \n", relationshipDetail);
				}

			}
			return "Create the Gain Ratio Decision Tree successful, " + confusionMatrix;
		}

	}


	/**
	 * User defined function to create the decision tree with nodes and relationships in neo4j
	 * path
	 * @return
	 * @throws Exception
	 */
	@UserFunction
	public String createTreeGIcsv(@Name("trainPath") String trainPath,@Name("testPath") String testPath, @Name("targetAttribute") String targetAttribute, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth ) throws Exception
	{
		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123" ) )
		{
			Scanner in = new Scanner(System.in);

			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTreeGI mine = new EvaluateTreeGI(trainPath, testPath, targetAttribute, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();

			PrintTree tree = new PrintTree();

			tree.createNodesForGraph(mine.getRoot());


			in.close();

			for (ArrayList<String> nodeDetail : tree.nodesBucket) {
				connector.createNode("DTGini","create nodes in neo4j", nodeDetail);
			}

			for (ArrayList<String> relationshipDetail : tree.relationshipsBucket) {
				System.out.println("Relationship " + relationshipDetail);
				connector.createRelationship("DTGini","create relationship in neo4j \n", relationshipDetail);
			}
		}

		return "Create the Gini Index Decision Tree successful, " + confusionMatrix;

	}
	/**
	 * This function creates tree from csv path which is based on gain ratio
	 *     is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
	 * @return
	 * @throws Exception
	 */

	@UserFunction
	public String createTreeGRcsv(@Name("trainPath") String trainPath,@Name("testPath") String testPath, @Name("targetAttribute") String targetAttribute, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth) throws Exception
	{
		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123" ) )
		{
			Scanner in = new Scanner(System.in);

			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTreeGR mine = new EvaluateTreeGR(trainPath, testPath, targetAttribute, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();

			PrintTree tree = new PrintTree();

			tree.createNodesForGraph(mine.getRoot());


			in.close();

			for (ArrayList<String> nodeDetail : tree.nodesBucket) {
				connector.createNode("DTGainRatio","create nodes in neo4j", nodeDetail);
			}

			for (ArrayList<String> relationshipDetail : tree.relationshipsBucket) {
				System.out.println("Relationship " + relationshipDetail);
				connector.createRelationship("DTGainRatio","create relationship in neo4j \n" , relationshipDetail);
			}
		}
		return "Create the Gain Ratio Decision Tree successful, " + confusionMatrix;
	}


	/**
	 * This function creates tree from csv path which is based on information gain
	 *
	 *  path - The path is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
	 * @return
	 * @throws Exception
	 */

	@UserFunction
	public String createTreeIGcsv(@Name("trainPath") String trainPath,@Name("testPath") String testPath, @Name("targetAttribute") String targetAttribute, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth )throws Exception
	{
		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123" ) )
		{
			Scanner in = new Scanner(System.in);

			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTree mine = new EvaluateTree(trainPath, testPath, targetAttribute, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();

			PrintTree tree = new PrintTree();

			tree.createNodesForGraph(mine.getRoot());


			in.close();

			for (ArrayList<String> nodeDetail : tree.nodesBucket) {
				connector.createNode("DTInfoGain","create nodes in neo4j", nodeDetail);
			}

			for (ArrayList<String> relationshipDetail : tree.relationshipsBucket) {
				System.out.println("Relationship " + relationshipDetail);
				connector.createRelationship("DTInfoGain","create relationship in neo4j \n", relationshipDetail);
			}
		}

		return "Create the Info Gain Decision Tree successful, " + confusionMatrix;

	}

	/**
	 * This function retrieves the confusion matrix of decision tree based on information gain
	 *  path
	 *  target
	 * @return
	 * @throws Exception
	 */
	@UserFunction
	@Description("retrieve the confusion matrix Information Gain Decision Tree")
	public String confmIGcsv(@Name("trainPath") String trainPath,@Name("testPath") String testPath, @Name("targetAttribute") String targetAttribute , @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth ) throws Exception
	{
		if(trainPath == null || testPath == null )
		{
			return null;
		}
		else
		{
			String confusionMatrix = "";
			Scanner in = new Scanner(System.in);

			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTree mine = new EvaluateTree(trainPath, testPath, targetAttribute, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();
			return "The confusion Matrix for Information Gain DT : " + confusionMatrix;
		}
	}


	/**
	 *
	 * This function retrieves the confusion matrix of decision tree based on gain ratio
	 *  path
	 * @return
	 * @throws Exception
	 */

	@UserFunction
	@Description("retrieve the confusion matrix Gain Ratio Decision Tree")
	public String confmGRcsv(@Name("trainPath") String trainPath,@Name("testPath") String testPath, @Name("targetAttribute") String targetAttribute, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth ) throws Exception
	{
		if(trainPath == null || testPath == null)
		{
			return null;
		}
		else
		{
			String confusionMatrix = "";
			Scanner in = new Scanner(System.in);

			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTreeGR mine = new EvaluateTreeGR(trainPath, testPath, targetAttribute, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();
			return "The confusion Matrix for Gain Ratio DT: " + confusionMatrix;
		}
	}

	/**
	 *
	 * This function retrieves the confusion matrix of decision tree based on gini index
	 *  path - The path is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
	 * @return A string with confusion matrix
	 * @throws Exception
	 */

	@UserFunction
	@Description("retrieve the confusion matrix Gini Index Decision Tree")
	public String confmGIcsv(@Name("trainPath") String trainPath,@Name("testPath") String testPath, @Name("targetAttribute") String targetAttribute, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth) throws Exception
	{
		if(trainPath == null || testPath == null)
		{
			return null;
		}
		else
		{
			String confusionMatrix = "";
			Scanner in = new Scanner(System.in);

			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTreeGI mine = new EvaluateTreeGI(trainPath, testPath, targetAttribute, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();
			return "The confusion Matrix for Gini Index DT: " + confusionMatrix;
		}
	}

	/**
	 * This function retrieves the confusion matrix of decision tree based on information gain
	 *  path
	 * @param target
	 * @return
	 * @throws Exception
	 */
	@UserFunction
	@Description("retrieve the confusion matrix Information Gain Decision Tree")
	public String confmIG(@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth ) throws Exception
	{
		boolean isTrainListEmpty = trainDataList.isEmpty();
		boolean isTestListEmpty = testDataList.isEmpty();
		if(isTrainListEmpty && isTestListEmpty) {
			return "Need to query to data";
		}
		else
		{
			String confusionMatrix = "";
			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTree mine = new EvaluateTree(trainDataList, testDataList, target, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();
			return "The confusion Matrix for Information Gain DT : " + confusionMatrix;
		}
	}

	@UserFunction
	@Description("retrieve the confusion matrix Gain Ratio Decision Tree")
	public String confmGR(@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth ) throws Exception
	{
		boolean isTrainListEmpty = trainDataList.isEmpty();
		boolean isTestListEmpty = testDataList.isEmpty();
		if(isTrainListEmpty && isTestListEmpty) {
			return "Need to query to data";
		}
		else
		{
			String confusionMatrix = "";
			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTreeGR mine = new EvaluateTreeGR(trainDataList, testDataList, target, isPruned, maxDepth);

			confusionMatrix = mine.calculateAccuracy();
			return "The confusion Matrix for Gain Ratio DT: " + confusionMatrix;
		}
	}

	/**
	 *
	 * This function retrieves the confusion matrix of decision tree based on gini index
	 *  path - The path is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
	 * @return A string with confusion matrix
	 * @throws Exception
	 */

	@UserFunction
	@Description("retrieve the confusion matrix Gini Index Decision Tree")
	public String confmGI(@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth ) throws Exception
	{
		boolean isTrainListEmpty = trainDataList.isEmpty();
		boolean isTestListEmpty = testDataList.isEmpty();
		if(isTrainListEmpty && isTestListEmpty) {
			return "Need to query to data";
		}
		else
		{
			String confusionMatrix = "";
			int maxDepth = Integer.parseInt(max_depth);
			EvaluateTreeGI mine = new EvaluateTreeGI(trainDataList, testDataList, target, isPruned, maxDepth);


			confusionMatrix = mine.calculateAccuracy();
			return "The confusion Matrix for Gini Index DT: " + confusionMatrix;
		}
	}

	@UserFunction
	@Description("cross validation time for data from graph database for InfoGain")
	public String cvIG(@Name("target") String target, @Name("numberOfFold") String numberOfFold) throws Exception
	{
		if(target == null)
		{
			return null;
		}
		else
		{

			CrossValidation cv = new CrossValidation(autoSplitDataList, target);

			ArrayList<Double> final_score = cv.validate(Integer.parseInt(numberOfFold), "InfoGain");
			double mcc = cv.getMccAverage();
			double generateTime = cv.getCvGenerationTimeAverage();
			double score = cv.getScoreAverage();
			String cfm = cv.getCfmDiabetes();
			String result = "calculated average mcc: " + mcc + "\n" + "calculated average generateTime: " + generateTime +
					"\n" + "confusion matrix: " + cfm +
					"\n" + "calculated average accuracy: " + score;

			return result;
		}
	}


	@UserFunction
	@Description("cross validation time for data from graph database for GainRatio")
	public String cvGR(@Name("target") String target, @Name("numberOfFold") String numberOfFold) throws Exception
	{
		if(target == null)
		{
			return null;
		}
		else
		{

			CrossValidation cv = new CrossValidation(autoSplitDataList, target);

			ArrayList<Double> final_score = cv.validate(Integer.parseInt(numberOfFold), "GainRatio");
			double mcc = cv.getMccAverage();
			double generateTime = cv.getCvGenerationTimeAverage();
			double score = cv.getScoreAverage();
			String cfm = cv.getCfmDiabetes();
			String result = "calculated average mcc: " + mcc + "\n" + "calculated average generateTime: " + generateTime +
					"\n" + "confusion matrix: " + cfm +
					"\n" + "calculated average accuracy: " + score;

			return result;
		}
	}

	@UserFunction
	@Description("cross validation time for data from graph database for GiniIndex")
	public String cvGI(@Name("target") String target, @Name("numberOfFold") String numberOfFold) throws Exception
	{
		if(target == null)
		{
			return null;
		}
		else
		{

			CrossValidation cv = new CrossValidation(autoSplitDataList, target);

			ArrayList<Double> final_score = cv.validate(Integer.parseInt(numberOfFold), "GiniIndex");
			double mcc = cv.getMccAverage();
			double generateTime = cv.getCvGenerationTimeAverage();
			double score = cv.getScoreAverage();
			String cfm = cv.getCfmDiabetes();
			String result = "calculated average mcc: " + mcc + "\n" + "calculated average generateTime: " + generateTime +
					"\n" + "confusion matrix: " + cfm +
					"\n" + "calculated average accuracy: " + score;

			return result;
		}
	}

	@UserFunction
	@Description("cross validation time for data from csv for InfoGain")
	public String cvIGcsv(@Name("path") String path, @Name("target") String target, @Name("numberOfFold") String numberOfFold) throws Exception
	{
		if(path == null)
		{
			return null;
		}
		else
		{
			ArrayList<String> customList = ProcessInputData.CustomListFromCSV(path);
			CrossValidation cv = new CrossValidation(customList, target);

			ArrayList<Double> final_score = cv.validate(Integer.parseInt(numberOfFold), "InfoGain");
			double mcc = cv.getMccAverage();
			double generateTime = cv.getCvGenerationTimeAverage();
			double score = cv.getScoreAverage();
			String cfm = cv.getCfmDiabetes();
			String result = "calculated average mcc: " + mcc + "\n" + "calculated average generateTime: " + generateTime +
					"\n" + "confusion matrix: " + cfm +
					"\n" + "calculated average accuracy: " + score;

			return result ;
		}
	}




	/**
	 * To calculate the average of a list
	 * @param final_score
	 * @return
	 */

	private double calculateAverage(ArrayList<Double> final_score) {
		return final_score.stream()
				.mapToDouble(d -> d)
				.average()
				.orElse(0.0);
	}


	@UserFunction
	@Description("cross validation time for data from csv for GainRatio")
	public String cvGRcsv(@Name("path") String path, @Name("target") String target, @Name("numberOfFold") String numberOfFold) throws Exception
	{
		if(path == null)
		{
			return null;
		}
		else
		{

			ArrayList<String> customList = ProcessInputData.CustomListFromCSV(path);
			CrossValidation cv = new CrossValidation(customList, target);

			ArrayList<Double> final_score = cv.validate(Integer.parseInt(numberOfFold), "GainRatio");
			double mcc = cv.getMccAverage();
			double generateTime = cv.getCvGenerationTimeAverage();
			double score = cv.getScoreAverage();
			String cfm = cv.getCfmDiabetes();
			String result = "calculated average mcc: " + mcc + "\n" + "calculated average generateTime: " + generateTime +
					"\n" + "confusion matrix: " + cfm +
					"\n" + "calculated average accuracy: " + score;

			return result ;
		}
	}

	@UserFunction
	@Description("cross validation time for data from csv for GiniIndex")
	public String cvGIcsv(@Name("path") String path, @Name("target") String target, @Name("numberOfFold") String numberOfFold) throws Exception
	{
		if(path == null)
		{
			return null;
		}
		else
		{

			ArrayList<String> customList = ProcessInputData.CustomListFromCSV(path);
			CrossValidation cv = new CrossValidation(customList, target);

			ArrayList<Double> final_score = cv.validate(Integer.parseInt(numberOfFold), "GiniIndex");
			double mcc = cv.getMccAverage();
			double generateTime = cv.getCvGenerationTimeAverage();
			double score = cv.getScoreAverage();
			String cfm = cv.getCfmDiabetes();
			String result = "calculated average mcc: " + mcc + "\n" + "calculated average generateTime: " + generateTime +
					"\n" + "confusion matrix: " + cfm +
					"\n" + "calculated average accuracy: " + score;


			return result ;
		}
	}

	@UserFunction
	@Description("generate the feature table from neo4j dataset")
	public String featureTable(@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth, @Name("Algorithm Type") String algoType) throws Exception
	{
		boolean isTrainListEmpty = trainDataList.isEmpty();
		boolean isTestListEmpty = testDataList.isEmpty();
		if(isTrainListEmpty && isTestListEmpty) {
			return "Need to query to data";
		}
		else
		{
			String featureTable = "";
			int maxDepth = Integer.parseInt(max_depth);
			if (algoType.equals("GR"))
			{
				EvaluateTreeGR mine = new EvaluateTreeGR(trainDataList, testDataList, target, isPruned, maxDepth);
				mine.calculateAccuracy();
				featureTable = mine.getFeatureTable();
			}
			else if (algoType.equals("GI"))
			{
				EvaluateTreeGI mine = new EvaluateTreeGI(trainDataList, testDataList, target, isPruned, maxDepth);
				mine.calculateAccuracy();
				featureTable = mine.getFeatureTable();
			}
			else
			{
				EvaluateTree mine = new EvaluateTree(trainDataList, testDataList, target, isPruned, maxDepth);
				mine.calculateAccuracy();
				featureTable = mine.getFeatureTable();
			}

			return "The feature table: " + featureTable;
		}
	}

	@UserFunction
	@Description("generate the feature table from neo4j dataset")
	public String featureTableCsv(@Name("trainPath") String trainPath,@Name("testPath") String testPath,@Name("target") String target, @Name("isPruned") String isPruned, @Name("maxDepth") String max_depth, @Name("Algorithm Type") String algoType) throws Exception
	{
		if(trainPath == null || testPath == null)
		{
			return null;
		}
		else
		{
			String featureTable = "";
			int maxDepth = Integer.parseInt(max_depth);
			if (algoType.equals("GR"))
			{
				EvaluateTreeGR mine = new EvaluateTreeGR(trainPath, testPath, target, isPruned, maxDepth);
				mine.calculateAccuracy();
				featureTable = mine.getFeatureTable();
			}
			else if (algoType.equals("GI"))
			{
				EvaluateTreeGI mine = new EvaluateTreeGI(trainPath, testPath, target, isPruned, maxDepth);
				mine.calculateAccuracy();
				featureTable = mine.getFeatureTable();
			}
			else
			{
				EvaluateTree mine = new EvaluateTree(trainPath, testPath, target, isPruned, maxDepth);
				mine.calculateAccuracy();
				featureTable = mine.getFeatureTable();
			}

			return "The feature table: " + featureTable;
		}

	}

	public static void main(String[] args) throws Exception {
		OutputDecisionTreeNeo4j outputDecisionTreeNeo4j = new OutputDecisionTreeNeo4j();
		//String dataPath = "D:/de/MASTER_THESIS/Decision-Tree-Neo4j/Java Plugin/DecisionTreePlugin/src/main/resources/test.csv";
		String Filename = "D:/de/MASTER_THESIS/SimKit/simkit/dataset_1_iris_numeric/Iris_Mythica_Data_Set.csv";
		String testDataPath = "C:/Users/abhiv/.Neo4jDesktop/relate-data/dbmss/dbms-7ff4399d-8f4a-48ed-ba21-c30aa62e37b7/import/test.csv";

//		outputDecisionTreeNeo4j.createGraphFromNodes("test","test","connectedGraph",2.0,2,2);
		outputDecisionTreeNeo4j.createGraph("C:/Users/abhiv/.Neo4jDesktop/relate-data/dbmss/dbms-7ff4399d-8f4a-48ed-ba21-c30aa62e37b7/import/test.csv","euclideanDistance",true,"knnGraph",2.0,"0","3");
		//outputDecisionTreeNeo4j.createConnectedGraph(testDataPath, "fully-conncted");

		//test
//		ReadCsvTestData readCsvTestData = new ReadCsvTestData(testDataPath);
//		ArrayList<TestData> testData = readCsvTestData.readCsvFile(testDataPath);
//		Double[][] DistanceMatrix = readCsvTestData.euclidianDistance(testData);
//		Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix);
//		Double[][] adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);
//
//		ArrayList<Nodelist> nodeList = ReadCsvTestData.getNodeList(testData);
//		ArrayList<EdgeList> edgeList = ReadCsvTestData.calulateEdgeList(adj_mat);
//		for (int i = 0; i < edgeList.size()-1; i++) {
//			EdgeList edgeListDetail = edgeList.get(i);
//			System.out.println("edgeListDetail " + edgeList.get(i));
//
//			long bid = edgeListDetail.getTarget();
//			String dtType = "connected";
//			String bindex = "Index" + bid;
//			double weightValue = (double)Math.round(edgeListDetail.getWeight() * 100000d) / 100000d;
//			String weight = "`" + Double.toString(weightValue) + "` {value: " + weightValue + "`}" ;
//			String finalString = "MATCH (a:" + dtType + ")S, (b:" + dtType + ") " +
//					"WHERE a.id = "+"\"" +"Index"+edgeListDetail.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
//					"CREATE (a)-[r:" + weight +  "]->(b)";
//			System.out.println("--" + finalString);
		}

	}




