package main;
import static org.neo4j.driver.Values.parameters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Pattern;

import org.apache.commons.math4.legacy.linear.BlockRealMatrix;
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
import org.neo4j.driver.exceptions.Neo4jException;
import org.neo4j.driver.types.Node;
import org.neo4j.driver.types.Relationship;
import org.neo4j.driver.util.Pair;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import cv.CrossValidation;
import eigendecomposed.EigenCalculation;
import eigendecomposed.MatrixCalculation;
import evaluate.EvaluateTree;
import gainratio.EvaluateTreeGR;
import gini.EvaluateTreeGI;
import graph.EdgeList;
import graph.GraphData;
import graph.Nodelist;
import graph.ReadCsvTestData;
import graph.TestData;
import input.ProcessInputData;
import output.PrintTree;

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
	private static ArrayList<String> mapNodeList =  new ArrayList<String>();

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
	public void createNodeConnectedGraph(final String GraphType, final String message, final Nodelist nodeDetail)
	{
		final String name = "Node" + nodeDetail.getIndex().toString();
		try ( Session session = driver.session() )
		{
			String greeting = session.writeTransaction( new TransactionWork<String>()
			{
				@Override
				public String execute( Transaction tx )
				{
					//CREATE (p:fullyConnected {id:"p2", a1: 2, a2: 8,a3: 2})
					//a is present for the node
					Result result = tx.run( "CREATE (connectedGraph:Index" +
//							"{id:" +"\""+name+"\""+
//							",x_cordinate: " + nodeDetail.getX_cordinate()
							"{x_cordinate: " + nodeDetail.getX_cordinate()
							+ ",y_cordinate: " + nodeDetail.getY_cordinate()
							+ ",class_attrribute: " + nodeDetail.getClass_attrribute()
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
							"CREATE (a)-[r:" + weight +  "]->(b)");
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


	@UserFunction
	public String createConnectedGraph(@Name("dataPath") String dataPath,@Name("distanceMeasure") String distanceMeasure)  throws Exception {

		String confusionMatrix = "";
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{


			if(dataPath == null && distanceMeasure == null) {
				return "Missing dataPath or distance measure type";
			}else {
				ReadCsvTestData readCsvTestData = new ReadCsvTestData(dataPath);
				ArrayList<TestData> testData = readCsvTestData.readCsvFile(dataPath);
				Double[][] DistanceMatrix = readCsvTestData.euclidianDistance(testData);
				Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix);
				Double[][] adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);

				ArrayList<Nodelist> nodeList = ReadCsvTestData.getNodeList(testData);
				ArrayList<EdgeList> edgeList = ReadCsvTestData.calulateEdgeList(adj_mat);



				for (Nodelist nodeDetail : nodeList) {

					connector.createNodeConnectedGraph("connectedGraph", "created nodes in neo4j", nodeDetail);
				}

				//for (EdgeList edgeListDetail : edgeList) {
				for (int i = 0; i < edgeList.size()-1; i++) {
					EdgeList edgeListDetail = edgeList.get(i);
//					ArrayList<String> relationshipDetail = new ArrayList<>();
//					relationshipDetail.add(String.valueOf(edgeListDetail.getSource()));
//					relationshipDetail.add(String.valueOf(edgeListDetail.getTarget()));
//					relationshipDetail.add(String.valueOf(edgeListDetail.getWeight()));
//					String testCypherQuery = "MATCH (a:" + dtType + "), (b:" + dtType + ") " +
//							"WHERE a.id = "+"\"" +"Index"+edgeListDetail.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
//							"CREATE (a)-[r:" + edgeListDetail.getWeight() + "]->(b)";
					connector.createRelationshipConnectedGraph("Index", "created relationship in neo4j \n", edgeListDetail);
				}

			}
			return "Create fully connected graph successful, " + confusionMatrix;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

    private static ArrayList<EdgeList> retrieveEdgeListFromNeo4j(final String nodeType) {
        ArrayList<EdgeList> edgeList = new ArrayList<>();

        try (Session session = driver.session()) {
	        String cypherQuery = "MATCH (n:" + nodeType + ")-[r]->(m:" + nodeType + ") RETURN toFloat(split(n.id, 'Index')[1]) AS source, toFloat(split(m.id, 'Index')[1]) AS target, toFloat(type(r)) AS weight, id(r) AS index";
	        Result result = session.run(cypherQuery);
	
	        while (result.hasNext()) {
	            Record record = result.next();
	            long source = record.get("source").asLong();
	            long target = record.get("target").asLong();
	            double weight = record.get("weight").asDouble();
	            int index = record.get("index").asInt();
	
	            EdgeList edge = new EdgeList(source, target, weight, index);
	            edgeList.add(edge);
	        }
	    } catch (Neo4jException e) {
	    	throw new RuntimeException("Error retrieving graph data from Neo4j: " + e.getMessage());
	    }    
		return edgeList;
	}
    
    @UserFunction
	public String displayEdgeList(@Name("nodeType") String nodeType) throws Exception {
		
    	String outputString = "";
		try (OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j("bolt://localhost:7687", "neo4j", "123412345")) {
	        if (nodeType == null) {
	            return "Missing nodeType";
	        } else {
	        	ArrayList<EdgeList> edgeList = new ArrayList<>();
	        	edgeList = retrieveEdgeListFromNeo4j(nodeType);
	        	if(edgeList.size()>0)
	        	{
	        		for(int i = 0; i < edgeList.size(); i++) {   
		        		outputString = outputString + " | " + edgeList.get(i).toString(); 
	        		}
	        	}
        		else
        		{
        			outputString = "could not retrieve edge list";
        		}
        	}
	        return "Edge List Data:  " + outputString;
        }
	     catch (Neo4jException e) {
	        throw new RuntimeException("Error creating laplacian graph in Neo4j: " + e.getMessage());
	    }
	}
    
	@UserFunction
	public String createLaplacianGraph(@Name("nodeType") String nodeType, @Name("threshold") double threshold) throws Exception {
		
		try (OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j("bolt://localhost:7687", "neo4j", "123412345")) {
	        if (nodeType == null) {
	            return "Missing nodeType";
	        } else {
	        	
	        	ArrayList<EdgeList> edgeList = retrieveEdgeListFromNeo4j(nodeType);
	        	
	            double[][] adjacencyMatrixData = MatrixCalculation.convertToAdjacencyMatrix(edgeList);
	            RealMatrix adjacencyMatrix = new BlockRealMatrix(adjacencyMatrixData);
	            RealMatrix degreeMatrix = MatrixCalculation.calculateDegreeMatrix(adjacencyMatrixData);
	            RealMatrix laplacianMatrix = MatrixCalculation.calculateSymmetricLaplacianMatrix(degreeMatrix, adjacencyMatrix);
	            					
				
	        	EigenCalculation.EigenResult eigenResult = EigenCalculation.calculateEigen(laplacianMatrix);
	        	double[] eigenvalues = eigenResult.eigenvalues;
	        	RealMatrix eigenvectors = eigenResult.eigenvectors;
	        	ArrayList<EdgeList> edgeListEigen = EigenCalculation.createEdgeList(eigenvectors, threshold);

	            
	            for (EdgeList edge : edgeListEigen) {
	                connector.createRelationshipConnectedGraph("eigendecomposedIndex", "created relationship in neo4j \n", edge);
	            }

	            
	        }
	        return "Create fully connected graph successful!";
	    } catch (Neo4jException e) {
	        throw new RuntimeException("Error creating laplacian graph in Neo4j: " + e.getMessage());
	    }
	}
	

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
	 * @param path
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
	 * @param path The path is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
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
	 * @param path - The path is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
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
	 * @param path
	 * @param target
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
	 * @param path
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
	 * @param path - The path is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
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
	 * @param path
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
	 * @param path - The path is composed of 3 parts, 1st-training dataset, 2nd-test dataset, 3rd- target attribute(as string)
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
	
    private static void displayMatrix(RealMatrix matrix, String matrixName) {
        System.out.println(matrixName + ": ");
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                System.out.printf("%.8f ", matrix.getEntry(row, col));
            }
            System.out.println();
        }
        System.out.println();
    }

	public static void main(String[] args) throws Exception {
		OutputDecisionTreeNeo4j outputDecisionTreeNeo4j = new OutputDecisionTreeNeo4j();
		//String dataPath = "D:/de/MASTER_THESIS/Decision-Tree-Neo4j/Java Plugin/DecisionTreePlugin/src/main/resources/test.csv";
		String Filename = "test.csv";
		String testDataPath = "data\\test.csv";
		//test
		ReadCsvTestData readCsvTestData = new ReadCsvTestData(testDataPath);
		ArrayList<TestData> testData = readCsvTestData.readCsvFile(testDataPath);
		Double[][] DistanceMatrix = readCsvTestData.euclidianDistance(testData);
		Double[] sigmas = ReadCsvTestData.calculateLocalSigmas(DistanceMatrix);
		Double[][] adj_mat = ReadCsvTestData.calculateAdjacencyMatrix(DistanceMatrix,sigmas);

		ArrayList<Nodelist> nodeList = ReadCsvTestData.getNodeList(testData);
		ArrayList<EdgeList> edgeList = ReadCsvTestData.calulateEdgeList(adj_mat);
		for (int i = 0; i < edgeList.size()-1; i++) {
			EdgeList edgeListDetail = edgeList.get(i);
			System.out.println("edgeListDetail " + edgeList.get(i));
			
			long bid = edgeListDetail.getTarget();
			String dtType = "connected";
			String bindex = "Index" + bid;
			double weightValue = (double)Math.round(edgeListDetail.getWeight() * 100000d) / 100000d;
			String weight = "`" + Double.toString(weightValue) + "` {value: " + weightValue + "`}" ;
			String finalString = "MATCH (a:" + dtType + ")S, (b:" + dtType + ") " +
					"WHERE a.id = "+"\"" +"Index"+edgeListDetail.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
					"CREATE (a)-[r:" + weight +  "]->(b)";
			System.out.println("--" + finalString);
		}
		
		
		
		int rows = adj_mat.length;
		int cols = adj_mat[0].length;

		double[][] newAdjMat = new double[rows][cols];
	
		for (int i = 0; i < rows; i++) {
		    for (int j = 0; j < cols; j++) {
		        newAdjMat[i][j] = adj_mat[i][j];
		    }
		}

		
		RealMatrix degreeMatrix = MatrixCalculation.calculateDegreeMatrix(newAdjMat);
		RealMatrix adjacencyMatrix = new BlockRealMatrix(newAdjMat);
		RealMatrix laplacianMatrix = MatrixCalculation.calculateSymmetricLaplacianMatrix(degreeMatrix, adjacencyMatrix);
		displayMatrix(adjacencyMatrix,"adjacencyMatrix");
		displayMatrix(degreeMatrix,"degreeMatrix");
		displayMatrix(laplacianMatrix,"laplacianMatrix");

        try {
        	EigenCalculation.EigenResult eigenResult = EigenCalculation.calculateEigen(laplacianMatrix);
            EigenCalculation.displayEigenResult(eigenResult);
            EigenCalculation.drawScatterPlot(eigenResult.eigenvalues);

            double threshold = 0.01;
            ArrayList<EdgeList> edgeListEigen = EigenCalculation.createEdgeList(eigenResult.eigenvectors, threshold);

//            System.out.println("Edge List:");
//            for (EdgeList edge : edgeListEigen) {
//                System.out.println("Source: " + edge.getSource() + ", Target: " + edge.getTarget() + ", Weight: " + edge.getWeight());
//            }
            
    		for (EdgeList edge : edgeListEigen) {
    			long bid = edge.getTarget();
    			String bindex = "Index" + bid;
    			String dtType = "laplacian";
    			double weightValue = (double)Math.round(edge.getWeight() * 100000d) / 100000d;
    			String weight = "`" + Double.toString(weightValue) + "` {value: " + weightValue + "`}" ;
    			String finalString = "MATCH (a:" + dtType + ")S, (b:" + dtType + ") " +
    					"WHERE a.id = "+"\"" +"Index"+ edge.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
    					"CREATE (a)-[r:" + weight +  "]->(b)";
    			System.out.println("--" + finalString);
    		}
        } catch (Exception e) {
            e.printStackTrace();
        }
		
	}
	
	public void connectNodes(final String nodeType, final String message, final String nodeCentroid, final String nodeCluster)
    {
    	final String name = "kmean";
        try ( Session session = driver.session() )
        {
            String greeting = session.writeTransaction( new TransactionWork<String>()
            {
                @Override
                public String execute( Transaction tx )
                {
                	//a is present for the node
            		Result result = tx.run( "MERGE (a"+":NodeType" +" {" + nodeCentroid +"}) " +
            				"MERGE (b "+":NodeType" + " {" + nodeCluster +"}) " +
            				"MERGE (a)-[:link]->(b) "
            				+ "RETURN a.message");
				    return result.single().get( 0 ).asString();
                }
            } );
        }
    }
	
	@UserFunction
    public String mapNodes(@Name("nodeType") String nodeType) throws Exception
   	{
       	String listOfData = "";
       	mapNodeList.clear();
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
       			        mapNodeList.add(valueOfNode);
       			    }
       				listOfData = listOfData + valueOfNode + " | ";
       			}
       		}
           }
       	return "Map all node data: " + listOfData;
   	}
	
	 /**
     * Procedure for k-means clustering and visualization in neo4j
     * @param nodeType type of node
     * @param numberOfCentroid 
     * @param numberOfInteration
     * @return cluster result and visualize
     * @throws Exception
     */
    @UserFunction
    @Description("Kmean clustering function")
	public String kmean(@Name("nodeType") String nodeType, @Name("numberOfCentroid") String numberOfCentroid, @Name("numberOfInteration") String numberOfInteration, @Name("distanceMeasure") String distanceMeasure) throws Exception
	{
    	try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
        {
			String kMeanAfterClustering = "";
			HashMap<String, ArrayList<String>> kmeanAssign = new HashMap<String, ArrayList<String>>();
			int numberOfCentroidInt = Integer.parseInt(numberOfCentroid);
			int numberOfInterationInt = Integer.parseInt(numberOfInteration);
			kmeanAssign = Unsupervised.KmeanClust(mapNodeList, numberOfCentroidInt, numberOfInterationInt, distanceMeasure);
			for (String centroid: kmeanAssign.keySet()) {
        		ArrayList<String> clusterNode = kmeanAssign.get(centroid);
        		for (String node : clusterNode)
        		{
        			connector.connectNodes(nodeType, "create relationship in kmean node",centroid,node);
        		}
    		    
    		}
	        return kMeanAfterClustering ;
		}
	}

	/**
	 * Procedure for dbscan clustering and visualization in neo4j
	 * @param nodeType type of node
	 * @param
	 * @param
	 * @return cluster result and visualize
	 * @throws Exception
	 */
	@UserFunction
	@Description("DBSCAN clustering function")
	public String dbscan(@Name("nodeType") String nodeType, @Name("epsilon") String epsilon, @Name("minimumPoints") String minimumPoints, @Name("distanceMeasure") String distanceMeasure) throws Exception
	{
		try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
		{
			String dbscanAfterClustering = "";
			HashMap<String,ArrayList<String>> dbAssign = new HashMap<String,ArrayList<String>>();
			double eps = Double.parseDouble(epsilon);
			int minPts = Integer.parseInt(minimumPoints);
			dbAssign = Unsupervised.DbClust(mapNodeList, eps, minPts, distanceMeasure);
			for (String centroid: dbAssign.keySet()) {
        		ArrayList<String> clusterNode = dbAssign.get(centroid);
        		for (String node : clusterNode)
        		{
        			connector.connectNodes(nodeType, "create relationship in dbscan node",centroid,node);
        		}
    		    
    		}
			return dbscanAfterClustering ;
		}
	}


}
