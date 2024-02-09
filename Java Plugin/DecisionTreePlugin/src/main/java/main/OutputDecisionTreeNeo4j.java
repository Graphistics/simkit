package main;
import static org.neo4j.driver.Values.parameters;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

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
import org.neo4j.driver.util.Pair;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.UserFunction;

import cv.CrossValidation;
import definition.EdgeList2;
import definition.NodeList2;
import eigendecomposed.EigenCalculation;
import eigendecomposed.MatrixCalculation;
import evaluate.EvaluateTree;
import gainratio.EvaluateTreeGR;
import gini.EvaluateTreeGI;
import global.Neo4jGraphHandler;
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

	/**
	 * Creates a Laplacian Eigen Transform graph based on Laplacian matrix type and specified number of eigenvectors.
	 *
	 * @param node_label            The label of the nodes in the graph.
	 * @param laplacian_type        The type of Laplacian matrix to be used.
	 * @param number_of_eigenvectors The number of desired eigenvectors to compute during eigen decomposition.
	 * @return String indicating the success of the graph creation and create Graph with Nodes and Relationships in Neo4j.
	 * @throws Exception.
	 */
	@UserFunction
	public String createLaplacianEigenTransformGraph(@Name("node_label") String node_label,  @Name("laplacian_type") String laplacian_type, @Name("number_of_eigenvectors") Double number_of_eigenvectors) throws Exception {
	    try (OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j("bolt://localhost:7687", "neo4j", "123412345")) {
	        if (node_label == null) {
	            return "Missing node label";
	        } else {
	            ArrayList<EdgeList2> edge_list = new ArrayList<>();
	            edge_list = Neo4jGraphHandler.retrieveEdgeListFromNeo4j(node_label, connector.getDriver());

	            double[][] adjacency_matrix_data = MatrixCalculation.convertToAdjacencyMatrix(edge_list);
	            RealMatrix adjacency_matrix = new BlockRealMatrix(adjacency_matrix_data);
	            RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix_data);
	            RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, laplacian_type);

	            EigenCalculation.EigenResult eigen_result = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);
	            ArrayList<NodeList2> node_list_eigen = Neo4jGraphHandler.retrieveNodeListFromNeo4j(node_label, connector.getDriver());
	            ArrayList<EdgeList2> edge_list_eigen = EigenCalculation.createEdgeList(node_list_eigen, eigen_result.X, edge_list);

	            
	            String graph_name = "eigendecomposedGraph_" + laplacian_type + "_" + node_label + "_" + Math.round(number_of_eigenvectors);
	            
	            for (NodeList2 node : node_list_eigen) {
	            	Neo4jGraphHandler.createNodeGraphEigenTransform(graph_name, "created nodes in neo4j", node, eigen_result.X, connector.getDriver());
	            }

//	            for (EdgeList2 edge : edge_list_eigen) {
	            for (int i = 0; i < edge_list_eigen.size(); i++) {
					EdgeList2 edge_list_detail = edge_list_eigen.get(i);
//					if(edge_list_detail.getWeight()==0.0){
//						continue;
//					}
	            	Neo4jGraphHandler.createRelationshipGraph(graph_name, "created relationship in neo4j \n", edge_list_detail, connector.getDriver());
	            }
	        }
	        return "Create eigendecomposed graph successful!";
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
    public String displayGraphList(@Name("node_label") String node_label, @Name("numberOfEigenvectors") Double number_of_eigenvectors) throws Exception {
        try (OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j("bolt://localhost:7687", "neo4j", "123412345")) {
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
                double[][] adjacency_matrix_data = MatrixCalculation.convertToAdjacencyMatrix(edge_list);
                RealMatrix adjacency_matrix = new BlockRealMatrix(adjacency_matrix_data);
                outputString.append("\n\nAdjacency Matrix:\n").append(matrixToString(adjacency_matrix));

                // Display degree matrix
                RealMatrix degree_matrix = MatrixCalculation.calculateDegreeMatrix(adjacency_matrix_data);
                outputString.append("\n\nDegree Matrix:\n").append(matrixToString(degree_matrix));

                // Display Laplacian matrix
                RealMatrix laplacian_matrix = MatrixCalculation.calculateLaplacianMatrix(degree_matrix, adjacency_matrix, "sym");
                outputString.append("\n\nLaplacian Matrix:\n").append(matrixToString(laplacian_matrix));

                // Display Eigenvalues, Eigenvectors, and X
                EigenCalculation.EigenResult eigen_result = EigenCalculation.calculateEigen(laplacian_matrix, number_of_eigenvectors);
                EigenCalculation.displayEigenResult(eigen_result);

                return outputString.toString();
            }
        } catch (Neo4jException e) {
            throw new RuntimeException("Error displaying edge list in Neo4j: " + e.getMessage());
        }
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
                	//a is present for the node
            		Result result = tx.run( "MERGE (a"+":ClusteringNodeType" +" {" + nodeCentroid +"}) " +
            				"MERGE (b "+":ClusteringNodeType" + " {" + nodeCluster +"}) " +
            				"MERGE (a)-[r:link]->(b) "  +
                            "SET r.distance = " + distance + " " + 
            				"RETURN a.message");
				    return result.single().get( 0 ).asString();
                }
            } );
        }
    }
	
	@UserFunction
	public String mapNodes(@Name("nodeSet") String nodeSet, @Name("overlook") String overLook) throws Exception {
	    String listOfData = "";
	    String[] overLookArray = new String[0];
	    mapNodeList.clear();
	    try (OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j("bolt://localhost:7687", "neo4j", "123412345")) {
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
	
	 /**
     * Procedure for k-means clustering and visualization in neo4j
     * @param nodeSet type of node
     * @param numberOfCentroid 
     * @param numberOfInteration
     * @return cluster result and visualize
     * @throws Exception
     */
    @UserFunction
    @Description("Kmean clustering function")
	public String kmean(@Name("nodeSet") String nodeSet, @Name("numberOfCentroid") String numberOfCentroid, @Name("numberOfInteration") String numberOfInteration, @Name("distanceMeasure") String distanceMeasure) throws Exception
	{
    	try ( OutputDecisionTreeNeo4j connector = new OutputDecisionTreeNeo4j( "bolt://localhost:7687", "neo4j", "123412345" ) )
        {
			String averageSilhouetteCoefficientString = "The average Silhouette Coefficient value is: ";
			HashMap<String, ArrayList<String>> kmeanAssign = new HashMap<String, ArrayList<String>>();
			int numberOfCentroidInt = Integer.parseInt(numberOfCentroid);
			int numberOfInterationInt = Integer.parseInt(numberOfInteration);
			
			kmeanAssign = Unsupervised.KmeanClust(mapNodeList, numberOfCentroidInt, numberOfInterationInt, distanceMeasure);
			double averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(kmeanAssign, distanceMeasure);
	        
			for (String centroid : kmeanAssign.keySet()) {
			    ArrayList<String> clusterNode = kmeanAssign.get(centroid);
			    for (String node : clusterNode) {
			    	DecimalFormat decimalFormat = new DecimalFormat("#.###");
			        double distance = Unsupervised.calculateDistance(node, centroid, distanceMeasure);
			        String formattedDistance = decimalFormat.format(distance);
			        double roundedDistance = Double.parseDouble(formattedDistance);
			        connector.connectNodes(nodeSet, "create relationship in kmean node", centroid, node, roundedDistance);
			    }
			}

			return averageSilhouetteCoefficientString + averageSilhouetteCoefficientValue ;
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
			kmeanAssign = Unsupervised.KmeanClust(mapNodeList, numberOfCentroidInt, numberOfInterationInt, distanceMeasure);
			double averageSilhouetteCoefficientValue = Unsupervised.averageSilhouetteCoefficient(kmeanAssign, distanceMeasure);
	        return averageSilhouetteCoefficientString + averageSilhouetteCoefficientValue ;
		}
    	else
    	{
    		return null;
    	}
	}

	public static void main(String[] args) throws Exception {
//		OutputDecisionTreeNeo4j outputDecisionTreeNeo4j = new OutputDecisionTreeNeo4j();
//		//String dataPath = "D:/de/MASTER_THESIS/Decision-Tree-Neo4j/Java Plugin/DecisionTreePlugin/src/main/resources/test.csv";
//		String Filename = "test.csv";
//		String testDataPath = "data\\test.csv";
//		//test
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
//		}
//		
		
//		
//		int rows = adj_mat.length;
//		int cols = adj_mat[0].length;
//
//		double[][] newAdjMat = new double[rows][cols];
//	
//		for (int i = 0; i < rows; i++) {
//		    for (int j = 0; j < cols; j++) {
//		        newAdjMat[i][j] = adj_mat[i][j];
//		    }
//		}
//
//		double[][] adjmatrxiEigen = MatrixCalculation.convertToAdjacencyMatrix(edgeList);
//		RealMatrix degreeMatrix = MatrixCalculation.calculateDegreeMatrix(newAdjMat);
//		RealMatrix adjacencyMatrix = new BlockRealMatrix(adjmatrxiEigen);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateLaplacianMatrix(degreeMatrix, adjacencyMatrix,"SYMMETRIC");
//		displayMatrix(adjacencyMatrix,"adjacencyMatrix");
//		displayMatrix(degreeMatrix,"degreeMatrix");
//		displayMatrix(laplacianMatrix,"laplacianMatrix");
//
//        try {
//        	EigenCalculation.EigenResult eigenResult = EigenCalculation.calculateEigen(laplacianMatrix);
//            EigenCalculation.displayEigenResult(eigenResult);
////            EigenCalculation.drawScatterPlot(eigenResult.eigenvalues);
//
//            double threshold = 0.01;
//            ArrayList<EdgeList> edgeListEigen = EigenCalculation.createEdgeList(eigenResult.eigenvectors, threshold);
//            
//    		for (EdgeList edge : edgeListEigen) {
//    			long bid = edge.getTarget();
//    			String bindex = "Index" + bid;
//    			String dtType = "eigendecomposedGraph";
//    			double weightValue = (double)Math.round(edge.getWeight() * 100000d) / 100000d;
//    			String weight = "`" + Double.toString(weightValue) + "` {value: " + weightValue + "}" ;
//    			String finalString = "MATCH (a:" + dtType + "), (b:" + dtType + ") " + 
//    					"WHERE a.id = "+"\"" +"Index"+ edge.getSource() +  "\""+" AND "+ "b.id ="+ "\""+bindex+"\""+" "+
//    					"CREATE (a)-[r:" + weight +  "]->(b)";
//    			System.out.println(finalString);
//    		}
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
		
	}



}
