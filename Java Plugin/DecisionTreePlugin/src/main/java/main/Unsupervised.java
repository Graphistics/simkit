package main;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import scala.util.Random;

/**
 * @author 49171
 *
 */
public class Unsupervised {
	
	public static ArrayList<String> dummyData(){
		ArrayList<String> data = new ArrayList<String>();
		String line1 ="y_cordinate:7.0,x_cordinate:1.0";
		String line2 ="y_cordinate:6.0,x_cordinate:1.0";
		String line3 ="y_cordinate:2.0, x_cordinate:6.0";
		String line4 ="y_cordinate:1.0, x_cordinate:8.0";
		String line5 ="y_cordinate:2.0, x_cordinate:10.0";
//		String line1 ="y_cordinate:0.536804,x_cordinate:0.449056";
//		String line2 ="y_cordinate:0.533213,x_cordinate:0.454964";
//		String line3 ="y_cordinate:-0.247063,x_cordinate:0.474114";
//		String line4 ="y_cordinate:-0.448465,x_cordinate:0.444607";
//		String line5 ="y_cordinate:-0.406650, x_cordinate:0.410971";
		data.add(line1);
		data.add(line2);
		data.add(line3);
		data.add(line4);
		data.add(line5);
		return data;
	}
	
	public static void main(String[] args)
	{
		  ArrayList<String> inputData = dummyData(); 
		  HashMap<String, ArrayList<String>> dbAssign = KmeanClust(inputData, 2, 20, "Euclidean");		  
		  for (String centroid: dbAssign.keySet()) {
			  System.out.println("1");
      		ArrayList<String> clusterNode = dbAssign.get(centroid);
      		System.out.println("number of centroid " +  centroid);
      		System.out.println(clusterNode.toString());
  		  }
		  
		  Double silhouetteValue = averageSilhouetteCoefficient(dbAssign, "Euclidean");
		  System.out.println(averageSilhouetteCoefficient(dbAssign, "Euclidean"));

	}
	
	/**
	-------------------------------------------------------------------------------DBSCAN Clustering ------------------------------------------------------------------------------------------
	*/
	
	/**
     * Determines the neighbours of a given input value.
     * 
     * @param inputValue value for which neighbours are to be calculated
     * @return list of neighbours
     */
    public static ArrayList<String> getNeighbours(final String inputValue, final String distanceMetric) 
    {
        ArrayList<String> neighbours = new ArrayList<String>();
        double distance = 0;
        
        for(int i=0; i<inputValues.size(); i++) {
        	
            String candidate = inputValues.get(i);
            
            if(distanceMetric.equals("bray-curtis"))
            {
            	distance = calBrayCurtis(inputValue, candidate);
            }
            else if(distanceMetric.equals("manhattan"))
            {
            	distance = calManhattanDist(inputValue, candidate);
            }
            else if(distanceMetric.equals("cosine"))
            {
            	distance = calCosineSimilarity(inputValue, candidate);
            }
            else
            {
            	distance = calEuclideanDist(inputValue, candidate);
            }
            System.out.println(distance);
            if (distance <= epsilon) {
                neighbours.add(candidate);
            }
        }
        return neighbours;
    }
    
    /**
     * Merges the elements of the right collection to the left one and returns
     * the combination.
     * 
     * @param neighbours1 left collection
     * @param neighbours2 right collection
     * @return Modified left collection
     */
    public static ArrayList<String> mergeRightToLeftCollection(final ArrayList<String> neighbours1,final ArrayList<String> neighbours2) 
    {
        for (int i = 0; i < neighbours2.size(); i++) {
            String tempPt = neighbours2.get(i);
            if (!neighbours1.contains(tempPt)) {
                neighbours1.add(tempPt);
            }
        }
        return neighbours1;
    }
    
    /**
     * Applies the clustering and returns a collection of clusters (i.e. a list
     * of lists of the respective cluster members).
     * 
     * @return
     */
    
	public static double epsilon;
	public static ArrayList<String> inputValues;
	
    public static HashMap<String, ArrayList<String>> DbClust(final ArrayList<String> inputData, double eps, int minPts, String distanceMetric )
    {

    	inputValues = inputData;
    	epsilon = eps;
    	
    	HashMap<String, ArrayList<String>> resultHashmap = new HashMap<String, ArrayList<String>>();
        ArrayList<ArrayList<String>> resultList = new ArrayList<ArrayList<String>>();
        ArrayList<String> visitedPoints = new ArrayList<String>();
		visitedPoints.clear();

        ArrayList<String> neighbours;
        int index = 0;
        while (inputValues.size() > index) {
            String p = inputValues.get(index);
            if (!visitedPoints.contains(p)) {
                visitedPoints.add(p);
                neighbours = getNeighbours(p , distanceMetric);
                if (neighbours.size() >= minPts) {
                    int ind = 0;
                    while (neighbours.size() > ind) {
                        String r = neighbours.get(ind);
                        if (!visitedPoints.contains(r)) {
                            visitedPoints.add(r);
                            ArrayList<String> individualNeighbours = getNeighbours(r, distanceMetric);
                            if (individualNeighbours.size() >= minPts) {
                                neighbours = mergeRightToLeftCollection(neighbours,individualNeighbours);
                            }
                        }
                        ind++;
                    }
                    resultList.add(neighbours);
                }
            }
            index++;
        }
        for (ArrayList<String> cluster : resultList)
        {
        	String newCentroid = calculateNewCentroid(cluster);
        	resultHashmap.put(newCentroid, cluster);
        }
        return resultHashmap;
    }
    
    
	/**
	-------------------------------------------------------------------------------K-MEANS Clustering ------------------------------------------------------------------------------------------
	*/
	 
	/**
	 * This is the main method to perform k-means clustering.
	 * @param inputData is a variable where the nodes from Neo4j are stored
	 * @param numberOfCentroids store the number of centroids specified by user for clustering
	 * @param numberOfInteration saves user specified iteration to find convergence
	 * @return
	 */
	public static HashMap<String, ArrayList<String>> KmeanClust (ArrayList<String> inputData, int numberOfCentroids, int numberOfInteration, String distanceMeasure)
	{
		HashMap<String, ArrayList<String>> kmeanAssign = new HashMap<String, ArrayList<String>>();
		ArrayList<String> listOfCentroid = new ArrayList<String>();
		ArrayList<String> listOfRemain = new ArrayList<String>(inputData);
		
		// Initializing centroid by random choice
	    java.util.Random rand = new java.util.Random();

	    while (listOfCentroid.size() < numberOfCentroids) {
	        int randomIndex = rand.nextInt(inputData.size());
	        String potentialCentroid = inputData.get(randomIndex);

	        if (!listOfCentroid.contains(potentialCentroid)) {
	        	listOfCentroid.add(potentialCentroid);
	        }
	    }
	    for(int i = 0; i < listOfRemain.size(); i ++)
	    {
	    	if(listOfCentroid.contains(listOfRemain.get(i)));
	    	{
	    		listOfRemain.remove(i);
	    	}
	    }
	    System.out.println("---------------- numberOfCentroids : " + listOfCentroid.size());
	    
		// First clusters
		HashMap<String, ArrayList<String>> hashClusterAssign = distanceAssign(listOfCentroid,listOfRemain, distanceMeasure);
		// All iterations
		kmeanAssign = kmeanIteration(hashClusterAssign,numberOfInteration,inputData, distanceMeasure);
		return kmeanAssign;
	}
	
	/**
	 * Method to perform the iterations of k-means
	 * @param clusterAssign contains the first cluster assignments
	 * @param numberOfInteration specified by user
	 * @param inputData specified by user
	 * @return
	 */
	public static HashMap<String, ArrayList<String>> kmeanIteration(HashMap<String, ArrayList<String>> clusterAssign, int numberOfInteration, ArrayList<String> inputData, String distanceMeasure) {
	    for (int i = 0; i < numberOfInteration; i++) {
	        // Calculate new centroids and update clusterAssign
	        clusterAssign = calculateAndUpdateCentroids(clusterAssign);

	        // Perform distance assignment again with the updated centroids
	        clusterAssign = distanceAssign(new ArrayList<>(clusterAssign.keySet()), inputData, distanceMeasure);
	    }
	    return clusterAssign;
	}
	
	/**
	 * Method to calculate new centroid points after each iteration
	 * @param listOfNodesInCluster nodes assigned to each cluster
	 * @return returns new centroids after each iteration
	 */
	public static String calculateNewCentroid(ArrayList<String> listOfNodesInCluster) {
	    if (listOfNodesInCluster.isEmpty()) {
	        return null; // Handle empty cluster case
	    }

	    // Extract attribute names from the first node
	    String[] firstNodeAttributes = listOfNodesInCluster.get(0).split(",");
	    String[] attributeNames = new String[firstNodeAttributes.length];
	    for (int i = 0; i < firstNodeAttributes.length; i++) {
	        attributeNames[i] = firstNodeAttributes[i].split(":")[0].trim();
	    }

	    double[] attributeSums = new double[attributeNames.length];

	    for (String node : listOfNodesInCluster) {
	        String[] attributes = node.split(",");
	        for (int i = 0; i < attributes.length; i++) {
	            try {
	                attributeSums[i] += Double.parseDouble(attributes[i].split(":")[1].trim());
	            } catch (NumberFormatException e) {
	                System.err.println("Error parsing number: " + attributes[i]);
	                return null; // or handle the error appropriately
	            }
	        }
	    }

	    StringBuilder newCentroid = new StringBuilder();
	    for (int i = 0; i < attributeSums.length; i++) {
	        double average = attributeSums[i] / listOfNodesInCluster.size();
	        newCentroid.append(attributeNames[i]).append(":").append(average);
	        if (i < attributeSums.length - 1) {
	            newCentroid.append(",");
	        }
	    }
	    return newCentroid.toString();
	}
	
	/**
	 * Method to calculate and update centroids in the clusterAssign.
	 *
	 * @param clusterAssign The current cluster assignments
	 * @return Updated cluster assignments with new centroids
	 */
	private static HashMap<String, ArrayList<String>> calculateAndUpdateCentroids(HashMap<String, ArrayList<String>> clusterAssign) {
	    HashMap<String, ArrayList<String>> updatedClusterAssign = new HashMap<>();

	    for (String key : clusterAssign.keySet()) {
	        ArrayList<String> clusterNodes = clusterAssign.get(key);
	        String newCentroid = calculateNewCentroid(clusterNodes);
	        if (newCentroid != null) { // Check for null to handle empty clusters
	            updatedClusterAssign.put(newCentroid, clusterNodes);
	        }
	    }
	    return updatedClusterAssign;
	}
	
	public static HashMap<String, ArrayList<String>> distanceAssign (ArrayList<String> listOfCentroid, ArrayList<String> listOfRemain, String distanceMeasure) {
	    HashMap<String, ArrayList<String>> hashClusterAssign = new HashMap<String, ArrayList<String>>();
	    // Calculate distance and assign points to clusters
	    for (int i = 0; i < listOfRemain.size(); i++) {
	        double minDistance = Double.MAX_VALUE;
	        String closestCentroid = null;

	        for (int j = 0; j < listOfCentroid.size(); j++) {
	            double distance = calculateDistance(listOfRemain.get(i), listOfCentroid.get(j), distanceMeasure);
	            
	            if (distance < minDistance) {
	                minDistance = distance;
	                closestCentroid = listOfCentroid.get(j);
	            }
	        }
	        hashClusterAssign.computeIfAbsent(closestCentroid, k -> new ArrayList<>()).add(listOfRemain.get(i));
	    }
	    // Ensure all centroids are in the hashmap, even if they have no assigned points
	    for (String centroid : listOfCentroid) {
	        hashClusterAssign.putIfAbsent(centroid, new ArrayList<>());
	    }

	    return hashClusterAssign;
	}

	private static double calculateDistance(String point1, String point2, String distanceMeasure) {
	    switch (distanceMeasure.toLowerCase()) {
	        case "manhattan":
	            return calManhattanDist(point1, point2);
	        case "cosine":
	            return calCosineSimilarity(point1, point2);
	        case "bray-curtis":
	            return calBrayCurtis(point1, point2);
	        default:
	            return calEuclideanDist(point1, point2);
	    }
	}
	
	
	/**
	--------------------------------------------------------------------------Distance Measures for Clustering ------------------------------------------------------------------------------------------
	*/
	
	/**
	 * Euclidean distance calculation from point A to point B
	 * @param start point A
	 * @param end point B
	 * @return
	 */
	public static double calEuclideanDist (String start, String end)
	{
		double distance = 0.00;
		String[] startSplit =  start.split(",");
		String[] endSplit = end.split(",");
		for(int i = 0; i < startSplit.length; i++)
		{
			float startValue = Float.parseFloat(startSplit[i].split(":")[1]);
			float endValue = Float.parseFloat(endSplit[i].split(":")[1]);
			distance = distance + Math.pow((startValue-endValue),2);
		}
		distance = Math.sqrt(distance);
		return distance;
		
	}
	/**
	 * Calculate Manhattan distance between point A and B
	 * @param start point A
	 * @param end point B
	 * @return
	 */
	public static double calManhattanDist (String start, String end)
	{
		double distance = 0.00;
		String[] startSplit =  start.split(",");
		String[] endSplit = end.split(",");
		
		for(int i = 0; i < startSplit.length; i++)
		{
			float startValue = Float.parseFloat(startSplit[i].split(":")[1]);
			float endValue = Float.parseFloat(endSplit[i].split(":")[1]);
			distance = distance + Math.abs(startValue - endValue);
		}
		return distance;
		
	}
	/**
	 * Calculate Cosine similarity between point A and B
	 * @param start point A
	 * @param end point B
	 * @return
	 */
	public static double calCosineSimilarity (String start, String end)
	{
		double distance = 0.00;
		double dotProduct = 0.00;
		double normA = 0.00;
		double normB = 0.00;
		
		String[] startSplit =  start.split(",");
		String[] endSplit = end.split(",");
		for(int i = 0; i < startSplit.length; i++)
		{
			float startValue = Float.parseFloat(startSplit[i].split(":")[1]);
			float endValue = Float.parseFloat(endSplit[i].split(":")[1]);
			dotProduct += startValue * endValue;
			normA += Math.pow(startValue, 2);
			normB += Math.pow(endValue, 2);
		}
		distance = dotProduct/ (Math.sqrt(normA) * Math.sqrt(normB));
		return distance;
		
	}
	/**
	 * Calculate Bray-Curtis dissimilarity between point A and B
	 * @param start point A
	 * @param end point B
	 * @return
	 */
	public static double calBrayCurtis (String start, String end)
	{
		double distance = 0.00;
		double num = 0.00;
		double den = 0.00;
		
		String[] startSplit =  start.split(",");
		String[] endSplit = end.split(",");
		for(int i = 0; i < startSplit.length; i++)
		{
			float startValue = Float.parseFloat(startSplit[i].split(":")[1]);
			float endValue = Float.parseFloat(endSplit[i].split(":")[1]);
			num = num + Math.abs(startValue - endValue);
			den = den + Math.abs(startValue + endValue);
		}
		distance = num/den;
		return distance;
		
	}
	
	//a(i) : Calculate the average distance of point i to other points in its cluster.
	public static double averageIntraClusterDistance(String point, ArrayList<String> cluster, String distanceMeasure) {
	    double sumDistance = 0.0;
	    for (String otherPoint : cluster) {
	        sumDistance += calculateDistance(point, otherPoint, distanceMeasure );
	    }
	    return sumDistance / (cluster.size() - 1); // Exclude the point itself
	}
	
	//b(i) : Calculate the smallest average distance of point i to all points in other clusters.
	public static double smallestInterClusterDistance(String point, HashMap<String, ArrayList<String>> allCluster,
			ArrayList<String> ownCluster, String distanceMeasure) {
		
	    double smallestAverage = Double.MAX_VALUE;
    	for (String key : allCluster.keySet())
    	{
    		ArrayList<String> cluster = allCluster.get(key);
    		if(cluster.equals(ownCluster)) continue; // Skip the same cluster contain the point
    		double sumDistance = 0.0;
    		for(String otherPoint : cluster) {
    			sumDistance += calculateDistance(point, otherPoint, distanceMeasure);
    		}
    		double averageDistance = sumDistance / cluster.size();
            if (averageDistance < smallestAverage) {
                smallestAverage = averageDistance;
    	        }
    	}
    	return smallestAverage;
	}
	
	//Calculate the Silhouette Coefficient for Each Point
	public static double silhouetteCoefficient(String point, ArrayList<String> ownCluster
			, HashMap<String, ArrayList<String>> allClusters, String distanceMeasure) {
		
	    double a = averageIntraClusterDistance(point, ownCluster, distanceMeasure);
	    double b = smallestInterClusterDistance(point, allClusters, ownCluster, distanceMeasure);
	    return (b - a) / Math.max(a, b);
	}
	
	//Calculate the SilhouetteCoefficient : Calculate the mean of the Silhouette Coefficients for all point
	public static double averageSilhouetteCoefficient(HashMap<String, ArrayList<String>> allCluster, String distanceMeasure) {
	    double sumSilhouette = 0.0;
	    int numPoints = 0;

	    for (String key : allCluster.keySet())
	    {
	    	ArrayList<String> cluster = allCluster.get(key);
	    	for (String point : cluster)
	    	{
	    		sumSilhouette += silhouetteCoefficient(point, cluster, allCluster, distanceMeasure);
	    		numPoints++;
	    	}
	    }
	    System.out.println("Sum " + sumSilhouette);
	    return sumSilhouette / numPoints;
	}
	
}
