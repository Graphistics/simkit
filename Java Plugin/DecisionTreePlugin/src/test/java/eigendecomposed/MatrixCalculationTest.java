//package eigendecomposed;
//
//import static org.junit.jupiter.api.Assertions.*;
//
//import java.io.IOException;
//
//import org.apache.commons.math4.legacy.linear.BlockRealMatrix;
//import org.apache.commons.math4.legacy.linear.RealMatrix;
//import org.junit.jupiter.api.Test;
//
//public class MatrixCalculationTest {
//	
//	double[][] adj_mat_fully = new double[][]{
//        {0., 0.97815581, 0.17071378, 0.00462688, 0.0235727},
//        {0.97815581, 0., 0.20173889, 0.00569376, 0.02265865},
//        {0.17071378, 0.20173889, 0., 0.57177084, 0.36787944},
//        {0.00462688, 0.00569376, 0.57177084, 0., 0.57177084},
//        {0.0235727 , 0.02265865, 0.36787944, 0.57177084, 0.}};
//	double[][] adj_mat_epsilon = new double[][]{
//        {0., 1., 0., 0., 0.},
//        {1., 0., 0., 0., 0.},
//        {0., 0., 0., 1., 1.},
//        {0., 0., 1., 0., 1.},
//        {0., 0., 1., 1., 0.}};
//	double[][] adj_mat_knn = new double[][]{
//        {0., 1., 1., 0., 0.},
//        {1., 0., 1., 0., 0.},
//        {0., 0., 0., 1., 1.},
//        {0., 0., 1., 0., 1.},
//        {0., 0., 1., 1., 0.}};;
//	double[][] adj_mat_mknn = new double[][]{
//		{0., 1., 0., 0., 0.},
//        {1., 0., 0., 0., 0.},
//        {0., 0., 0., 1., 1.},
//        {0., 0., 1., 0., 1.},
//        {0., 0., 1., 1., 0.}};;
//	double[][] degree_mat_fully = new double[][]{
//        {1.17706917, 0.        , 0.        , 0.        , 0.        },
//        {0.        , 1.20824711, 0.        , 0.        , 0.        },
//        {0.        , 0.        , 1.31210294, 0.        , 0.        },
//        {0.        , 0.        , 0.        , 1.15386233, 0.        },
//        {0.        , 0.        , 0.        , 0.        , 0.98588163}};;
//	double[][] degree_mat_epsilon = new double[][]{
//        {1., 0., 0., 0., 0.},
//        {0., 1., 0., 0., 0.},
//        {0., 0., 2., 0., 0.},
//        {0., 0., 0., 2., 0.},
//        {0., 0., 0., 0., 2.}};;
//	double[][] degree_mat_knn = new double[][]{
//        {1., 0., 0., 0., 0.},
//        {0., 1., 0., 0., 0.},
//        {0., 0., 4., 0., 0.},
//        {0., 0., 0., 2., 0.},
//        {0., 0., 0., 0., 2.}};;
//	double[][] degree_mat_mknn = new double[][]{
//        {1., 0., 0., 0., 0.},
//        {0., 1., 0., 0., 0.},
//        {0., 0., 2., 0., 0.},
//        {0., 0., 0., 2., 0.},
//        {0., 0., 0., 0., 2.}};;
//        
//    MatrixCalculationTest() throws IOException {
//    }
//	
//	@Test
//	public void calculateDegreeMatrixFully() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_fully);
//		RealMatrix expected_degree_mat= new BlockRealMatrix(degree_mat_fully);
//		RealMatrix degreeMatrix = MatrixCalculation.calculateDegreeMatrix(adj_mat_data);
//		assertTrue(expected_degree_mat.equals(degreeMatrix));
//	}
//	
//	@Test
//	public void calculateSymmetricLaplacianMatrixFully() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_fully);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_fully);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateSymmetricLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{
//            {0.        , 0.82021775, 0.13736738, 0.00397018, 0.02188245},
//            {0.82021775, 0.        , 0.16022408, 0.00482219, 0.02076078},
//            {0.13736738, 0.16022408, 0.        , 0.46468757, 0.3234515},
//            {0.00397018, 0.00482219, 0.46468757, 0.        , 0.53608372},
//            {0.02188245, 0.02076078, 0.3234515 , 0.53608372, 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//        try{
//        	assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//        }catch (AssertionError ex) {
//            //Do Something
//            throw(ex);
//        }
//		
//	}
//	
//	@Test
//	public void calculateRandomWalkLaplacianMatrixFully() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_fully);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_fully);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateRandomWalkLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{
//            {0.        , 0.83100963, 0.14503292, 0.00393085, 0.02002661},
//            {0.80956603, 0.        , 0.16696823, 0.00471241, 0.01875332},
//            {0.130107  , 0.15375233, 0.        , 0.43576675, 0.28037392},
//            {0.00400991, 0.00493452, 0.49552778, 0.        , 0.49552778},
//            {0.02391028, 0.02298313, 0.37314768, 0.57995892, 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//		
//		assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//	}
//	
//	@Test
//	public void calculateDegreeMatrixEpsilon() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_epsilon);
//		RealMatrix expected_degree_mat= new BlockRealMatrix(degree_mat_epsilon);
//		RealMatrix degreeMatrix = MatrixCalculation.calculateDegreeMatrix(adj_mat_data);
//		assertTrue(expected_degree_mat.equals(degreeMatrix));
//	}
//	
//	@Test
//	public void calculateSymmetricLaplacianMatrixEpsilon() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_epsilon);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_epsilon);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateSymmetricLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{
//            {0. , 1. , 0. , 0. , 0.},
//            {1. , 0. , 0. , 0. , 0.},
//            {0. , 0. , 0. , 0.5, 0.5},
//            {0. , 0. , 0.5, 0. , 0.5},
//            {0. , 0. , 0.5, 0.5, 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//		
//		assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//	}
//	
//	@Test
//	public void calculateRandomWalkLaplacianMatrixEpsilon() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_epsilon);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_epsilon);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateRandomWalkLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{
//            {0. , 1. , 0. , 0. , 0.},
//            {1. , 0. , 0. , 0. , 0.},
//            {0. , 0. , 0. , 0.5, 0.5},
//            {0. , 0. , 0.5, 0. , 0.5},
//            {0. , 0. , 0.5, 0.5, 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//		
//		assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//	}
//	
//	@Test
//	public void calculateDegreeMatrixKnn() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_knn);
//		RealMatrix expected_degree_mat= new BlockRealMatrix(degree_mat_knn);
//		RealMatrix degreeMatrix = MatrixCalculation.calculateDegreeMatrix(adj_mat_data);
//		assertTrue(expected_degree_mat.equals(degreeMatrix));
//	}
//	
//	@Test
//	public void calculateSymmetricLaplacianMatrixKnn() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_knn);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_knn);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateSymmetricLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{
//            {0.        , 1.        , 0.5       , 0.        , 0.},
//            {1.        , 0.        , 0.5       , 0.        , 0.},
//            {0.        , 0.        , 0.        , 0.35355339, 0.35355339},
//            {0.        , 0.        , 0.35355339, 0.        , 0.5},
//            {0.        , 0.        , 0.35355339, 0.5       , 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//		
//		assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//	}
//	
//	@Test
//	public void calculateRandomWalkLaplacianMatrixKnn() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_knn);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_knn);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateRandomWalkLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{
//            {0.  , 1.  , 1.  , 0.  , 0.},
//            {1.  , 0.  , 1.  , 0.  , 0.},
//            {0.  , 0.  , 0.  , 0.25, 0.25},
//            {0.  , 0.  , 0.5 , 0.  , 0.5},
//            {0.  , 0.  , 0.5 , 0.5 , 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//		
//		assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//	}
//	
//	@Test
//	public void calculateDegreeMatrixMknn() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_mknn);
//		RealMatrix expected_degree_mat= new BlockRealMatrix(degree_mat_mknn);
//		RealMatrix degreeMatrix = MatrixCalculation.calculateDegreeMatrix(adj_mat_data);
//		assertTrue(expected_degree_mat.equals(degreeMatrix));
//	}
//	
//	@Test
//	public void calculateSymmetricLaplacianMatrixMknn() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_mknn);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_mknn);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateSymmetricLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{ //change
//            {0. , 1. , 0. , 0. , 0.},
//            {1. , 0. , 0. , 0. , 0.},
//            {0. , 0. , 0. , 0.5, 0.5},
//            {0. , 0. , 0.5, 0. , 0.5},
//            {0. , 0. , 0.5, 0.5, 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//		
//		assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//	}
//	
//	@Test
//	public void calculateRandomWalkLaplacianMatrixMknn() {
//		RealMatrix adj_mat_data = new BlockRealMatrix(adj_mat_mknn);
//		RealMatrix degree_mat= new BlockRealMatrix(degree_mat_mknn);
//		RealMatrix laplacianMatrix = MatrixCalculation.calculateRandomWalkLaplacianMatrix(degree_mat, adj_mat_data);
//		
//        double[][] expected_laplacian = new double[][]{
//            {0. , 1. , 0. , 0. , 0.},
//            {1. , 0. , 0. , 0. , 0.},
//            {0. , 0. , 0. , 0.5, 0.5},
//            {0. , 0. , 0.5, 0. , 0.5},
//            {0. , 0. , 0.5, 0.5, 0.}};
//            
//        RealMatrix expected_laplacian_mat= new BlockRealMatrix(expected_laplacian);
//		
//		assertTrue(expected_laplacian_mat.equals(laplacianMatrix));
//	}
//
//}
