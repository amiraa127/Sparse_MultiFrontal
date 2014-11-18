#include "Eigen_IML_Vector.hpp"
#include "Eigen_IML_Matrix.hpp"
#include "fastSparse_IML_Precond.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "gmres.h"
int main(){
  /*
  // test simple Constructor
  Eigen_IML_Vector test1;
  Eigen_IML_Vector test2(3);
  std::cout<<test2.size()<<std::endl;
  
  //Test Assignment
  Eigen::VectorXd test3(3);
  test3.setConstant(2);
  Eigen_IML_Vector test4(test3);
  std::cout<<test4<<std::endl;
  test3.setConstant(4);
  Eigen_IML_Vector test5(test3);
  //vector to vector
  test4 = test5;
  std::cout<<test4<<std::endl;
  //vector to scalar 
  test4 = 10.5;
  std::cout<<test4<<std::endl;
  

  //Test operations
  Eigen_IML_Vector test6 = test4+test3;
  Eigen_IML_Vector test7 = test4-test3;
  Eigen_IML_Vector test8 = 12* test4;
  std::cout<<test6<<std::endl;
  std::cout<<test7<<std::endl;
  std::cout<<test8<<std::endl;
  test6(1) = 18.7;
  
  //Element Access
  std::cout<<test6<<std::endl;
  //Dot product
  std::cout<<dot(test6,test7)<<std::endl;
  //Norm
  std::cout<<norm(test6)<<std::endl;

  //Test Sparse Matrix
  Eigen::MatrixXd inputMatrix = Eigen::MatrixXd::Zero(12,12);
  for (int i = 0; i < 12; i++)
    inputMatrix(i,i)   = 10;
  inputMatrix(0,1)   =  2; inputMatrix(0,4)  =  4;
  inputMatrix(1,2)   = -1; inputMatrix(1,5)  = -3;
  inputMatrix(2,3)   =  2 ; inputMatrix(2,6)  = -1;
  inputMatrix(3,7)   =  5 ;
  inputMatrix(4,5)   = -3; inputMatrix(4,8)  =  2;
  inputMatrix(5,6)   = -2; inputMatrix(5,9)  = -1;
  inputMatrix(6,7)   =  4; inputMatrix(6,10) = -2; inputMatrix(6,11) = 3 ;
  inputMatrix(7,11)  = -1;
  inputMatrix(8,9)   = -3;
  inputMatrix(9,10)  =  5;
  inputMatrix(10,11) =  2;
  
  for (int i = 0; i < 12;i++)
    for (int j = i; j < 12; j++)
      inputMatrix(j,i) = inputMatrix(i,j);
  Eigen::SparseMatrix<double> inputSpMatrix = inputMatrix.sparseView();
  
  //Test matrix Initialization
  Eigen_IML_Matrix testMatrix;
  testMatrix = inputSpMatrix;
  std::cout<<testMatrix<<std::endl;
  Eigen::VectorXd exactSoln_Sp = Eigen::MatrixXd::Random(12,1);
  Eigen_IML_Vector testVector(exactSoln_Sp);
  std::cout<<testVector<<std::endl;
  Eigen_IML_Vector multTest = testMatrix * testVector;
  std::cout<<multTest<<std::endl;

  //Test trans mult
  std::cout<<testMatrix.trans_mult(testVector)<<std::endl;
  */
  std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  std::cout<<"Testing fast iterative solver on a 100k matrix...."<<std::endl;
  //Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("../benchmarks/data/input_FETI/TardecAres/localmat1");
  // Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("../benchmarks/data/stiffness/GenericHull/localmat0");                                                                         

  Eigen_IML_Matrix inputSpMatrix;
  //inputSpMatrix = readMtxIntoSparseMatrix("../../benchmarks/data/input_FETI/TardecAres/localmat1");
  //inputSpMatrix = readMtxIntoSparseMatrix("../../benchmarks/data/UF/AMD/G3_circuit/G3_circuit.mtx");
  //inputSpMatrix = readMtxIntoSparseMatrix("../../benchmarks/data/UF/Botonakis/thermomech_dM/thermomech_dM.mtx");
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/CEMW/tmt_sym/tmt_sym.mtx");         
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/GHS_psdef/apache2/apache2.mtx");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/McRae/ecology2/ecology2.mtx");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/Wissgott/parabolic_fem/parabolic_fem.mtx");    
  inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/stiffness/unStructured/cylinderHead/2.3m/localmat0");    

 
  //testSolveSp(inputSpMatrix, "implicit");
  //testSolveSp(inputSpMatrix, "fast_Iterative");
  Eigen_IML_Vector exactSoln_Sp = Eigen::VectorXd::LinSpaced(Eigen::Sequential,inputSpMatrix.rows(),-2,2);
  Eigen_IML_Vector RHS = inputSpMatrix * exactSoln_Sp;
  fastSparse_IML_Precond precond(inputSpMatrix);
  precond.printResultInfo = true;
  Eigen_IML_Vector x0      = precond.solve(RHS);
  Eigen_IML_Vector soln_Sp = precond.iterative_Solve(RHS,100,1e-10,1e-1);
  precond.printResultInfo = false;
  
  double tol = 1e-10;
  int result, maxit = 150,restart = 32;
  Eigen::MatrixXd H =Eigen::MatrixXd::Zero(restart+1,restart);
  result = GMRES(inputSpMatrix,x0,RHS,precond,H,restart,maxit,tol);

  std::cout<<"GMRES flag = "<<result<<std::endl;
  std::cout<<"iterations performed "<<maxit<<std::endl;
  std::cout<<"tolerance achieved : "<<tol<<std::endl;
  return 0;
}
