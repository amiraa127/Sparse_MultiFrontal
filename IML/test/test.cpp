#include <unsupported/Eigen/SparseExtra>
#include "Eigen_IML_Vector.hpp"
#include "Eigen_IML_Matrix.hpp"
#include "fastSparse_IML_Precond.hpp"
#include "diag_IML_Precond.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "gmres.h"
int main(){

  std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  std::cout<<"Testing fast iterative solver on a 100k matrix...."<<std::endl;
  //Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("../benchmarks/data/input_FETI/TardecAres/localmat1");
  // Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("../benchmarks/data/stiffness/GenericHull/localmat0");                                                                         

  Eigen_IML_Matrix inputSpMatrix;
  //inputSpMatrix = readMtxIntoSparseMatrix("../../../benchmarks/data/input_FETI/TardecAres/localmat1");
 
  //loadMarket(inputSpMatrix,"../../../benchmarks/data/input_FETI/TardecAres/localmat1");
  //inputSpMatrix = readMtxIntoSparseMatrix("../../benchmarks/data/UF/AMD/G3_circuit/G3_circuit.mtx");
  //inputSpMatrix = readMtxIntoSparseMatrix("../../benchmarks/data/UF/Botonakis/thermomech_dM/thermomech_dM.mtx");
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/CEMW/tmt_sym/tmt_sym.mtx");         
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/GHS_psdef/apache2/apache2.mtx");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/McRae/ecology2/ecology2.mtx");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/Wissgott/parabolic_fem/parabolic_fem.mtx");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/stiffness/unStructured/cylinderHead/2.3m/localmat0");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/SMatrices/linearElasticity/ElasticityS32_660k");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../../benchmarks/data/SMatrices/Poisson/PoissonS64_1M");    
 
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/input_FETI/structured/localmat1.800k");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../benchmarks/data/UF/Janna/Cube_Coup_dt6/Cube_Coup_dt6.mtx");    
  inputSpMatrix   = readMtxIntoSparseMatrix("../../../benchmarks/data/input_FETI/structured/localmat0.100k");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../../benchmarks/data/input_FETI/unStructured/cube/localmat0.500k");

  //inputSpMatrix   = readMtxIntoSparseMatrix("../../../benchmarks/data/stiffness/unStructured/beam/stiffness.300k");
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../../benchmarks/data/stiffness/unStructured/cylinderHead/330k/localmat0");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../../benchmarks/data/input_FETI/structured/localmat0.400k");    
  //inputSpMatrix   = readMtxIntoSparseMatrix("../../../benchmarks/data/input_FETI/unStructured/engine/localmat4");    
 
  
  //inputSpMatrix = rowScaling(inputSpMatrix);
 
  //testSolveSp(inputSpMatrix, "implicit");
  //testSolveSp(inputSpMatrix, "fast_Iterative");
  
  //Eigen_IML_Vector exactSoln_Sp = Eigen::VectorXd::LinSpaced(Eigen::Sequential,inputSpMatrix.rows(),-2,2);
  //Eigen_IML_Vector RHS = inputSpMatrix * exactSoln_Sp;
  Eigen_IML_Vector RHS = Eigen::MatrixXd::Random(inputSpMatrix.rows(),1);
  
  fastSparse_IML_Precond precond(inputSpMatrix);
  precond.printResultInfo = true;
  //Eigen_IML_Vector x1      = precond.implicit_Solve(RHS);
  Eigen_IML_Vector x0      = precond.solve(RHS);
  //Eigen_IML_Vector soln_Sp = precond.iterative_Solve(RHS,10,1e-10,1e-1);
  //Eigen_IML_Vector x0 = Eigen::MatrixXd::Zero(inputSpMatrix.rows(),1);
  precond.printResultInfo = false;
  std::cout<<"RHS norm = "<<RHS.norm()<<std::endl;
  double tol = 1e-4;
  int result, maxit = 5000,restart = 32;
  Eigen::MatrixXd H =Eigen::MatrixXd::Zero(restart+1,restart);
  //result = GMRES(inputSpMatrix,x0,RHS,precond,H,restart,maxit,tol);
  precond.printResultInfo = true;
  Eigen::MatrixXd y = precond.iterative_Solve(RHS,maxit,1e-4,1e-1);
  std::cout<<"GMRES flag = "<<result<<std::endl;
  std::cout<<"iterations performed "<<maxit<<std::endl;
  std::cout<<"tolerance achieved : "<<tol<<std::endl;

  
  diag_IML_Precond diagPrecond(inputSpMatrix);
  Eigen_IML_Vector x2 = diagPrecond.solve(RHS);
  H = Eigen::MatrixXd::Zero(restart+1,restart);
  maxit = 1000;
  tol = 1e-4;
  std::cout<<"here"<<std::endl;
  //result = GMRES(inputSpMatrix,x2,RHS,diagPrecond,H,restart,maxit,tol);

  std::cout<<"GMRES flag = "<<result<<std::endl;
  std::cout<<"iterations performed "<<maxit<<std::endl;
  std::cout<<"tolerance achieved : "<<tol<<std::endl;
  



  return 0;
}
