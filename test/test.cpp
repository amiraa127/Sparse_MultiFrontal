#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "scotch.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/MetisSupport>
#include "helperFunctions.hpp"
#include "HODLR_Matrix.hpp"
#include "sparseMF.hpp"
#include <string>

#define thresh 30

int main(int argc, char* argv[]){
  std::cout<<"Reading sparse matrix...."<<std::endl;
  Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/localmat0.100k");
  std::cout<<"Sparse matrix read successfully."<<std::endl; 
 
  std::cout<<"Solving..."<<std::endl;
  Eigen::VectorXd exactSoln_Sp = Eigen::VectorXd::LinSpaced(Eigen::Sequential,inputSpMatrix.rows(),-2,2); 
  Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
 
  sparseMF solver(inputSpMatrix);
  solver.printResultInfo = true;
  //solver.testResults = true;
  //Eigen::MatrixXd soln_Sp = solver.fastSolve(RHS_Sp);
  //Eigen::MatrixXd soln_Sp = solver.LU_ExactSolve(RHS_Sp);
  Eigen::MatrixXd soln_Sp = solver.ultraSolve(RHS_Sp);
  
  double error_Sp = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
  std::cout<<error_Sp<<std::endl;
  
  /* // Eigen Conventional Solve
  Eigen::SparseLU<Eigen::SparseMatrix<double>,Eigen::MetisOrdering<int> > Eigen_Solver;
  double startTime = clock();
  inputSpMatrix.makeCompressed();
  Eigen_Solver.isSymmetric(true);
  Eigen_Solver.analyzePattern(inputSpMatrix);
  Eigen_Solver.factorize(inputSpMatrix);
  Eigen::MatrixXd soln_Sp_Eigen = Eigen_Solver.solve(RHS_Sp);
  double endTime = clock();
  double Eigen_SolveTime = (endTime - startTime)/(CLOCKS_PER_SEC);
  double error_Sp_Eigen = (exactSoln_Sp - soln_Sp_Eigen).norm()/exactSoln_Sp.norm();
  std::cout<<error_Sp_Eigen<<std::endl;
  std::cout<<"Eigen Solve Time = "<<Eigen_SolveTime<<" seconds"<<std::endl;
  */ 


  /*
  std::cout<<"Creating Schur complement..."<<std::endl;
  user_IndexTree usrTree;
  Eigen::MatrixXd denseSchur =  createOneLevelSchurCmpl(inputSpMatrix,usrTree,thresh,"ACA","data/Schur_FETI/schur100k_new.bin");
  //saveMatrixXdToBinary(denseSchur,"data/Schur_FETI/schur100k_new.bin");
  std::cout<<"Schur complement created successfully."<<std::endl;
  
  std::cout<<"Solving Schur complement..."<<std::endl;
  int schurSize = denseSchur.rows();
  Eigen::VectorXd exactSoln = Eigen::VectorXd::LinSpaced(Eigen::Sequential,schurSize,-2,2);
  HODLR_Matrix denseHODLR(denseSchur,thresh,usrTree);
  //denseHODLR.set_LRTolerance(1e-1);
  //solver.printLevelAccuracy = true;
  //solver.printLevelRankInfo = true;
  denseHODLR.printResultInfo = true;
  //Eigen::MatrixXd dummy(denseSchur);
  //saveMatrixXdToBinary(dummy,"data/Schur_FETI/schur400k.bin");
  Eigen::VectorXd RHS = denseSchur * exactSoln;
  //Eigen::VectorXd solution = denseHODLR.recLU_Solve(RHS);
  //Eigen::VectorXd solution = denseHODLR.extendedSp_Solve(RHS);
  
  Eigen::VectorXd solution = denseHODLR.iterative_Solve(RHS,50,1e-10,1e-2,"ACA","recLU"); 
  double error = (exactSoln - solution).norm()/exactSoln.norm();
  std::cout<<error<<std::endl;
  double start = clock();
  Eigen::MatrixXd EigenSoln = denseSchur.colPivHouseholderQr().solve(RHS);
  double end = clock();
  std::cout<<(end - start)/CLOCKS_PER_SEC<<std::endl;
  */
  return 0;
}
