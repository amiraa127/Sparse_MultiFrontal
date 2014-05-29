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
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>

#define thresh 30

/*This file contains various tests for the Sparse Solver package*/

class Sparse_Solver_Test: public CppUnit::TestCase
{
  /*----------------Creating a Test Suite----------------------*/
  CPPUNIT_TEST_SUITE(Sparse_Solver_Test);
  
  CPPUNIT_TEST(LU_Solver_Test_Small);
  CPPUNIT_TEST(implicit_Solver_Test_Small);
  CPPUNIT_TEST(LU_Solver_Test);
  CPPUNIT_TEST(implicit_Solver_Test);
  
  CPPUNIT_TEST_SUITE_END();

public:
  Sparse_Solver_Test(): CppUnit::TestCase("Sparse Solver Test"){}
  
  void LU_Solver_Test_Small(){
    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    std::cout<<"Testing full LU factorization on a small matrix...."<<std::endl;
    Eigen::MatrixXd inputMatrix = Eigen::MatrixXd::Zero(12,12);
    for (int i = 0; i < 12; i++){
      inputMatrix(i,i)   = 10;
      inputMatrix(0,1)   = 2 ; inputMatrix(0,4) = 4;
      inputMatrix(1,2)   = -1; inputMatrix(1,5) = -3;
      inputMatrix(2,3)   = 2 ; inputMatrix(2,6) = -1;
      inputMatrix(3,7)   = 5 ;
      inputMatrix(4,5)   = -3; inputMatrix(4,8) = 2;
      inputMatrix(5,6)   = -2; inputMatrix(5,9) = -1;
      inputMatrix(6,7)   = 4;  inputMatrix(6,10) = -2; inputMatrix(6,11) = 3 ; 
      inputMatrix(7,11)  = -1;
      inputMatrix(8,9)   = -3;
      inputMatrix(9,10)  = 5;
      inputMatrix(10,11) = 2;
    }
    for (int i = 0; i < 12;i++)
      for (int j = i; j < 12; j++)
	inputMatrix(j,i) = inputMatrix(i,j);
    Eigen::SparseMatrix<double> inputSpMatrix = inputMatrix.sparseView();
    Eigen::VectorXd exactSoln_Sp = Eigen::MatrixXd::Random(12,1);
    Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
    sparseMF solver(inputSpMatrix);
    solver.printResultInfo = true;
    Eigen::MatrixXd soln_Sp = solver.LU_Solve(RHS_Sp);
    double error = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-14);
  }

  void implicit_Solver_Test_Small(){
    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    std::cout<<"Testing implicit solver on a small matrix...."<<std::endl;
    Eigen::MatrixXd inputMatrix = Eigen::MatrixXd::Zero(12,12);
    for (int i = 0; i < 12; i++){
      inputMatrix(i,i)  = 10;
      inputMatrix(0,1)   = 2 ; inputMatrix(0,4) = 4;
      inputMatrix(1,2)   = -1; inputMatrix(1,5) = -3;
      inputMatrix(2,3)   = 2 ; inputMatrix(2,6) = -1;
      inputMatrix(3,7)   = 5 ;
      inputMatrix(4,5)   = -3; inputMatrix(4,8) = 2;
      inputMatrix(5,6)   = -2; inputMatrix(5,9) = -1;
      inputMatrix(6,7)   =  4; inputMatrix(6,10) = -2; inputMatrix(6,11) = 3 ; 
      inputMatrix(7,11)  = -1;
      inputMatrix(8,9)   = -3;
      inputMatrix(9,10)  = 5;
      inputMatrix(10,11) = 2;
    }
    for (int i = 0; i < 12;i++)
      for (int j = i; j < 12; j++)
	inputMatrix(j,i) = inputMatrix(i,j);
    Eigen::SparseMatrix<double> inputSpMatrix = inputMatrix.sparseView();
    Eigen::VectorXd exactSoln_Sp = Eigen::MatrixXd::Random(12,1);
    Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
    sparseMF solver(inputSpMatrix);
    solver.printResultInfo = true;
    Eigen::MatrixXd soln_Sp = solver.implicit_Solve(RHS_Sp);
    double error = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-14);
  }

  void LU_Solver_Test(){
    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    std::cout<<"Testing full LU factorization on a 100k matrix...."<<std::endl;
    Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/localmat0.100k");
    Eigen::VectorXd exactSoln_Sp = Eigen::MatrixXd::Random(inputSpMatrix.rows(),1);
    Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
    sparseMF solver(inputSpMatrix);
    solver.printResultInfo = true;
    Eigen::MatrixXd soln_Sp = solver.LU_Solve(RHS_Sp);
    double error = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-14);
  }

  void implicit_Solver_Test(){
    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    std::cout<<"Testing implicit solver on a 100k matrix...."<<std::endl;
    Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/localmat0.100k");
    Eigen::VectorXd exactSoln_Sp = Eigen::MatrixXd::Random(inputSpMatrix.rows(),1);
    Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
    sparseMF solver(inputSpMatrix);
    solver.printResultInfo = true;
    Eigen::MatrixXd soln_Sp = solver.implicit_Solve(RHS_Sp);
    double error = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-14);
  }


};





int main(int argc, char* argv[]){
  
  CppUnit::TextUi::TestRunner runner;
  runner.addTest(Sparse_Solver_Test::suite());
  runner.run();
  
  std::cout<<"Reading sparse matrix...."<<std::endl;
  Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/localmat0.100k");
  std::cout<<"Sparse matrix read successfully."<<std::endl; 
 
  std::cout<<"Solving..."<<std::endl;
  Eigen::VectorXd exactSoln_Sp = Eigen::VectorXd::LinSpaced(Eigen::Sequential,inputSpMatrix.rows(),-2,2); 
  Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
 
  sparseMF solver(inputSpMatrix);
  solver.printResultInfo = true;
  Eigen::MatrixXd soln_Sp = solver.ultra_Solve(RHS_Sp);
  
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
