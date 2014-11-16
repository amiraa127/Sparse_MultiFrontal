#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "scotch.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "helperFunctions.hpp"
#include "HODLR_Matrix.hpp"
#include "sparseMF.hpp"
#include "matrixIO.hpp"
#include <string>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>
#include <random>
#include <chrono>
#define thresh 30

/*This file contains various tests for the Sparse Solver package*/

class Sparse_Solver_Test: public CppUnit::TestCase
{
  /*----------------Creating a Test Suite----------------------*/
  CPPUNIT_TEST_SUITE(Sparse_Solver_Test);
  
  //CPPUNIT_TEST(LU_Solver_Test_Small);
  //CPPUNIT_TEST(implicit_Solver_Test_Small);
  //CPPUNIT_TEST(LU_Solver_Test);
  //CPPUNIT_TEST(implicit_Solver_Test);
  CPPUNIT_TEST(fastIterative_Solver_Test);
  
  //CPPUNIT_TEST(extendAdd_LowRankToHODLR_LUQR_Test);
  //CPPUNIT_TEST(extractFromMatrixBlk_Test);
  //CPPUNIT_TEST(extractFromLR_Test);
  //CPPUNIT_TEST(extractFromChild_Test);
  CPPUNIT_TEST(extendAdd_DenseToHODLR_Array_Test);
  
  /*
  CPPUNIT_TEST(extendAdd_DenseToHODLR_Test);
  CPPUNIT_TEST(extend_HODLRtoHODLR_Test);
  CPPUNIT_TEST(extendAdd_HODLRtoHODLR_Test);
  */
  CPPUNIT_TEST_SUITE_END();

public:
  Sparse_Solver_Test(): CppUnit::TestCase("Sparse Solver Test"){}



  void extendAdd_LowRankToHODLR_LUQR_Test(){
    std::cout<<"Testing low-rank to HODLR extend-add with LU and QR Compressions..."<<std::endl;
    int matrixSize = 10000;
    int rank = 10;
    int updateSize = matrixSize/2;  
    std::vector<int> extendIdxVec = createUniqueRndIdx(0,matrixSize-1,updateSize);
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(updateSize,rank);
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(updateSize,rank);
    Eigen::MatrixXd exact_Update = U * V.transpose();
    Eigen::MatrixXd HODLR_Result,exact_Matrix,exact_Result;
    double error;
    //*********************************LU Compression**************************
    std::cout<<"         Testing LU Compression...."<<std::endl;
    HODLR_Matrix sampleMatrix_LU;
    exact_Matrix  = sampleMatrix_LU.createExactHODLR(10,matrixSize,30);
    extendAddUpdate(sampleMatrix_LU,U,V,extendIdxVec,1e-6,"Compress_LU");
    HODLR_Result = sampleMatrix_LU.block(0,0,matrixSize,matrixSize);
    exact_Result = exact_Matrix + extend(extendIdxVec,matrixSize,exact_Update,0,0,updateSize,updateSize,"RowsCols");
    error =(HODLR_Result - exact_Result).norm();
    //std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-5);
    //**********************************QR Compression**************************
    std::cout<<"         Testing QR Compression...."<<std::endl;
    HODLR_Matrix sampleMatrix_QR;
    exact_Matrix  = sampleMatrix_QR.createExactHODLR(10,matrixSize,30);
    extendAddUpdate(sampleMatrix_QR,U,V,extendIdxVec,1e-6,"Compress_QR");
    HODLR_Result = sampleMatrix_QR.block(0,0,matrixSize,matrixSize);
    exact_Result = exact_Matrix + extend(extendIdxVec,matrixSize,exact_Update,0,0,updateSize,updateSize,"RowsCols");
    error =(HODLR_Result - exact_Result).norm();
    //std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-5);
  }

  void extractFromMatrixBlk_Test(){

    int matrixSize = 10000;
    int extractSize = matrixSize/2; 

    std::cout<<"Testing Row Extraction From a Full Dense Matrix...."<<std::endl;
    Eigen::MatrixXd fullMatrix = Eigen::MatrixXd::Random(matrixSize,matrixSize);
    std::vector<int> idxVec    = createUniqueRndIdx(0,matrixSize-1,extractSize);
    Eigen::MatrixXd extract    = Eigen::MatrixXd::Zero(matrixSize,extractSize + 100); 
    extractFromMatrixBlk(fullMatrix,0,0,matrixSize,matrixSize,idxVec,"Rows",extract);
    Eigen::MatrixXd exactExtract = Eigen::MatrixXd::Zero(matrixSize,extractSize);
    for (int i = 0; i < extractSize; i++)
      exactExtract.col(i) = fullMatrix.row(idxVec[i]).transpose();
    double error = (exactExtract - extract.leftCols(extractSize)).norm();
    CPPUNIT_ASSERT(error < 1e-16);

    std::cout<<" Testing Col Extraction From a Full Dense Matrix...."<<std::endl;
    extract    = Eigen::MatrixXd::Zero(matrixSize,extractSize + 100);
    extractFromMatrixBlk(fullMatrix,0,0,matrixSize,matrixSize,idxVec,"Cols",extract);
    exactExtract = Eigen::MatrixXd::Zero(matrixSize,extractSize);
    for (int i = 0; i < extractSize; i++)
      exactExtract.col(i) = fullMatrix.col(idxVec[i]);
    error = (exactExtract - extract.leftCols(extractSize)).norm();
    CPPUNIT_ASSERT(error < 1e-16);

    std::cout<<" Testing Col Extraction From a Dense Subblock...."<<std::endl;
    int subBlkSize = matrixSize/5;
    extractSize    = subBlkSize/2;
    int min_i = 3642;
    int min_j = 5000;
    idxVec         = createUniqueRndIdx(0,subBlkSize-1,extractSize);
    extract        = Eigen::MatrixXd::Zero(subBlkSize,extractSize + 100);
    extractFromMatrixBlk(fullMatrix,min_i,min_j,subBlkSize,subBlkSize,idxVec,"Cols",extract);
    exactExtract = Eigen::MatrixXd::Zero(subBlkSize,extractSize);
    for (int i = 0; i < extractSize; i++)
      exactExtract.col(i) = fullMatrix.col(min_j+idxVec[i]).block(min_i,0,subBlkSize,1);
    error = (exactExtract - extract.leftCols(extractSize)).norm();
    CPPUNIT_ASSERT(error < 1e-16);

    std::cout<<" Testing Row Extraction From a HODLR Dense Matrix...."<<std::endl;
    HODLR_Matrix fullMatrix_HODLR;
    fullMatrix = fullMatrix_HODLR.createExactHODLR(10,matrixSize,30);
    idxVec    = createUniqueRndIdx(0,matrixSize-1,extractSize);
    extract    = Eigen::MatrixXd::Zero(matrixSize,extractSize + 100); 
    extractFromMatrixBlk(fullMatrix_HODLR,0,0,matrixSize,matrixSize,idxVec,"Rows",extract);
    exactExtract = Eigen::MatrixXd::Zero(matrixSize,extractSize);
    for (int i = 0; i < extractSize; i++)
      exactExtract.col(i) = fullMatrix.row(idxVec[i]).transpose();
    error = (exactExtract - extract.leftCols(extractSize)).norm();
    CPPUNIT_ASSERT(error < 1e-16);
   
  }

  void extractFromLR_Test(){
    int rank = 1000;
    int matrixSize = 10000;
    int extractSize = matrixSize/2;
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(matrixSize,rank);
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(matrixSize,rank);
    Eigen::MatrixXd exactUpdate = U * V.transpose();
    std::cout<<"Testing Full Col Extraction From a Low-Rank Product......"<<std::endl;
    std::vector<int> idxVec = createUniqueRndIdx(0,matrixSize - 1,extractSize);
    Eigen::MatrixXd extract = extractFromLR(U,V,0,0,matrixSize,matrixSize,idxVec,"Cols",extractSize + 100);
    Eigen::MatrixXd exactExtract = Eigen::MatrixXd::Zero(matrixSize,extractSize);
    for (int i = 0; i < extractSize; i++)
      exactExtract.col(i) = exactUpdate.col(idxVec[i]);
    double error = (exactExtract - extract.leftCols(extractSize)).norm();
    CPPUNIT_ASSERT(error < 1e-16);
  
    std::cout<<" Testing Block Col Extraction From a Low-Rank Product......"<<std::endl;
    int subBlkSize  = matrixSize/3;
    extractSize = matrixSize/4;
    int min_i   = 100;
    int min_j   = 3456;
    idxVec = createUniqueRndIdx(0,subBlkSize - 1,extractSize);
    extract = extractFromLR(U,V,min_i,min_j,subBlkSize,subBlkSize,idxVec,"Cols",extractSize + 100);
    exactExtract = Eigen::MatrixXd::Zero(subBlkSize,extractSize);
    for(int i = 0; i < extractSize; i++)
      exactExtract.col(i) = exactUpdate.col(min_j + idxVec[i]).block(min_i,0,subBlkSize,1);
    error = (exactExtract - extract.leftCols(extractSize)).norm();
    CPPUNIT_ASSERT(error < 1e-16);
  }

  void extractFromChild_Test(){
    int parentSize  = 10000;
    int subBlkSize  = parentSize/2;
    int extractSize = parentSize/3;
    int childSize   = parentSize/2;
    int min_i = 3456;
    int min_j = 2342;
    std::cout<< "Testing Col Child Extraction From a Dense Child Matrix...." <<std::endl;
    Eigen::MatrixXd childMatrix    = Eigen::MatrixXd::Random(childSize,childSize);
    std::vector<int> updateIdxVec  = createUniqueRndIdx(0,parentSize - 1,childSize);
    std::vector<int> extractIdxVec = createUniqueRndIdx(0,subBlkSize - 1,extractSize);
    Eigen::MatrixXd childExtract   = Eigen::MatrixXd::Zero(subBlkSize,extractSize + 100);
    extractFromChild(parentSize,min_i,min_j,subBlkSize,subBlkSize,childMatrix,extractIdxVec,updateIdxVec,"Cols",childExtract);
    Eigen::MatrixXd childExtend = extend(updateIdxVec,parentSize,childMatrix,0,0,childSize,childSize,"RowsCols");
    Eigen::MatrixXd exactExtract = Eigen::MatrixXd::Zero(subBlkSize,extractSize);
    for (int i = 0; i < extractSize; i++)
      exactExtract.col(i) = childExtend.col(min_j + extractIdxVec[i]).block(min_i,0,subBlkSize,1);
    double error = (exactExtract - childExtract.leftCols(extractSize)).norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-16);
  }
  /*
  void extendAdd_DenseToHODLR_Test(){
    std::cout<<"Testing Dense to HODLR extend-add..."<<std::endl;
    int matrixSize = 1000;
    HODLR_Matrix sampleMatrix;
    Eigen::MatrixXd exact_Matrix  = sampleMatrix.createExactHODLR(10,matrixSize,30);
    std::vector<int> idxVec;
    for (int i = 0; i < matrixSize; i++)
      idxVec.push_back(i);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(idxVec.begin(),idxVec.end(),std::default_random_engine(seed));
    int updateSize = matrixSize/2;
    std::vector<int> extendIdxVec = std::vector<int>(idxVec.begin(),idxVec.begin() + updateSize);
    std::sort(extendIdxVec.begin(),extendIdxVec.end());
    Eigen::MatrixXd D = Eigen::MatrixXd::Random(updateSize,updateSize);
    sampleMatrix.extendAddUpdate(D,extendIdxVec,1e-6,"Compress_LU");
    Eigen::MatrixXd HODLR_Result = sampleMatrix.block(0,0,matrixSize,matrixSize);
    Eigen::MatrixXd exact_Result = exact_Matrix + extend(extendIdxVec,matrixSize,D,0,0,updateSize,updateSize,"RowsCols");
    double error =(HODLR_Result - exact_Result).norm();
    //std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-5);
  }
 
  void extend_HODLRtoHODLR_Test(){
    std::cout<<"Testing HODLR to HODLR extend..."<<std::endl;
    int matrixSize = 10000;
    std::vector<int> idxVec;
    for (int i = 0; i < matrixSize; i++)
      idxVec.push_back(i);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(idxVec.begin(),idxVec.end(),std::default_random_engine(seed));
    int updateSize = matrixSize/2;
    std::vector<int> extendIdxVec = std::vector<int>(idxVec.begin(),idxVec.begin() + updateSize);
     std::sort(extendIdxVec.begin(),extendIdxVec.end());
    HODLR_Matrix D_HODLR;
    Eigen::MatrixXd D = D_HODLR.createExactHODLR(10,updateSize,30);
    D_HODLR.extend(extendIdxVec,matrixSize);
    Eigen::MatrixXd HODLR_Result = D_HODLR.block(0,0,matrixSize,matrixSize);
    Eigen::MatrixXd exact_Result = extend(extendIdxVec,matrixSize,D,0,0,updateSize,updateSize,"RowsCols");
    double error =(HODLR_Result - exact_Result).norm();
    //std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-16);
  } 
  
  void extendAdd_HODLRtoHODLR_Test(){
    std::cout<<"Testing HODLR to HODLR extend-add..."<<std::endl;
    int matrixSize = 10000;
    HODLR_Matrix sampleMatrix;
    Eigen::MatrixXd exact_Matrix  = sampleMatrix.createExactHODLR(10,matrixSize,30);
    std::vector<int> idxVec;
    for (int i = 0; i < matrixSize; i++)
      idxVec.push_back(i);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(idxVec.begin(),idxVec.end(),std::default_random_engine(seed));
    int updateSize = matrixSize/2;
    std::vector<int> extendIdxVec = std::vector<int>(idxVec.begin(),idxVec.begin() + updateSize);
    std::sort(extendIdxVec.begin(),extendIdxVec.end());
    HODLR_Matrix D_HODLR;
    Eigen::MatrixXd D = D_HODLR.createExactHODLR(10,updateSize,30);
    sampleMatrix.extendAddUpdate(D_HODLR,extendIdxVec,1e-6,"Compress_LU");
    Eigen::MatrixXd HODLR_Result = sampleMatrix.block(0,0,matrixSize,matrixSize);
    Eigen::MatrixXd exact_Result = exact_Matrix + extend(extendIdxVec,matrixSize,D,0,0,updateSize,updateSize,"RowsCols");
    double error =(HODLR_Result - exact_Result).norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-4);
  } 
*/


  void extendAdd_DenseToHODLR_Array_Test(){
    std::cout<<"Testing Dense to HODLR Array Extend-Add..."<<std::endl;
    int parentSize = 2000;
    HODLR_Matrix    parentHODLR;
    Eigen::MatrixXd exact_Parent  = parentHODLR.createExactHODLR(10,parentSize,30);
    int updateSize1 = parentSize/2;
    int updateSize2 = parentSize/3;
    std::vector<int> updateIdxVec1 = createUniqueRndIdx(0,parentSize-1,updateSize1);
    std::vector<int> updateIdxVec2 = createUniqueRndIdx(0,parentSize-1,updateSize2);
    Eigen::MatrixXd D1 = Eigen::MatrixXd::Random(updateSize1,updateSize1);
    Eigen::MatrixXd D2 = Eigen::MatrixXd::Random(updateSize2,updateSize2);
    std::vector<std::vector<int> > updateIdxVec_Array;
    std::vector<std::vector<int> > updateIdxVec_Array_HODLR;
    updateIdxVec_Array.push_back(updateIdxVec1);
    updateIdxVec_Array.push_back(updateIdxVec2);
    std::vector<Eigen::MatrixXd *> D_Array;
    D_Array.push_back(&D1);
    D_Array.push_back(&D2);
    std::vector<HODLR_Matrix*> D_HODLR_Array;
    std::vector<Eigen::MatrixXd*> U_Array;
    std::vector<Eigen::MatrixXd*> V_Array;
    extendAddUpdate(parentHODLR,D_Array,D_HODLR_Array,U_Array,V_Array,updateIdxVec_Array,updateIdxVec_Array_HODLR,1e-6,"PS_Boundary");
    Eigen::MatrixXd exact_Result = exact_Parent + extend(updateIdxVec1,parentSize,D1,0,0,updateSize1,updateSize1,"RowsCols") + extend(updateIdxVec2,parentSize,D2,0,0,updateSize2,updateSize2,"RowsCols");
    double error =(parentHODLR.block(0,0,parentSize,parentSize) - exact_Result).norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-5);

  }

  void LU_Solver_Test_Small(){
    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    std::cout<<"Testing full LU factorization on a small matrix...."<<std::endl;
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
    for (int i = 0; i < 12; i++)
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
    Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.100k");
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
    Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.100k");
    Eigen::VectorXd exactSoln_Sp = Eigen::MatrixXd::Random(inputSpMatrix.rows(),1);
    Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
    sparseMF solver(inputSpMatrix);
    solver.printResultInfo = true;
    Eigen::MatrixXd soln_Sp = solver.implicit_Solve(RHS_Sp);
    double error = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-14);
  }
  
  void fastIterative_Solver_Test(){
    std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
    std::cout<<"Testing fast iterative solver on a 100k matrix...."<<std::endl;
    //Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.300k");
    Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("../benchmarks/data/input_FETI/TardecAres/localmat1");
    // Eigen::SparseMatrix<double> inputSpMatrix = readMtxIntoSparseMatrix("../benchmarks/data/stiffness/GenericHull/localmat0");

    Eigen::VectorXd exactSoln_Sp = Eigen::VectorXd::LinSpaced(Eigen::Sequential,inputSpMatrix.rows(),-2,2); 
    Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
    sparseMF solver(inputSpMatrix);
    solver.printResultInfo = true;
    //Eigen::MatrixXd soln_Sp = solver.fast_Solve(RHS_Sp);
    Eigen::MatrixXd soln_Sp = solver.iterative_Solve(RHS_Sp,100,1e-10,1e-1);
    double error = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
    std::cout<<error<<std::endl;
    CPPUNIT_ASSERT(error < 1e-10);
  }

};


int main(int argc, char* argv[]){
  
  CppUnit::TextUi::TestRunner runner;
  runner.addTest(Sparse_Solver_Test::suite());
  runner.run();

  

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
