#include "sparseMF.hpp"
#include "matrixIO.hpp"
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "helperFunctions.hpp" 
#include <iostream>



int main(int argc, char* argv[]){
  Eigen::SparseMatrix<double> inputSpMatrix;
  std::cout<<"Bencmarking refrence implementation"<<std::endl;
  std::cout<<"++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.100k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.200k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.300k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.400k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.500k");
  testSolveSp(inputSpMatrix, "implicit");
  std::cout<<"Benchmarking fast iterative implementation"<<std::endl;
  std::cout<<"++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.100k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.200k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.300k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.400k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat0.500k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  return 0;
}
