#include "sparseMF.hpp"
#include "matrixIO.hpp"
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "helperFunctions.hpp" 

#include "testSolve.hpp"

#include <iostream>



int main(int argc, char* argv[]){
  Eigen::SparseMatrix<double> inputSpMatrix;
  std::cout<<"_________________________________________"<<std::endl;
  std::cout<<"Benchmarking structured mesh matrices"<<std::endl;
  std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"<<std::endl;
  std::cout<<"Benchmarking refrence implementation"<<std::endl;
  std::cout<<"++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  /*
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
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat1.800k");
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
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/structured/localmat1.800k"); 
  testSolveSp(inputSpMatrix, "fast_Iterative");
  */
  /*
  std::cout<<"_________________________________________"<<std::endl;
  std::cout<<"Benchmarking unStructured mesh matrices "<<std::endl;
  std::cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"<<std::endl;
  std::cout<<"Benchmarking refrence implementation"<<std::endl;
  std::cout<<"++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat1.100k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat0.200k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat4.300k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat0.400k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat0.500k");
  testSolveSp(inputSpMatrix, "implicit");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat1.900k");
  testSolveSp(inputSpMatrix, "implicit");
  */
  /*
  std::cout<<"Benchmarking fast iterative implementation"<<std::endl;
  std::cout<<"++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat1.100k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat0.200k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat4.300k");
  testSolveSp(inputSpMatrix, "fast_Iterative");*/
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat0.400k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat0.500k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  inputSpMatrix = readMtxIntoSparseMatrix("data/input_FETI/unStructured/cube/localmat1.900k");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  
  
  /*
  inputSpMatrix = readMtxIntoSparseMatrix("data/stiffness/unStructured/cylinderHead/54k/localmat0");
  testSolveSp(inputSpMatrix, "implicit");

  inputSpMatrix = readMtxIntoSparseMatrix("data/stiffness/unStructured/cylinderHead/330k/localmat0");
  testSolveSp(inputSpMatrix, "implicit");
  
  inputSpMatrix = readMtxIntoSparseMatrix("data/stiffness/unStructured/cylinderHead/2.3m/localmat0");
  testSolveSp(inputSpMatrix, "implicit");
  
  inputSpMatrix = readMtxIntoSparseMatrix("data/stiffness/unStructured/cylinderHead/54k/localmat0");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  
  inputSpMatrix = readMtxIntoSparseMatrix("data/stiffness/unStructured/cylinderHead/330k/localmat0");
  testSolveSp(inputSpMatrix, "fast_Iterative");
 
  inputSpMatrix = readMtxIntoSparseMatrix("data/stiffness/unStructured/cylinderHead/2.3m/localmat0");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  */

    
  inputSpMatrix = readMtxIntoSparseMatrix("data/SMatrices/PoissonS");
  testSolveSp(inputSpMatrix, "implicit");
  
  inputSpMatrix = readMtxIntoSparseMatrix("data/SMatrices/PoissonS");
  testSolveSp(inputSpMatrix, "fast_Iterative");
  
  




  return 0;
}
