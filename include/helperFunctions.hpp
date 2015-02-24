#ifndef HELPERFUNCTIONS_SPARSE_MF_HPP
#define HELPERFUNCTIONS_SPARSE_MF_HPP

#include <string>
#include "scotch.h"
#include <fstream>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense> 
#include <vector>
#include <algorithm>
#include <set>
#include "HODLR_Matrix.hpp"
#include "eliminationTree.hpp"
#include "sparseMF.hpp"

/* Function : permuteRowsCols
 * --------------------------
 * This function permutes the rows and columns of a sparse matrix in a symmetric manner according to a given permutation vector.
 * It returns the permuted matrix as an Eigen sparse matrix object. 
 * originalMatrix : Eigen sparse matrix object to be permuted.
 * permVector : Permutation vector. If k = permVector[i], then row i of the original matrix is now row k of the reordered matrix.
 */

Eigen::MatrixXd permuteRows(const Eigen::MatrixXd &originalMatrix, const std::vector<int> &permVector,bool transpose);

Eigen::SparseMatrix<double> permuteRowsCols(const Eigen::SparseMatrix<double> &originalMatrix, const std::vector<int> &permVector);

Eigen::SparseMatrix<double> permuteRows(const Eigen::SparseMatrix<double> &originalMatrix, const std::vector<int> &permVector,bool transpose);

void convertSparseMatrixIntoGraph(const Eigen::SparseMatrix<double> &inputMatrix,SCOTCH_Graph* graphPtr,const std::string fileName = "default");

std::vector<int> convertBinPartArrayIntoPermVector(SCOTCH_Num* parttab,int arrSize);

void recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix,std::vector<int> gloablInvPerm, const int globalStartIndex, const int threshold, user_IndexTree::node* root, const std::string LR_Method);

std::vector<int> recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix, const int threshold, user_IndexTree & usrTree, const std::string LR_Method);

Eigen::MatrixXd createOneLevelSchurCmpl(const Eigen::SparseMatrix<double> &inputSpMatrix,user_IndexTree &usrTree,const int treeSizeThresh, const std::string LR_Method, std::string inputFileName = "default");

void testSolveSp(Eigen::SparseMatrix<double> & inputMatrix,std::string mode);

Eigen::SparseMatrix<double> rowScaling(Eigen::SparseMatrix<double> &originalMatrix);

 
#endif
