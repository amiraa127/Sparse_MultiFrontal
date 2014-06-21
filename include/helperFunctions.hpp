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
 

/* Function : readMtxIntoSparseMatrix
 *-------------------------------------
 * This function reads a sparse matrix market format (*.mtx) file and returns an Eigen sparse matrix object.
 * Currently it only supports matrix object type with coordinate format. Only real or double data types are acceptable at this time.
 * The symmetricity can only be general or symmetric.
 * inputFileName : The path of the input matrix market file.
 */
Eigen::SparseMatrix<double> readMtxIntoSparseMatrix(const std::string inputFileName);

void saveSparseMatrixIntoMtx(const Eigen::SparseMatrix<double> &inputMatrix,const std::string outputFileName);

/* Function: saveMatrixXdToBinary                                                       
 * ------------------------------                                                       
 * This function saves a dense matrix (Eigen's MatrixXd) as a binary file (SVD_F_DB) file format.                                                                               
 * inputMatrix : The dense matrix being saved.                                          
 * outputFileName : Path of the output file.                       
 */
void saveMatrixXdToBinary(const Eigen::MatrixXd& inputMatrix, const std::string outputFileName);

/* Function: readBinaryIntoMatrixXd                                                     
 * --------------------------------                                                    
 * This function reads a dense matrix binary file (SVD_F_DB) and outputs on return, a dense matrix (Eigen's MatrixXd).                                                          
* inputFileName : Path of the input file.                                               
*/                                                                                      
Eigen::MatrixXd readBinaryIntoMatrixXd(const std::string inputFileName);   

/* Function : permuteRowsCols
 * --------------------------
 * This function permutes the rows and columns of a sparse matrix in a symmetric manner according to a given permutation vector.
 * It returns the permuted matrix as an Eigen sparse matrix object. 
 * originalMatrix : Eigen sparse matrix object to be permuted.
 * permVector : Permutation vector. If k = permVector[i], then row i of the original matrix is now row k of the reordered matrix.
 */

Eigen::MatrixXd permuteRows(const Eigen::MatrixXd &originalMatrix, const std::vector<int> &permVector,bool transpose);

Eigen::SparseMatrix<double> permuteRowsCols(const Eigen::SparseMatrix<double> &originalMatrix, const std::vector<int> &permVector);


void convertSparseMatrixIntoGraph(const Eigen::SparseMatrix<double> &inputMatrix,SCOTCH_Graph* graphPtr,const std::string fileName = "default");

std::vector<int> convertBinPartArrayIntoPermVector(SCOTCH_Num* parttab,int arrSize);

void recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix,std::vector<int> gloablInvPerm, const int globalStartIndex, const int threshold, user_IndexTree::node* root, const std::string LR_Method);

std::vector<int> recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix, const int threshold, user_IndexTree & usrTree, const std::string LR_Method);

Eigen::MatrixXd createOneLevelSchurCmpl(const Eigen::SparseMatrix<double> &inputSpMatrix,user_IndexTree &usrTree,const int treeSizeThresh, const std::string LR_Method, std::string inputFileName = "default");

Eigen::MatrixXd createFrontalMatrixFromNode(const eliminationTree::node* node, const Eigen::SparseMatrix<double> & inputSpMatrix);

Eigen::MatrixXd solveSp(const Eigen::SparseMatrix<double> & inputMatrix, const Eigen::MatrixXd RHS);


 
#endif
