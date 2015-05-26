#ifndef SMF_INTERFACE_EIGEN_HPP
#define SMF_INTERFACE_EIGEN_HPP

// c++ std library
#include <string>
#include <vector>

// Eigen library
#include <Eigen/Sparse>
#include <Eigen/Dense>

// SCOTCH library
#include <scotch.h>

#include "eliminationTree.hpp"


/** \brief This function permutes the rows of a dense matrix in a 
 * to a given permutation vector.
 * 
 * \param originalMatrix Eigen sparse matrix object to be permuted.
 * \param permVector     Permutation vector. If k = permVector[i], then row i of the 
 *                       original matrix is now row k of the reordered matrix.
 * \param transpose      ???
 * 
 * \result It returns the permuted matrix as an Eigen dense matrix object. 
 **/
Eigen::MatrixXd permuteRows(const Eigen::MatrixXd &originalMatrix, 
			    const std::vector<int> &permVector, 
			    bool transpose);


/** \brief This function permutes the rows and columns of a sparse matrix in a 
 * symmetric manner according to a given permutation vector.
 * 
 * \param originalMatrix Eigen sparse matrix object to be permuted.
 * \param permVector     Permutation vector. If k = permVector[i], then row i of the 
 *                       original matrix is now row k of the reordered matrix.
 * 
 * \result It returns the permuted matrix as an Eigen sparse matrix object. 
 **/
Eigen::SparseMatrix<double> permuteRowsCols(const Eigen::SparseMatrix<double> &originalMatrix, 
					    const std::vector<int> &permVector);


/** \brief This function permutes the rows of a sparse matrix in a 
 * to a given permutation vector.
 * 
 * \param originalMatrix Eigen sparse matrix object to be permuted.
 * \param permVector     Permutation vector. If k = permVector[i], then row i of the 
 *                       original matrix is now row k of the reordered matrix.
 * \param transpose      ???
 * 
 * \result It returns the permuted matrix as an Eigen sparse matrix object. 
 **/
Eigen::SparseMatrix<double> permuteRows(const Eigen::SparseMatrix<double> &originalMatrix, 
					const std::vector<int> &permVector, 
					bool transpose);

Eigen::SparseMatrix<double> rowScaling(Eigen::SparseMatrix<double> &originalMatrix);


// in the following the struct SCOTCH_Graph and SCOTCH_Num are needed

void convertSparseMatrixIntoGraph(const Eigen::SparseMatrix<double> &inputMatrix,
				  SCOTCH_Graph* graphPtr,
				  const std::string fileName = "default");

std::vector<int> convertBinPartArrayIntoPermVector(SCOTCH_Num* parttab,int arrSize);


// in the following the struct user_IndexTree is needed

void recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix,
			      std::vector<int> gloablInvPerm, 
			      const int globalStartIndex, 
			      const int threshold, 
			      user_IndexTree::node* root, 
			      const std::string LR_Method);

std::vector<int> recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix, 
					  const int threshold, 
					  user_IndexTree & usrTree, 
					  const std::string LR_Method);

Eigen::MatrixXd createOneLevelSchurCmpl(const Eigen::SparseMatrix<double> &inputSpMatrix,
					user_IndexTree &usrTree,
					const int treeSizeThresh, 
					const std::string LR_Method, 
					std::string inputFileName = "default");
 
#endif // SMF_INTERFACE_EIGEN_HPP
