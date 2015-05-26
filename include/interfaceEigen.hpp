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


namespace smf
{

  /** \brief This function permutes the rows of a dense/ sparse matrix in a 
  * to a given permutation vector.
  * 
  * \param originalMatrix Eigen matrix object to be permuted.
  * \param permVector     Permutation vector. If k = permVector[i], then row i of the 
  *                       original matrix is now row k of the reordered matrix.
  * \param transpose      if transpose == true, multiply with the transposed permutation matrix
  * 
  * \result It returns the permuted matrix as an Eigen dense/ sparse matrix object. 
  **/
  template <typename EigenMatrix>
  EigenMatrix permuteRows(const EigenMatrix &originalMatrix, 
			  const std::vector<int> &permVector, 
			  bool transpose = false)
  {
    size_t numVertices = originalMatrix.rows();
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix(numVertices);
    for (size_t i = 0; i < numVertices; i++)
      permMatrix.indices()[i] = permVector[i];
    if (!transpose)
      return (permMatrix * originalMatrix);
    else 
      return permMatrix.transpose() * originalMatrix;
  }


  /** \brief This function permutes the rows and columns of a dense/ sparse matrix in a 
  * symmetric manner according to a given permutation vector.
  * 
  * \param originalMatrix Eigen matrix object to be permuted.
  * \param permVector     Permutation vector. If k = permVector[i], then row i of the 
  *                       original matrix is now row k of the reordered matrix.
  * 
  * \result It returns the permuted matrix as an Eigen dense/ sparse matrix object. 
  **/
  template <typename EigenMatrix>
  EigenMatrix permuteRowsCols(const EigenMatrix &originalMatrix, 
			      const std::vector<int> &permVector)
  {
    size_t numVertices = originalMatrix.rows();
    assert(originalMatrix.rows() == originalMatrix.cols());
  
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix(numVertices);
    for (size_t i = 0; i < numVertices; i++)
      permMatrix.indices()[i] = permVector[i];
    return (permMatrix * (permMatrix * originalMatrix).transpose()).transpose();
  }
  

  Eigen::SparseMatrix<double> rowScaling(Eigen::SparseMatrix<double> &originalMatrix);


  // in the following the struct SCOTCH_Graph and SCOTCH_Num are needed

  void convertSparseMatrixIntoGraph(const Eigen::SparseMatrix<double> &inputMatrix,
				    SCOTCH_Graph* graphPtr,
				    const std::string fileName = "default");

  std::vector<int> convertBinPartArrayIntoPermVector(SCOTCH_Num* parttab,int arrSize);


  // in the following the struct user_IndexTree is needed

  void recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix,
				std::vector<int> gloablInvPerm, 
				int globalStartIndex, 
				int threshold, 
				user_IndexTree::node* root, 
				std::string LR_Method);

  std::vector<int> recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix, 
					    int threshold, 
					    user_IndexTree & usrTree, 
					    std::string LR_Method);

  Eigen::MatrixXd createOneLevelSchurCmpl(const Eigen::SparseMatrix<double> &inputSpMatrix,
					  user_IndexTree &usrTree,
					  int treeSizeThresh, 
					  std::string LR_Method, 
					  std::string inputFileName = "default");
 
} // end namespace smf
  
#endif // SMF_INTERFACE_EIGEN_HPP
