#ifndef SPARSE_MF_HPP
#define SPARSE_MF_HPP

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
#include "helperFunctions.hpp"
#include "HODLR_Matrix.hpp"

class sparseMF{

public:

  bool testResults;
  bool printResultInfo;
  sparseMF(Eigen::SparseMatrix<double> & inputSpMatrix);
  ~sparseMF();

  /*************************************Exact LU Solver***************************************/
  void LU_compute();
  Eigen::MatrixXd LU_ExactSolve(const Eigen::MatrixXd & inputRHS);
  Eigen::MatrixXd fastSolve(const Eigen::MatrixXd & inputRHS);
 
private:
  Eigen::SparseMatrix<double> reorderedMatrix;
  Eigen::SparseMatrix<double> L_Matrix;
  Eigen::SparseMatrix<double> U_Matrix;
  Eigen::VectorXd LU_Permutation;
  std::vector<int> permVector;
  eliminationTree* matrixElmTreePtr;
  std::vector<Eigen::Triplet<double,int> > L_TripletVec;
  std::vector<Eigen::Triplet<double,int> > U_TripletVec;
  int Sp_MatrixSize;
  int frontID;

  int fast_MatrixSizeThresh;
  int fast_HODLR_LeafSize;
  double fast_LR_Tol;
  double fast_MinValueACA;
  std::string fast_LR_Method;
  

  bool LU_Factorized;
  bool fast_Factorized;

  double matrixReorderingTime;
  double SCOTCH_ReorderingTime;
  double matrixGraphConversionTime;

  double fast_FactorizationTime;
  double fast_SolveTime;
  double fast_TotalTime;
  double fast_ExtendAddTime;
  double fast_SymbolicFactorTime;

  double LU_FactorizationTime;
  double LU_SolveTime;
  double LU_TotalTime;
  double LU_ExtendAddTime;
  double LU_SymbolicFactorTime;
  double LU_AssemblyTime;
  /*************************************Graph Related Functions*******************************/
  Eigen::SparseMatrix<double> reorderMatrix(Eigen::SparseMatrix<double> & inputSpMatrix);


  /**********************************Exact LU Factorization Functions*************************/
  
  void createFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);
  void updateNodeIdxWithChildrenFillins(eliminationTree::node* root,std::set<int> & idxSet);
  void assembleUFactor(const Eigen::MatrixXd & nodeMatrix_U, const Eigen::MatrixXd & update_U, const std::vector<int> & mappingVector);
  void assembleLFactor(const Eigen::MatrixXd & nodeMatrix_L, const Eigen::MatrixXd & update_L, const std::vector<int> & mappingVector);
  void nodeExtendAddUpdate(eliminationTree::node* root,Eigen::MatrixXd & nodeFrontalMatrix, std::vector<int> & nodeMappingVector);
  void LU_FactorizeMatrix();
  void assembleLUMatrix();
  void testLUFactorization();

  /************************************Fast Solve Functions***********************************/
  void fast_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);
  void fast_NodeExtendAddUpdate(eliminationTree::node* root,Eigen::MatrixXd & nodeFrontalMatrix, std::vector<int> & nodeMappingVector);
  void fast_FactorizeMatrix();
  Eigen::MatrixXd fastSolve_UpwardPass(const Eigen::MatrixXd &inputRHS);
  void fast_CreateUpdateMatrixForNode(eliminationTree::node* root,const Eigen::MatrixXd & nodeUpdateSoln,const Eigen::MatrixXd & bottomRightMatrix);
  Eigen::MatrixXd fast_NodeToUpdateMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS);
  Eigen::MatrixXd fast_UpdateToNodeMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS);

  void fastSolve_UpwardPass_Update(eliminationTree::node* root,Eigen::MatrixXd &modifiedRHS);
  Eigen::MatrixXd fastSolve_DownwardPass(const Eigen::MatrixXd & upwardPassRHS);
  void fastSolve_DownwardPass(eliminationTree::node* root,const Eigen::MatrixXd & upwardPassRHS,Eigen::MatrixXd & finalSoln);
  Eigen::MatrixXd fastSolve_NodeSolve(eliminationTree::node* root,const Eigen::MatrixXd & RHS);
  void fastSolve_LRApprox(Eigen::MatrixXd & inputMatrix,Eigen::MatrixXd & U, Eigen::MatrixXd & V,int & calculatedRank,double LR_Tol, std::string input_LRMethod);



 Eigen::MatrixXd getRowBlkMatrix(const Eigen::MatrixXd & inputMatrix, const std::vector<int> & inputIndex);
  void setRowBlkMatrix(const Eigen::MatrixXd &srcMatrix, Eigen::MatrixXd &destMatrix, const std::vector<int> &destIndex);
};

#endif
