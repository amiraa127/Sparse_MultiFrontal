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

  void LU_Compute();
  Eigen::MatrixXd LU_Solve(const Eigen::MatrixXd & inputRHS);
  Eigen::MatrixXd implicit_Solve(const Eigen::MatrixXd & inputRHS);
  Eigen::MatrixXd ultra_Solve(const Eigen::MatrixXd & inputRHS);

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
  bool implicit_Factorized;
  bool ultra_Factorized;
  
  double matrixReorderingTime;
  double SCOTCH_ReorderingTime;
  double matrixGraphConversionTime;

  double implicit_FactorizationTime;
  double implicit_SolveTime;
  double implicit_TotalTime;
  double implicit_ExtendAddTime;
  double implicit_SymbolicFactorTime;

  double ultra_FactorizationTime;
  double ultra_SolveTime;
  double ultra_TotalTime;
  double ultra_ExtendAddTime;
  double ultra_SymbolicFactorTime;

  double LU_FactorizationTime;
  double LU_SolveTime;
  double LU_TotalTime;
  double LU_ExtendAddTime;
  double LU_SymbolicFactorTime;
  double LU_AssemblyTime;


  /*************************************Reordering Related Functions*******************************/
  void reorderMatrix(Eigen::SparseMatrix<double> & inputSpMatrix);

  /***********************************Symbolic Factorization Functions****************************/
  void updateNodeIdxWithChildrenFillins(eliminationTree::node* root,std::set<int> & idxSet);
  
  
  
  /**********************************Numerical Factorization Functions*****************************/
  void LU_FactorizeMatrix();
  void implicit_FactorizeMatrix();
  void ultra_FactorizeMatrix();
  
  void LU_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);
  void implicit_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);
  void ultra_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);

  // Extend/Add
  void nodeExtendAddUpdate(eliminationTree::node* root,Eigen::MatrixXd & nodeFrontalMatrix, std::vector<int> & nodeMappingVector);
  void ultra_NodeExtendAddUpdate(eliminationTree::node* root,HODLR_Matrix & panelHODLR,std::vector<int> & parentIdxVec);
  

  /*****************************************Solve Functions****************************************/

  //LU
  void assembleUFactor(const Eigen::MatrixXd & nodeMatrix_U, const Eigen::MatrixXd & update_U, const std::vector<int> & mappingVector);
  void assembleLFactor(const Eigen::MatrixXd & nodeMatrix_L, const Eigen::MatrixXd & update_L, const std::vector<int> & mappingVector);
  void assembleLUMatrix();

  //Implicit
  Eigen::MatrixXd implicit_UpwardPass(const Eigen::MatrixXd &inputRHS);
  void implicit_UpwardPass_Update(eliminationTree::node* root,Eigen::MatrixXd &modifiedRHS);

  Eigen::MatrixXd implicit_DownwardPass(const Eigen::MatrixXd & upwardPassRHS);
  void implicit_DownwardPass(eliminationTree::node* root,const Eigen::MatrixXd & upwardPassRHS,Eigen::MatrixXd & finalSoln);
  

  Eigen::MatrixXd ultra_UpwardPass(const Eigen::MatrixXd &inputRHS);
  void ultra_UpwardPass_Update(eliminationTree::node* root,Eigen::MatrixXd &modifiedRHS);

  Eigen::MatrixXd ultra_DownwardPass(const Eigen::MatrixXd & upwardPassRHS);
  void ultra_DownwardPass(eliminationTree::node* root,const Eigen::MatrixXd & upwardPassRHS,Eigen::MatrixXd & finalSoln);
  


  void ultra_CreateUpdateMatrixForNode(eliminationTree::node* root,const Eigen::MatrixXd & nodeUpdateSoln,const Eigen::MatrixXd & bottomRightMatrix);

  Eigen::MatrixXd fast_NodeToUpdateMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS);
  Eigen::MatrixXd fast_UpdateToNodeMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS);

  Eigen::MatrixXd fast_NodeSolve(eliminationTree::node* root,const Eigen::MatrixXd & RHS);
 



 Eigen::MatrixXd getRowBlkMatrix(const Eigen::MatrixXd & inputMatrix, const std::vector<int> & inputIndex);
  void setRowBlkMatrix(const Eigen::MatrixXd &srcMatrix, Eigen::MatrixXd &destMatrix, const std::vector<int> &destIndex);

  /******************************************Test************************************************/
  void test_LU_Factorization();


  /**************************************General Extend Add*************************************/
  std::vector<int> extendIdxVec(std::vector<int> & childIdxVec, std::vector<int> & parentIdxVec);
};

#endif
