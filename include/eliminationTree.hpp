#ifndef ELIMINATION_TREE_HPP
#define ELIMINATION_TREE_HPP

#include <vector>
#include <iostream>
#include "assert.h"
#include "Eigen/Dense"
#include "HODLR_Matrix.hpp"


class eliminationTree{

public:
  struct node{
    std::vector<node*> children;
    int currLevel;
    bool isLeaf;
    int numCols;
    int min_Col;
    int max_Col;    
    Eigen::MatrixXd updateMatrix;
    std::vector<int> updateIdxVector;
   
    // Fast solver data
    
    // Update Matrix
    Eigen::MatrixXd fast_UpdateMatrix;

    // Coupling Matrices
    bool nodeToUpdate_LR;
    bool updateToNode_LR;

    Eigen::MatrixXd nodeToUpdate_U;
    Eigen::MatrixXd nodeToUpdate_V;
    Eigen::MatrixXd updateToNode_U;
    Eigen::MatrixXd updateToNode_V;

    // Node Matrix
    HODLR_Matrix fast_NodeMatrix_HODLR;
    Eigen::MatrixXd fast_NodeMatrix_LU, fast_NodeMatrix_P;
   
  };
  
  int numCols;
  unsigned int numBlocks;
  std::vector<int> levelCols;
  int numLevels;
  node *root;
  std::vector<std::vector<node*> > nodeLevelVec;

  eliminationTree();
  void test(const int* colPtr,const int* row_ind,const int input_numCols);
  eliminationTree(const int* colPtr,const int* row_ind,const int input_numCols);
  eliminationTree(const int input_numCols,const int input_numBlocks,const std::vector<int> & rangVec,const std::vector<int> & treeVec);
  ~eliminationTree();
  
private:
  void freeTree(node* root);
  std::vector<int> createParentVec(const int* colPtr,const int* row_ind,const int input_NumCols);
  std::vector<int> createRangVec(const std::vector<int> & parentVec,const int input_NumCols);
  std::vector<int> createTreeVec(const std::vector<int> & parentVec,const std::vector<int> & rangVec);

  void createTree(const std::vector<int> & rangVec,const std::vector<int> & treeVec);
  void analyzeTree(node *root);
  int  countColsAtLevel(const int level,const node* root)const;
  void countBlks(node* root);
  bool findBlkNum(node* root, int num);

};


#endif
