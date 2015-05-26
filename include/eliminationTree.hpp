#ifndef SMF_ELIMINATION_TREE_HPP
#define SMF_ELIMINATION_TREE_HPP

#include <vector>

#include <Eigen/Dense>

#include "HODLR_Matrix.hpp"

namespace smf
{

  class eliminationTree{

  public:
    struct node{
      std::vector<node*> children;
      int currLevel;
      bool isLeaf;
      int numCols;
      int min_Col;
      int max_Col;    
      Eigen::MatrixXd  updateMatrix;
      std::vector<int> updateIdxVector;
      std::vector<int> panelIdxVector;
    
      // Implicit solver data
      /***************************/
      

      // Coupling Matrices
      bool nodeToUpdate_LR;
      bool updateToNode_LR;

      Eigen::MatrixXd nodeToUpdate_U;
      Eigen::MatrixXd nodeToUpdate_V;
      Eigen::MatrixXd updateToNode_U;
      Eigen::MatrixXd updateToNode_V;

      // Node Matrix
      HODLR_Matrix fast_NodeMatrix_HODLR;
      Eigen::PartialPivLU<Eigen::MatrixXd> nodeMatrix_LU;

      // Ultra solver data
      /*******************************/
      Eigen::MatrixXd updateU;
      Eigen::MatrixXd updateV;
      HODLR_Matrix D_HODLR;
      bool D_UpdateDense;
      bool criterion;
      int frontSize;
    };
    
    int numCols;
    unsigned int numBlocks;
    std::vector<int> levelCols;
    int numLevels;
    node *root;
    std::vector<std::vector<node*> > nodeLevelVec;

    /// default constructor
    eliminationTree();
    
    /// constructor
    eliminationTree(const int* colPtr,
		    const int* row_ind,
		    int input_numCols);
    
    /// constructor
    eliminationTree(int input_numCols,
		    int input_numBlocks,
		    const std::vector<int>& rangVec,
		    const std::vector<int>& treeVec);
    
    /// destructor
    ~eliminationTree();
    
    void test(const int* colPtr,
	      const int* row_ind,
	      int input_numCols);
  private:
    void freeTree(node* root);
    
    std::vector<int> createParentVec(const int* colPtr,
				     const int* row_ind,
				     int input_NumCols);
    
    std::vector<int> createRangVec(const std::vector<int>& parentVec,
				   int input_NumCols);
    
    std::vector<int> createTreeVec(const std::vector<int>& parentVec,
				   std::vector<int>& rangVec);
    
    void build_eliminationTree(int input_numCols,
			       int input_numBlocks,
			       const std::vector<int>& rangVec,
			       const std::vector<int>& treeVec);
    
    void createTree(const std::vector<int>& rangVec,
		    const std::vector<int>& treeVec);
    
    void analyzeTree(node *root);
    int  countColsAtLevel(int level, const node* root) const;
    void countBlks(node* root);
    bool findBlkNum(node* root, int num);

  };

} // end namespace smf

#endif // SMF_ELIMINATION_TREE_HPP
