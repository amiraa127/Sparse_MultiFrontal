#ifndef SMF_SPARSE_MF_HPP
#define SMF_SPARSE_MF_HPP

//Standard C++                   
#include <set>      
#include <string>                                              
#include <vector> 

//External Dependencies   
//Eigen
#include <Eigen/Sparse>                                                                            
#include <Eigen/Dense>                         
//IML
#include "Eigen_IML_Vector.hpp"
#include "gmres.h" 

//SCOTCH
#include <scotch.h>

//Custom Dependencies 
#include "HODLR_Matrix.hpp" 
#include "eliminationTree.hpp"
#include "interfaceEigen.hpp"
#include "extendAdd.hpp"

namespace smf
{
  /** \addtogroup solver
  *  @{
  */
  
  /// \brief Container to hold options for the sparseMF solver.
  struct sparseMF_options
  {
    /// \brief This parameter determines where to switch from dense algebra to 
    /// fast H algebra. 
    /** 
    * Set this parameter based on your time and memory constraints. 
    * A small value will decrease the memory requirement significantly. But if 
    * the value is too small, the accuracy and the running time might suffer 
    * significantly. If you are dealing with a complicated 3D problem, you will 
    * likely need to increase this threshold. See the paper for sample thresholds 
    * for some problems and their respective memory and time performance. 
    * **Default:** 3000
    **/
    int fast_MatrixSizeThresh;
    
    /// \brief This value should typically be 1/10th of the value \ref fast_MatrixSizeThresh.
    /** **Default:** 400 */
    int fast_HODLR_LeafSize;
    
    /// \brief This value refers to the `d` parameter in the BDLR method. 
    /**
     * It can be increased to get a better preconditioner at the expense of a 
     * slower and more memory demanding solver. If the solver does not converge 
     * as expected, it’s a good idea to increase this parameter.
     * **Default:** 1
     **/
    int fast_BoundaryDepth;
    
    int fast_MaxRank;
    
    /// \brief This parameter can also be set using the PrecondTol parameter of 
    /// the iterative_Solve function. 
    /**
     * Smaller values lead to a better preconditioner. Note that decreasing the 
     * value of this parameter without sufficiently increasing the \ref fast_BoundaryDepth 
     * parameter may result in a very inaccurate preconditioner. Decreasing this 
     * value will result in a slower and more memory dependent preconditioner but 
     * for some problems (specially unstructured grids), one needs to play with 
     * this parameter to find the best value.
     * **Default:** 1.e-1
     **/
    double fast_LR_Tol;
    
    double fast_MinPivot;
  };
  
  
  /// \brief Base class for Sparse Multifrontal Solvers
  class sparseMF
  {
  public:
    /// default constructor
    sparseMF(Eigen::SparseMatrix<double>& inputSpMatrix);
    
    /// constructor taking a set of options as second argument
    sparseMF(Eigen::SparseMatrix<double>& inputSpMatrix, const sparseMF_options& options);
    
    /// destructor
    virtual ~sparseMF();
    
    Eigen_IML_Vector solve(const Eigen_IML_Vector & other);
    Eigen::MatrixXd LU_Solve(const Eigen::MatrixXd & inputRHS);
    Eigen::MatrixXd implicit_Solve(const Eigen::MatrixXd & inputRHS);
    Eigen::MatrixXd fast_Solve(const Eigen::MatrixXd & inputRHS);
    Eigen::MatrixXd iterative_Solve(const Eigen::MatrixXd & input_RHS, 
				    int maxIterations, 
				    double stop_tolerance,
				    double LR_Tolerance);

    /// \brief In the case that you want to factorize multiple matrices with 
    /// the same non-zero structure, you can update the values and perform
    /// the numerical facotrization only
    void updateNumericalEntries(Eigen::SparseMatrix<double> newMatrix);

    /// set value of variable \ref printResultInfo
    void setPrintResultInfo(bool v) { printResultInfo = v; }
    
  private:
    bool testResults;
    bool printResultInfo;
    
    Eigen::SparseMatrix<double> reorderedMatrix;
    Eigen::SparseMatrix<double> reorderedMatrix_T;
    Eigen::SparseMatrix<double> L_Matrix;
    Eigen::SparseMatrix<double> U_Matrix;
    Eigen::VectorXd LU_Permutation;
    std::vector<int> permVector;
    eliminationTree* matrixElmTreePtr;
    std::vector<Eigen::Triplet<double,int> > L_TripletVec;
    std::vector<Eigen::Triplet<double,int> > U_TripletVec;
    int Sp_MatrixSize;
    int frontID;
    double averageLargeFrontSize;
    int numLargeFronts;

    sparseMF_options options;
    
    bool symbolic_Factorized;
    bool LU_Factorized;
    bool implicit_Factorized;
    bool fast_Factorized;

    // set initial values
    void init(Eigen::SparseMatrix<double>& inputSpMatrix);

    /*************************************Reordering Related Functions*******************************/
    void reorderMatrix(Eigen::SparseMatrix<double> & inputSpMatrix);

    /***********************************Symbolic Factorization Functions****************************/
    void symbolic_Factorize();
    void symbolic_Factorize(eliminationTree::node* root);
    void updateNodeIdxWithChildrenFillins(eliminationTree::node* root,std::set<int> & idxSet);
    
    
    
    /**********************************Numerical Factorization Functions*****************************/
    Eigen::MatrixXd createPanelMatrix(eliminationTree::node* root);
    Eigen::SparseMatrix<double> createPanelMatrix_Sp(eliminationTree::node* root);
    Eigen::SparseMatrix<double> createPanelGraph(eliminationTree::node* root);
    void createPanelAndGraphMatrix(eliminationTree::node* root, Eigen::SparseMatrix<double> & panelMatrix, Eigen::SparseMatrix<double> & panelGraph);
		  

    void LU_FactorizeMatrix();
    void implicit_FactorizeMatrix();
    void fast_FactorizeMatrix();
    
    void LU_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);
    void implicit_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);
    void fast_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root);

    // Extend/Add
    void nodeExtendAddUpdate(eliminationTree::node* root,Eigen::MatrixXd & nodeFrontalMatrix, std::vector<int> & nodeMappingVector);
    void fast_NodeExtendAddUpdate(eliminationTree::node* root,HODLR_Matrix & panelHODLR,std::vector<int> & parentIdxVec);
    void fast_NodeExtendAddUpdate_Array(eliminationTree::node* root,HODLR_Matrix & panelHODLR,std::vector<int> & parentIdxVec);


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
    

    Eigen::MatrixXd fast_UpwardPass(const Eigen::MatrixXd &inputRHS);
    void fast_UpwardPass_Update(eliminationTree::node* root,Eigen::MatrixXd &modifiedRHS);

    Eigen::MatrixXd fast_DownwardPass(const Eigen::MatrixXd & upwardPassRHS);
    void fast_DownwardPass(eliminationTree::node* root,const Eigen::MatrixXd & upwardPassRHS,Eigen::MatrixXd & finalSoln);
    
    Eigen::MatrixXd fast_NodeToUpdateMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS);
    Eigen::MatrixXd fast_UpdateToNodeMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS);

    Eigen::MatrixXd fast_NodeSolve(eliminationTree::node* root,const Eigen::MatrixXd & RHS);
  
    Eigen::MatrixXd getRowBlkMatrix(const Eigen::MatrixXd & inputMatrix, const std::vector<int> & inputIndex);
    void setRowBlkMatrix(const Eigen::MatrixXd &srcMatrix, Eigen::MatrixXd &destMatrix, const std::vector<int> &destIndex);
    
    
    /*
    Eigen::MatrixXd oneStep_Iterate(const Eigen::MatrixXd & prevStep_result, const Eigen::MatrixXd & initSolveGuess, Eigen::MatrixXd & prevStep_Product);
    */


    /******************************************Test************************************************/
    void test_LU_Factorization();


    /**************************************General Extend Add*************************************/
    std::vector<int> extendIdxVec(std::vector<int> & childIdxVec, std::vector<int> & parentIdxVec);
  };
  
  /** @}*/

} // end namespace smf
  
#endif // SMF_SPARSE_MF_HPP
