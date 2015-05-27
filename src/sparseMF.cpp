#include "sparseMF.hpp"

#include <algorithm>
#include <fstream>

#include "timer.hpp"
#include "output.hpp"

namespace smf
{

  sparseMF::sparseMF(Eigen::SparseMatrix<double>& inputSpMatrix)
  {
    // set default options
    options.fast_MatrixSizeThresh = 3000;
    options.fast_HODLR_LeafSize = 400;
    options.fast_LR_Tol = 1.0e-1;
    options.fast_BoundaryDepth = 1;
    options.fast_MinPivot = 0.0;
    options.fast_MaxRank  = -1;
    
    init(inputSpMatrix);
  }
  

  sparseMF::sparseMF(Eigen::SparseMatrix<double>& inputSpMatrix, const sparseMF_options& options_)
      : options(options_)
  {
    init(inputSpMatrix);
  }
  
  
  void sparseMF::init(Eigen::SparseMatrix<double>& inputSpMatrix)
  {  
    frontID = 0;
    matrixElmTreePtr = NULL;

    averageLargeFrontSize    = 0;
    numLargeFronts           = 0;

    symbolic_Factorized     = false;
    LU_Factorized           = false;
    implicit_Factorized     = false;
    fast_Factorized         = false;

    testResults             = false;
    printResultInfo         = false;
    
  #ifndef NDEBUG
    int numRows = inputSpMatrix.rows();
    int numCols = inputSpMatrix.cols();
    assert( numCols == numRows );
  #endif
    
    Sp_MatrixSize = inputSpMatrix.rows();
    LU_Permutation = Eigen::VectorXd::LinSpaced(Eigen::Sequential, Sp_MatrixSize, 0, Sp_MatrixSize - 1);
    
    inputSpMatrix.prune(1.0e-14);  
    reorderMatrix(inputSpMatrix);
  }


  sparseMF::~sparseMF()
  {
    if (matrixElmTreePtr != NULL)
      delete(matrixElmTreePtr);
  }


  void sparseMF::reorderMatrix(Eigen::SparseMatrix<double> & inputSpMatrix)
  {
    Timer t0("reorderedMatrix");
    
    // **********Convert Matrix into SCOTCH_Graph********                   
    Timer t1("matrixGraphConversionTime");
    
    Eigen::SparseMatrix<double> partitionMatrix = inputSpMatrix.triangularView<Eigen::StrictlyLower>() + inputSpMatrix.triangularView<Eigen::StrictlyUpper>();
    
    if (!partitionMatrix.isCompressed())
      partitionMatrix.makeCompressed();
    
    SCOTCH_Num baseval = 0;
    SCOTCH_Num vertnbr = partitionMatrix.rows();
    SCOTCH_Num edgenbr = partitionMatrix.nonZeros();
    SCOTCH_Num*  verttab;
    SCOTCH_Num*  edgetab;
    bool memAlloc = false; 
    // Create verttab and edgetab
    int* outerIndexPtr = partitionMatrix.outerIndexPtr();
    int* innerIndexPtr = partitionMatrix.innerIndexPtr();
    if (sizeof(SCOTCH_Num) == sizeof(int)){
      verttab = outerIndexPtr;
      edgetab = innerIndexPtr;
    }else {
      verttab = (SCOTCH_Num*)calloc(vertnbr + 1,sizeof(SCOTCH_Num));
      edgetab = (SCOTCH_Num*)calloc(edgenbr,sizeof(SCOTCH_Num));
      for (int i = 0; i <= vertnbr; i++)
	verttab[i] = outerIndexPtr[i];
      for (int i = 0; i < edgenbr; i++)
	edgetab[i] = innerIndexPtr[i];
      memAlloc = true;
    }
    
    SCOTCH_Graph* graphPtr = SCOTCH_graphAlloc();
    assert_msg(graphPtr != NULL, "Could not allocate graph.");
    assert_msg(SCOTCH_graphInit(graphPtr) == 0, "Could not initialize graph.");
    assert_msg(SCOTCH_graphBuild(graphPtr,baseval,vertnbr,verttab,verttab + 1,NULL,NULL,edgenbr,edgetab,NULL) ==0, 
	      "Failed to build graph.");
    assert_msg(SCOTCH_graphCheck(graphPtr) == 0, "Graph inconsistent.");
    t1.stop(); // matrixGraphConversionTime

    
    // Find graph statistics                                                               
    SCOTCH_Num numVertices,numEdges;
    SCOTCH_graphSize(graphPtr,&numVertices,&numEdges);
    std::cout<<numVertices<<" "<<numEdges<<std::endl;

    //******************** Order graph**********************                                                                                                                                                                   
    // Initialize ordering strategy                                                         
    SCOTCH_Strat* orderingStratPtr = SCOTCH_stratAlloc() ;
    assert_msg(SCOTCH_stratInit(orderingStratPtr) == 0, 
	       "Could not initialize ordering strategy.");
    
    //std::string orderingStratStr = "n{sep=m{vert=50,low=h{pass=10},asc=f{bal=0.1}}|m{vert=50,low=h{pass=10},asc=f{bal=0.1}},ole=s,ose=g}";
    //std::string orderingStratStr = "n{sep=/(vert>2000)?m{vert=50,low=h{pass=10},asc=f{bal=0.1}}|m{vert=50,low=h{pass=10},asc=f{bal=0.1}};,ole=s,ose=g}"; 
    //std::string orderingStratStr =  "c{rat=0.7,cpr=n{sep=/(vert>120)?m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}}|m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}};,ole=f{cmin=0,cmax=100000,frat=0.0},ose=g},unc=n{sep=/(vert>120)?(m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}})|m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}};,ole=f{cmin=15,cmax=100000,frat=0.08},ose=g}}";

    std::string orderingStratStr =  "c{rat=0.7,cpr=n{sep=/(vert>120)?m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}}|m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}};,ole=f{cmin=0,cmax=100000,frat=0.0},ose=g},unc=n{sep=/(vert>120)?(m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}})|m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}};,ole=f{cmin=15,cmax=100000,frat=0.08},ose=g}}";
    assert_msg(SCOTCH_stratGraphOrder(orderingStratPtr , orderingStratStr.c_str()) == 0, 
	       "Could not set strategy string.");
  
    // Initialize variables                                                           
    SCOTCH_Num* permtab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
    SCOTCH_Num* peritab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
    SCOTCH_Num* treetab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
    SCOTCH_Num* rangtab = (SCOTCH_Num*)calloc((numVertices+1),sizeof(SCOTCH_Num));
    SCOTCH_Num* cblknbr = (SCOTCH_Num*)calloc(1,sizeof(SCOTCH_Num));

    
    // Reorder graph         
    Timer t2("SCOTCH_ReorderingTime");
    msg_dbg("reordering graph ...");
    
    assert_msg(SCOTCH_graphOrder(graphPtr, orderingStratPtr, permtab, peritab, cblknbr, rangtab, treetab) == 0, 
	       "Graph ordering failed.");
    msg_dbg("reordering complete");
    t2.stop(); // SCOTCH_ReorderingTime

    // Permute rows and columns of the original sparse matrix and RHS                   
    permVector = std::vector<int>(permtab, permtab + numVertices);
    reorderedMatrix   =  permuteRowsCols(inputSpMatrix, permVector);
    reorderedMatrix_T =  reorderedMatrix.transpose();

    // Create elimination tree  
    int numBlocks = *cblknbr;
    std::cout<<numBlocks<<std::endl;
    std::vector<int> rangVector(rangtab, rangtab + numBlocks + 1);
    std::vector<int> treeVector(treetab, treetab + numBlocks);
    msg_dbg("Creating elimination tree..");
    int* col_Ptr = reorderedMatrix.outerIndexPtr();
    int* row_Ind = reorderedMatrix.innerIndexPtr();
  //matrixElmTreePtr = new eliminationTree(numVertices,numBlocks,rangVector,treeVector);     
    matrixElmTreePtr = new eliminationTree(col_Ptr,row_Ind,numVertices);
    
    msg_dbg("Elimination tree created successfully.");

    // Free space     
    free(permtab);
    free(peritab);
    free(treetab);
    free(rangtab);
    free(cblknbr);
    SCOTCH_graphExit(graphPtr);
    if (memAlloc == true){
      free(edgetab);
      free(verttab);
    }
  }
  

  void sparseMF::updateNumericalEntries(Eigen::SparseMatrix<double> newMatrix)
  {
    symbolic_Factorize();
    reorderedMatrix   =  permuteRowsCols(newMatrix, permVector);
    reorderedMatrix_T =  reorderedMatrix.transpose();
    LU_Factorized = false;
    implicit_Factorized = false;
    fast_Factorized = false;
  }

  
  void sparseMF::symbolic_Factorize()
  {
    if (symbolic_Factorized == false){
      Timer t("symbolic_FactorizationTime");
      for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
	int currLevel = matrixElmTreePtr->numLevels - i ;
	std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
	for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	  eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	  symbolic_Factorize(currNodePtr);
	}  
	msg_dbg("Symbolic factorization done for level ",currLevel);
      }
      symbolic_Factorized = true;
      if (numLargeFronts > 0)
	averageLargeFrontSize = pow(averageLargeFrontSize * 1.0/numLargeFronts,1.0/3);
    }
  }


  void sparseMF::symbolic_Factorize(eliminationTree::node* root)
  {
    int minIdx = root->min_Col;
  #ifndef NDEBUG
    int maxIdx = root->max_Col;
  #endif
    int nodeSize = root->numCols;
    int endNodeIdx = minIdx + nodeSize - 1;
    std::set<int> idxSet;
  
    // Find the set of connected indices for cols        
    for (int k = minIdx; k <= endNodeIdx; ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix,k); it; ++it)     {
	int rowIdx = it.row();
	int colIdx = it.col();
	if (rowIdx >= minIdx && colIdx >=minIdx){
	  idxSet.insert(it.col());
	  idxSet.insert(it.row());
	}
      }
    
    // Find the set of connected indices for rows   
    for (int k = minIdx; k <= endNodeIdx; ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix_T,k); it; ++it)     {
	int rowIdx = it.row();
	int colIdx = it.col();
	if (rowIdx >= minIdx && colIdx >=minIdx){
	  idxSet.insert(it.col());
	  idxSet.insert(it.row());
	}
      }
    
    updateNodeIdxWithChildrenFillins(root,idxSet);
    root->panelIdxVector = std::vector<int>(idxSet.begin(),idxSet.end());
    std::sort(root->panelIdxVector.begin(),root->panelIdxVector.end());
    assert(root->panelIdxVector[0] == minIdx);
    assert(root->panelIdxVector[nodeSize - 1] == maxIdx);
    root->updateIdxVector = std::vector<int>(root->panelIdxVector.begin() + nodeSize,root->panelIdxVector.end());
    if (((int)idxSet.size() >= options.fast_MatrixSizeThresh)){
      averageLargeFrontSize += pow(idxSet.size(),3);
      numLargeFronts++;
    }
  }


  Eigen::MatrixXd sparseMF::createPanelMatrix(eliminationTree::node* root)
  {
    int panelSize = root->panelIdxVector.size();
    int minIdx = root->panelIdxVector[0];
    int nodeSize = root->numCols;
    int endNodeIdx = minIdx + nodeSize - 1;
    std::map<int,int> idxMap;
    for(int i = 0; i < panelSize; i++)
      idxMap[root->panelIdxVector[i]] = i; 
    Eigen::MatrixXd panelMatrix = Eigen::MatrixXd::Zero(panelSize,panelSize);
    std::set<int> idxSet(root->panelIdxVector.begin(), root->panelIdxVector.end());
    
    // loop cols
    for (int k = minIdx; k <= endNodeIdx; ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix,k); it; ++it){
	int rowIdx = it.row();
	int colIdx = it.col();
	if (rowIdx >= minIdx)
	  panelMatrix(idxMap[rowIdx],idxMap[colIdx]) = it.value();
      }
    
    if (root->currLevel == 0)
      return panelMatrix;
    
    //loop rows
    for (int k = minIdx; k <= endNodeIdx; ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix_T,k); it; ++it){
	int rowIdx = it.row();
	int colIdx = it.col();
	if (rowIdx >= minIdx)
	  panelMatrix(idxMap[colIdx],idxMap[rowIdx]) = it.value();
      }
    
    return panelMatrix;
  }
  

  void sparseMF::createPanelAndGraphMatrix(eliminationTree::node* root, 
					   Eigen::SparseMatrix<double>& panelMatrix, 
					   Eigen::SparseMatrix<double>& panelGraph)
  {
    int panelSize  = root->panelIdxVector.size();
    int nodeSize   = root->numCols;
    panelGraph     = Eigen::SparseMatrix<double> (panelSize,panelSize);
    panelMatrix    = Eigen::SparseMatrix<double> (panelSize,panelSize);
    int minIdx     = root->panelIdxVector[0];
    int maxIdx     = root->panelIdxVector[panelSize - 1];
    int endNodeIdx = minIdx + nodeSize - 1;
    std::map<int,int> idxMap;
    std::set<int> idxSet(root->panelIdxVector.begin(), root->panelIdxVector.end());
    std::vector<Eigen::Triplet<double,int> > tripletVec_Matrix,tripletVec_Graph;
    
    for(int i = 0; i < panelSize; i++)
      idxMap[root->panelIdxVector[i]] = i; 
    
    if (root->currLevel == 0){
      for (int k = minIdx; k <= maxIdx; ++k)
	for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix,k); it; ++it){
	  int rowIdx   = it.row();
	  int colIdx   = it.col();
	  if (rowIdx >= minIdx){
	    Eigen::Triplet<double,int> currEntry(idxMap[rowIdx],idxMap[colIdx],it.value());
	    tripletVec_Matrix.push_back(currEntry);
	  }
	}
      panelMatrix.setFromTriplets(tripletVec_Matrix.begin(),tripletVec_Matrix.end());
      return;
    }
    
    //panel and graph matrix frontal col
    for (int k = minIdx; k <= endNodeIdx; ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix,k); it; ++it){
	int rowIdx = it.row();
	int colIdx = it.col();
	if (rowIdx >= minIdx){
	  Eigen::Triplet<double,int> currEntry(idxMap[rowIdx],idxMap[colIdx],it.value());
	  tripletVec_Matrix.push_back(currEntry);
	  tripletVec_Graph.push_back(currEntry);
	}
      }
    
    // panel and graph matrix row
    for (int k = minIdx; k <= endNodeIdx; ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix_T,k); it; ++it){
	int rowIdx = it.row();
	int colIdx = it.col();
	if (rowIdx > endNodeIdx){
	  Eigen::Triplet<double,int> currEntry(idxMap[colIdx],idxMap[rowIdx],it.value());
	  tripletVec_Matrix.push_back(currEntry);
	  tripletVec_Graph.push_back(currEntry);
	}
      }
    
    // graph matrix inside
    for (int i = 0; i < (int)root->updateIdxVector.size();i++){
      int k = root->updateIdxVector[i];
      for (Eigen::SparseMatrix<double>::InnerIterator it(reorderedMatrix,k); it; ++it){
	int rowIdx = it.row();
	int colIdx = it.col();
	bool rowFind = idxSet.count(rowIdx);
	if (rowFind){
	  Eigen::Triplet<double,int> currEntry(idxMap[rowIdx],idxMap[colIdx],it.value());
	  tripletVec_Graph.push_back(currEntry);
	}
      }
    }
    
    panelMatrix.setFromTriplets(tripletVec_Matrix.begin(),tripletVec_Matrix.end());
    panelGraph.setFromTriplets(tripletVec_Graph.begin(),tripletVec_Graph.end());
  }
  

  void sparseMF::LU_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root)
  {
    int nodeSize = root->numCols;
    Eigen::MatrixXd LU_Permutation_Blk = LU_Permutation.block(root->min_Col,0,nodeSize,1);

    int frontalMatrixSize = root->panelIdxVector.size();
    int updateMatrixSize  = frontalMatrixSize - nodeSize;
  
    Eigen::MatrixXd frontalMatrix = createPanelMatrix(root);

    // Update frontal matrix using updates from children
    Timer t("LU_ExtendAddTime");
    nodeExtendAddUpdate(root,frontalMatrix,root->panelIdxVector);
    t.stop(); // LU_ExtendAddTime
    
    // Create update matrices
    Eigen::MatrixXd nodeMatrix = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
    Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
    Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);
    Eigen::PartialPivLU<Eigen::MatrixXd> nodeMatrix_LU(nodeMatrix);
    Eigen::MatrixXd updateMatrix = frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize) - updateToNode * nodeMatrix_LU.solve(nodeToUpdate);
    root->updateMatrix = updateMatrix;
    //root->updateIdxVector = updateIdxVector;
    
    // Update L and U factors
    Eigen::MatrixXd nodeMatrix_LUMatrix = nodeMatrix_LU.matrixLU();
    Eigen::MatrixXd nodeMatrix_L = Eigen::MatrixXd::Identity(nodeSize,nodeSize);
    nodeMatrix_L.triangularView<Eigen::StrictlyLower>() = nodeMatrix_LUMatrix.triangularView<Eigen::StrictlyLower>();
    Eigen::MatrixXd nodeMatrix_U = nodeMatrix_LUMatrix.triangularView<Eigen::Upper>();
    //Eigen::MatrixXd nodeMatrix_P = nodeMatrix_LU.permutationP();
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> nodeMatrix_P = nodeMatrix_LU.permutationP();

    Eigen::MatrixXd update_U = nodeMatrix_L.triangularView<Eigen::UnitLower>().solve(nodeToUpdate);
    Eigen::MatrixXd update_L = ((nodeMatrix_U.transpose()).triangularView<Eigen::Lower>().solve(updateToNode.transpose())).transpose();

    // Update LU_Permutation 
    LU_Permutation.block(root->min_Col,0,nodeSize,1) = nodeMatrix_P * LU_Permutation_Blk;
    // Assemble L and U factors 
    assembleUFactor(nodeMatrix_U,update_U,root->panelIdxVector);
    assembleLFactor(nodeMatrix_L,update_L,root->panelIdxVector);
    
  }


  void sparseMF::implicit_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root)
  {
    int nodeSize = root->numCols;

    int frontalMatrixSize = root->panelIdxVector.size();
    int updateMatrixSize  = frontalMatrixSize - nodeSize;
    //std::map<int,int> idxMap;
  
    Eigen::MatrixXd frontalMatrix = createPanelMatrix(root);
    
    // Update frontal matrix using updates from children
    Timer t("implicit_ExtendAddTime");
    nodeExtendAddUpdate(root,frontalMatrix,root->panelIdxVector);
    t.stop();
    
    Eigen::MatrixXd updateSoln,updateMatrix;
    Eigen::MatrixXd nodeToUpdate_U,nodeToUpdate_V;
    Eigen::MatrixXd updateToNode_U,updateToNode_V;

    // Create update matrices
    if (root->currLevel != 0){
      Eigen::MatrixXd nodeMatrix   = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
      Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
      Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);
      root->nodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(nodeMatrix);
      updateSoln = (root->nodeMatrix_LU).solve(nodeToUpdate);
      root->nodeToUpdate_U = nodeToUpdate;
      root->updateToNode_U = updateToNode;
      root->updateMatrix   = frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize) - (root->updateToNode_U) * updateSoln;
    }else{
      root->nodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(frontalMatrix);
    }

      //Special Operations :DD
    /*
      std::cout<<nodeSize<<std::endl;
      if (nodeSize >= 2000){
      std::stringstream ss,ss2;
      int minIdx = root->min_Col;
      ss << frontID; 
      ss2<< root->currLevel;
      std::string outputFileName = "300_front_num_" + ss.str() +"_level_" + ss2.str();
      std::string outputFileNameSp = "300_front_num_" + ss.str() +"_level_" +  ss2.str() + "_Graph";
      saveMatrixXdToBinary(frontalMatrix.topLeftCorner(nodeSize,nodeSize),outputFileName); 
      saveSparseMatrixIntoMtx(reorderedMatrix.block(minIdx,minIdx,nodeSize,nodeSize),outputFileNameSp);
      frontID ++;
    }
    */
    
  }


  void sparseMF::fast_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root)
  {
    int nodeSize = root->numCols;
    int frontalMatrixSize = root->panelIdxVector.size();
    int updateMatrixSize  = frontalMatrixSize - nodeSize;

    // Update frontal matrix using updates from children
    root->D_UpdateDense = true;
    root->frontSize = frontalMatrixSize;
    root->criterion = (frontalMatrixSize >= options.fast_MatrixSizeThresh && nodeSize >= 5);
    std::vector<eliminationTree::node*> nodeChildren = root->children;
    int numChildren = nodeChildren.size();

    /*
    for (int i = 0; i < numChildren; i++){
      eliminationTree::node* childNode = nodeChildren[i];
      if (childNode->D_UpdateDense == false){
	root->criterion = true;
	//childHODLR = true;
	break;
      }
    }
    */

    if (root->criterion == false){

      for (int i = 0; i < numChildren; i++){
	eliminationTree::node* childNode = nodeChildren[i];
	if (childNode->D_UpdateDense == false){
	  childNode->updateMatrix = childNode->D_HODLR.block(0,0,childNode->D_HODLR.rows(),childNode->D_HODLR.cols()) + childNode->updateU * childNode->updateV.transpose();
	  childNode->D_HODLR.destroyAllData();
	  childNode->updateU.resize(0,0);
	  childNode->updateV.resize(0,0);
	}
      }
    }
    
    if (root->criterion == true){
      Eigen::SparseMatrix<double> panelMatrix,panelGraph;
      createPanelAndGraphMatrix(root,panelMatrix,panelGraph);
      root->D_UpdateDense = false;
      if (root->currLevel != 0){  
	user_IndexTree usrTree;
	usrTree.rootNode = new user_IndexTree::node;
	usrTree.rootNode->splitIndex          = nodeSize - 1;
	//usrTree.rootNode->topOffDiag_minRank  = -1;
	//usrTree.rootNode->bottOffDiag_minRank = -1;
	usrTree.rootNode->LR_Method           = "identifyBoundary";
	usrTree.rootNode->left                = NULL;
	usrTree.rootNode->right               = NULL;
	HODLR_Matrix panelHODLR = HODLR_Matrix(panelMatrix,panelGraph, options.fast_HODLR_LeafSize,usrTree,"identifyBoundary");   
	
	panelHODLR.set_LRTolerance(options.fast_LR_Tol);
	panelHODLR.set_BoundaryDepth(options.fast_BoundaryDepth);
	Timer t("fast_ExtendAddTime");
	fast_NodeExtendAddUpdate_Array(root,panelHODLR,root->panelIdxVector);
	t.stop();
	
	Eigen::MatrixXd UB    = panelHODLR.returnTopOffDiagU();
	Eigen::MatrixXd VB    = panelHODLR.returnTopOffDiagV();
	Eigen::MatrixXd UC    = panelHODLR.returnBottOffDiagU();
	Eigen::MatrixXd VC    = panelHODLR.returnBottOffDiagV();

	root->fast_NodeMatrix_HODLR = HODLR_Matrix();
	root->D_HODLR = HODLR_Matrix();
	
	splitAtTop(panelHODLR,root->fast_NodeMatrix_HODLR,root->D_HODLR);
	
	root->updateU         = UC;
	root->updateV         = (-VC.transpose() * root->fast_NodeMatrix_HODLR.recLU_Solve(UB) * VB.transpose()).transpose();
	root->nodeToUpdate_U  = UB;
	root->nodeToUpdate_V  = VB;
	root->updateToNode_U  = UC;
	root->updateToNode_V  = VC;
	
	root->nodeToUpdate_LR = true;
	root->updateToNode_LR = true;
      }else{
    
	root->fast_NodeMatrix_HODLR = HODLR_Matrix(panelMatrix, options.fast_HODLR_LeafSize, "identifyBoundary") ;
	root->fast_NodeMatrix_HODLR.set_LRTolerance(options.fast_LR_Tol);
	root->fast_NodeMatrix_HODLR.set_BoundaryDepth(options.fast_BoundaryDepth);
	Timer t("fast_ExtendAddTime");
	fast_NodeExtendAddUpdate_Array(root,root->fast_NodeMatrix_HODLR,root->panelIdxVector);
	t.stop();
      }
      (root->fast_NodeMatrix_HODLR).recLU_Compute();
    }else{

      Eigen::MatrixXd frontalMatrix = createPanelMatrix(root);
      Timer t("fast_ExtendAddTime");
      nodeExtendAddUpdate(root,frontalMatrix,root->panelIdxVector);
      t.stop();
    
      // Create update matrices
      if (root->currLevel != 0){      
	Eigen::MatrixXd nodeMatrix = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
	root->nodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(nodeMatrix);
	Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
	Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);
	Eigen::MatrixXd updateSoln   = (root->nodeMatrix_LU).solve(nodeToUpdate);
	root->nodeToUpdate_LR = false;
	root->updateToNode_LR = false;
	root->nodeToUpdate_U  = nodeToUpdate;
	root->updateToNode_U  = updateToNode;
	root->updateMatrix = frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize) - (root->updateToNode_U) * updateSoln;
	assert((int)root->updateMatrix.rows() == (int)root->updateIdxVector.size());
      }else{
	root->nodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(frontalMatrix);
      }
    }
  }
  

  void sparseMF::updateNodeIdxWithChildrenFillins(eliminationTree::node* root,
						  std::set<int>& idxSet)
  {
    if (root->isLeaf == true)
      return;
    // Go over all childern
    std::vector<eliminationTree::node*> nodeChildren = root->children;
    int numChildren = nodeChildren.size();
    for (int i = 0; i < numChildren; i++){
      eliminationTree::node* childNode = nodeChildren[i];
      std::vector<int> childUpdateIdxVector = childNode->updateIdxVector;
      int updateIdxVectorSize = childUpdateIdxVector.size();
      Eigen::MatrixXd childUpdateMatrix = childNode->updateMatrix;
      for (int j = 0; j < updateIdxVectorSize; j++)
	idxSet.insert(childUpdateIdxVector[j]);
    }
  }


  void sparseMF::assembleUFactor(const Eigen::MatrixXd& nodeMatrix_U, 
				 const Eigen::MatrixXd& update_U, 
				 const std::vector<int>& mappingVector)
  {
    int nodeSize = nodeMatrix_U.cols();
    int updateSize = update_U.cols();
    assert((int)(nodeSize + updateSize) == (int)mappingVector.size());
    
    // Insert nodeMatrix_U
    for (int i = 0; i < nodeSize; i++)
      for (int j = i; j < nodeSize; j++){
	Eigen::Triplet<double,int> currEntry(mappingVector[i],mappingVector[j],nodeMatrix_U(i,j));
	U_TripletVec.push_back(currEntry);
      }

    //Insert update_U
    for (int i = 0; i < nodeSize; i++)
      for (int j = 0; j < updateSize; j++){
	Eigen::Triplet<double,int> currEntry(mappingVector[i],mappingVector[j + nodeSize],update_U(i,j));
	U_TripletVec.push_back(currEntry);
      }
  }


  void sparseMF::assembleLFactor(const Eigen::MatrixXd& nodeMatrix_L, 
				 const Eigen::MatrixXd& update_L, 
				 const std::vector<int>& mappingVector)
  {
    int nodeSize = nodeMatrix_L.rows();
    int updateSize = update_L.rows();
    assert((int)(nodeSize + updateSize) == (int)mappingVector.size());
    
    // Insert nodeMatrix_U
    for (int i = 0; i < nodeSize; i++)
      for (int j = 0; j <= i; j++){
	Eigen::Triplet<double,int> currEntry(mappingVector[i],mappingVector[j],nodeMatrix_L(i,j));
	L_TripletVec.push_back(currEntry);
      }

    // Insert update_U
    for (int i = 0; i < updateSize; i++)
      for (int j = 0; j < nodeSize; j++){
	Eigen::Triplet<double,int> currEntry(mappingVector[i + nodeSize],mappingVector[j],update_L(i,j));
	L_TripletVec.push_back(currEntry);
      }
  }


  void sparseMF::nodeExtendAddUpdate(eliminationTree::node* root,
				     Eigen::MatrixXd & nodeFrontalMatrix,
				     std::vector<int> & parentIdxVec)
  {
    if (root->isLeaf == true)
      return;
    
    // Go over all childern
    std::vector<eliminationTree::node*> nodeChildren = root->children;
    int numChildren = nodeChildren.size();
    for (int i = 0; i < numChildren; i++){
      eliminationTree::node* childNode = nodeChildren[i];
      int updateMatrixSize = childNode->updateIdxVector.size();
      // Find update matrix extend add indices
      std::vector<int> childUpdateExtendVec = extendIdxVec(childNode->updateIdxVector,parentIdxVec);
      // Go over all rows and columns in the update matrix
    
      for (int k = 0; k < updateMatrixSize; k++){
	for (int j = 0; j < updateMatrixSize; j++){
	  int rowIdx = childUpdateExtendVec[j];
	  int colIdx = childUpdateExtendVec[k];
	  nodeFrontalMatrix(rowIdx,colIdx) += childNode->updateMatrix(j,k);
	}
      } 
      (childNode->updateMatrix).resize(0,0);
    }
  }


  void sparseMF::fast_NodeExtendAddUpdate(eliminationTree::node* root,
					  HODLR_Matrix & panelHODLR,
					  std::vector<int> & parentIdxVec)
  {
    if (root->isLeaf == true){
      panelHODLR.set_FreeMatrixMemory(true);
      panelHODLR.storeLRinTree();
      return;
    }  
    // Go over all childern
    std::vector<eliminationTree::node*> nodeChildren = root->children;
    int numChildren = nodeChildren.size();
    for (int i = 0; i < numChildren; i++){
      eliminationTree::node* childNode = nodeChildren[i]; 
      // Find update matrix extend add indices
      std::vector<int> childUpdateExtendVec = extendIdxVec(childNode->updateIdxVector,parentIdxVec);
    
      if (childNode->D_UpdateDense == true){
	msg_dbg("dense D");
	extendAddUpdate(panelHODLR,childNode->updateMatrix,childUpdateExtendVec, options.fast_LR_Tol, "PS_Boundary");
      }else{
	msg_dbg("HODLR D");
	extendAddUpdate(panelHODLR,childNode->updateU,childNode->updateV,childUpdateExtendVec, options.fast_LR_Tol, "PS_Boundary");
	extendAddUpdate(panelHODLR,childNode->D_HODLR,childUpdateExtendVec, options.fast_LR_Tol, "PS_Boundary");
      }
    }
  }


  void sparseMF::fast_NodeExtendAddUpdate_Array(eliminationTree::node* root,
						HODLR_Matrix & panelHODLR,
						std::vector<int> & parentIdxVec)
  {
    if (root->isLeaf == true)
      return;
    
    // Go over all childern
    std::vector<Eigen::MatrixXd *> U_Array,V_Array;
    std::vector<Eigen::MatrixXd *> D_Array;
    std::vector<HODLR_Matrix *>    D_HODLR_Array;
    std::vector<std::vector<int> > updateIdxVec_Array_D;
    std::vector<std::vector<int> > updateIdxVec_Array_D_HODLR;
    std::vector<eliminationTree::node*> nodeChildren = root->children;
    int numChildren = nodeChildren.size();
    for (int i = 0; i < numChildren; i++){
      eliminationTree::node* childNode = nodeChildren[i]; 
      // Find update matrix extend add indices
      std::vector<int> childUpdateExtendVec = extendIdxVec(childNode->updateIdxVector,parentIdxVec);
      if (childNode->D_UpdateDense == true){
	msg_dbg("dense D");
	D_Array.push_back(&(childNode->updateMatrix));
	updateIdxVec_Array_D.push_back(childUpdateExtendVec);
    
      }else{
	msg_dbg("HODLR D");
	U_Array.push_back(&(childNode->updateU));
	V_Array.push_back(&(childNode->updateV));
	D_HODLR_Array.push_back(&(childNode->D_HODLR));
	updateIdxVec_Array_D_HODLR.push_back(childUpdateExtendVec);
      }
    }

    extendAddUpdate(panelHODLR, D_Array, D_HODLR_Array, U_Array, V_Array,
		    updateIdxVec_Array_D, updateIdxVec_Array_D_HODLR, 
		    options.fast_LR_Tol, "PS_Boundary", options.fast_MaxRank);
    for (int i = 0; i < (int)D_Array.size(); i++){
      D_Array[i]->resize(0,0);
    }
    for (int i = 0; i < (int)D_HODLR_Array.size(); i++){
      U_Array[i]->resize(0,0);
      V_Array[i]->resize(0,0);
      D_HODLR_Array[i]->destroyAllData();
    }
  }


  std::vector<int> sparseMF::extendIdxVec(std::vector<int> & childIdxVec, 
					  std::vector<int> & parentIdxVec)
  {
    int updateMatrixSize = childIdxVec.size();
    std::vector<int> extendIdxVec(updateMatrixSize);
    std::vector<int>::iterator currStart = parentIdxVec.begin();  
    for (int i = 0; i < updateMatrixSize; i++){
      std::vector<int>::iterator iter;
      iter = std::lower_bound(currStart,parentIdxVec.end(),childIdxVec[i]);
      int extendPos = iter - parentIdxVec.begin();
      extendIdxVec[i] = extendPos;
      currStart = iter;
    }
    return extendIdxVec;
  }


  void sparseMF::LU_FactorizeMatrix()
  {
    symbolic_Factorize();
    if (LU_Factorized == false){
      Timer t0("LU_FactorizationTime");
      for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
	int currLevel = matrixElmTreePtr->numLevels - i ;
	msg_dbg("Eliminating nodes at level ",currLevel);
	std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
	for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	  eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	  LU_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
	}  
      }
  #ifndef NDEBUG
      double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
      assert(permuteErr == 0);
  #endif
      t0.stop(); // LU_FactorizationTime
      
      //Assemble L and U factors
      Timer t1("LU_AssemblyTime");
      assembleLUMatrix();
      t1.stop();
      LU_Factorized = true;
    }
  }


  void sparseMF::implicit_FactorizeMatrix()
  {
    symbolic_Factorize();
    if (implicit_Factorized ==  false){
      Timer t("implicit_FactorizationTime");
      for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
	int currLevel = matrixElmTreePtr->numLevels - i ;
	msg_dbg("Eliminating nodes at level ", currLevel);
	std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
	for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	  eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	  implicit_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
	}   
      }
  #ifndef NDEBUG
      double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
      assert(permuteErr == 0);
  #endif
      implicit_Factorized = true;
    }
  }


  void sparseMF::fast_FactorizeMatrix()
  {
    symbolic_Factorize();
    if (fast_Factorized == false){
      Timer t("fast_FactorizationTime");
      for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
	int currLevel = matrixElmTreePtr->numLevels - i ;
	msg_dbg("Eliminating nodes at level ", currLevel);
	std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
	for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	  eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	  fast_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
	}   
      }
  #ifndef NDEBUG
      double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
      assert(permuteErr == 0);
  #endif
      fast_Factorized = true;
    }
  }


  void sparseMF::assembleLUMatrix()
  {
    L_Matrix.resize(Sp_MatrixSize,Sp_MatrixSize);
    L_Matrix.setFromTriplets(L_TripletVec.begin(),L_TripletVec.end());
    U_Matrix.resize(Sp_MatrixSize,Sp_MatrixSize);
    U_Matrix.setFromTriplets(U_TripletVec.begin(),U_TripletVec.end());
      
    // Erase triplet vectors
    L_TripletVec.erase(L_TripletVec.begin(),L_TripletVec.end());
    U_TripletVec.erase(U_TripletVec.begin(),U_TripletVec.end());
    LU_Factorized = true;
    
    // Test factorization
    if (testResults == true)
      test_LU_Factorization();
  }

  void sparseMF::test_LU_Factorization()
  {
    msg("Testing factorization....");
    Eigen::SparseMatrix<double> reconstructedMatrix = (L_Matrix * U_Matrix).pruned(1e-20);
    double absError = (reconstructedMatrix - reorderedMatrix).norm();
    double relError = absError/reorderedMatrix.norm();
    msg("Absolute Error = ", absError);
    msg("Relative Error = ", relError);
  }


  Eigen::MatrixXd sparseMF::LU_Solve(const Eigen::MatrixXd& inputRHS)
  {
    if (LU_Factorized == false)
      LU_FactorizeMatrix();
    
    Timer t("permTime");
    Eigen::MatrixXd permutedRHS = permuteRows(inputRHS, permVector, false);
    t.stop();

    
    Timer* t0 = new Timer("LU_SolveTime");
    Eigen::MatrixXd L_Soln = L_Matrix.triangularView<Eigen::UnitLower>().solve(permutedRHS);
    Eigen::MatrixXd U_Soln = U_Matrix.triangularView<Eigen::Upper>().solve(L_Soln);
    delete t0;

    t.start();
    Eigen::MatrixXd result = permuteRows(U_Soln,permVector,true);
    t.stop();

    double LU_TotalTime = get_time("matrixReorderingTime") + 
			  get_time("LU_FactorizationTime") + 
			  get_time("symbolic_FactorizationTime") + 
			  get_time("LU_AssemblyTime") + 
			  get_time("LU_SolveTime") + t.elapsed();

    if (printResultInfo){
      std::cout<<"**************************************************"<<std::endl;
      std::cout<<"Solver Type                           = "<<"LU"<<std::endl;
      std::cout<<"Average Large Front Size              = "<<averageLargeFrontSize<<std::endl;
      std::cout<<"Number of Large Fronts                = "<<numLargeFronts<<std::endl;
      std::cout<<"Matrix Reordering Time                = "<<get_time("matrixReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"     Matrix Graph Conversion Time     = "<<get_time("matrixGraphConversionTime")<<" seconds"<<std::endl;
      std::cout<<"     SCOTCH Reordering Time           = "<<get_time("SCOTCH_ReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"Factorization Time                    = "<<get_time("LU_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"     Extend Add Time                  = "<<get_time("LU_ExtendAddTime")<<" seconds"<<std::endl;   
      std::cout<<"Symbolic Factorization Time           = "<<get_time("symbolic_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"LU Assembly Time                      = "<<get_time("LU_AssemblyTime")<<" seconds"<<std::endl;
      std::cout<<"Solve Time                            = "<<get_time("LU_SolveTime")<<" seconds"<<std::endl;
      std::cout<<"Total Solve Time                      = "<<LU_TotalTime<<" seconds"<<std::endl;
      std::cout<<"Residual l2 Relative Error            = "<<((reorderedMatrix * U_Soln) - permutedRHS).norm()/permutedRHS.norm()<<std::endl;
    }
    return result;
  }


  Eigen::MatrixXd sparseMF::implicit_Solve(const Eigen::MatrixXd& inputRHS)
  {
    if (!implicit_Factorized)
      implicit_FactorizeMatrix();
    
    Timer t("permTime");
    Eigen::MatrixXd permutedRHS = permuteRows(inputRHS, permVector, false);
    t.stop();

    Timer* t0 = new Timer("implicit_SolveTime");
    Eigen::MatrixXd fast_UpwardPass_Soln = implicit_UpwardPass(permutedRHS);
    //std::cout<<"Upward pass completed. Attempting downward pass..."<<std::endl;
    Eigen::MatrixXd finalSoln = implicit_DownwardPass(fast_UpwardPass_Soln);
    //std::cout<<"Downward pass completed"<<std::endl;
    delete t0;
    
    t.start();
    Eigen::MatrixXd result = permuteRows(finalSoln,permVector,true);
    t.stop();
  
    double implicit_TotalTime = get_time("matrixReorderingTime") + 
				get_time("implicit_FactorizationTime") + 
				get_time("symbolic_FactorizationTime") + 
				get_time("implicit_SolveTime") + t.elapsed();
    
    if (printResultInfo){
      std::cout<<"**************************************************"<<std::endl;
      std::cout<<"Solver Type                           = "<<"Implicit"<<std::endl;
      std::cout<<"Average Large Front Size              = "<<averageLargeFrontSize<<std::endl;
      std::cout<<"Number of Large Fronts                = "<<numLargeFronts<<std::endl;
      std::cout<<"Low-Rank Tolerance                    = "<<options.fast_LR_Tol<<std::endl;
      std::cout<<"Matrix Reordering Time                = "<<get_time("matrixReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"     Matrix Graph Conversion Time     = "<<get_time("matrixGraphConversionTime")<<" seconds"<<std::endl;
      std::cout<<"     SCOTCH Reordering Time           = "<<get_time("SCOTCH_ReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"Factorization Time                    = "<<get_time("implicit_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"     Extend Add Time                  = "<<get_time("implicit_ExtendAddTime")<<" seconds"<<std::endl;   
      std::cout<<"Symbolic Factorization Time           = "<<get_time("symbolic_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"Solve Time                            = "<<get_time("implicit_SolveTime")<<" seconds"<<std::endl;
      std::cout<<"Total Solve Time                      = "<<implicit_TotalTime<<" seconds"<<std::endl;
      std::cout<<"Residual l2 Relative Error            = "<<((reorderedMatrix * finalSoln) - permutedRHS).norm()/permutedRHS.norm()<<std::endl;
    }
    return result;
  }


  Eigen::MatrixXd sparseMF::fast_Solve(const Eigen::MatrixXd& inputRHS)
  {
    if (!fast_Factorized)
      fast_FactorizeMatrix();
    
    Timer t("permTime");
    Eigen::MatrixXd permutedRHS = permuteRows(inputRHS,permVector,false);
    t.stop();
    
    Timer* t0 = new Timer("fast_SolveTime");
    Eigen::MatrixXd ultraSolve_UpwardPass_Soln = fast_UpwardPass(permutedRHS);
    //std::cout<<"Upward pass completed. Attempting downward pass..."<<std::endl;
    Eigen::MatrixXd finalSoln = fast_DownwardPass(ultraSolve_UpwardPass_Soln);
    //std::cout<<"Downward pass completed"<<std::endl;
    delete t0;
  
    t.start();
    Eigen::MatrixXd result = permuteRows(finalSoln,permVector,true);
    t.stop();
  
    double fast_TotalTime = get_time("matrixReorderingTime") + 
			    get_time("fast_FactorizationTime") + 
			    get_time("symbolic_FactorizationTime") + 
			    get_time("fast_SolveTime") + t.elapsed();
    
    if (printResultInfo == true){
      std::cout<<"**************************************************"<<std::endl;
      std::cout<<"Solver Type                           = "<<"Ultra Fast"<<std::endl;
      std::cout<<"Average Large Front Size              = "<<averageLargeFrontSize<<std::endl;
      std::cout<<"Number of Large Fronts                = "<<numLargeFronts<<std::endl;
      std::cout<<"Low-Rank Tolerance                    = "<<options.fast_LR_Tol<<std::endl;
      std::cout<<"Boundary Depth                        = "<<options.fast_BoundaryDepth<<std::endl;
      std::cout<<"Matrix Reordering Time                = "<<get_time("matrixReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"     Matrix Graph Conversion Time     = "<<get_time("matrixGraphConversionTime")<<" seconds"<<std::endl;
      std::cout<<"     SCOTCH Reordering Time           = "<<get_time("SCOTCH_ReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"Factorization Time                    = "<<get_time("fast_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"     Extend Add Time                  = "<<get_time("fast_ExtendAddTime")<<" seconds"<<std::endl;   
      std::cout<<"Symbolic Factorization Time           = "<<get_time("symbolic_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"Solve Time                            = "<<get_time("fast_SolveTime")<<" seconds"<<std::endl;
      std::cout<<"Total Solve Time                      = "<<fast_TotalTime<<" seconds"<<std::endl;
      std::cout<<"Residual l2 Relative Error            = "<<((reorderedMatrix * finalSoln) - permutedRHS).norm()/permutedRHS.norm()<<std::endl;
    }
    return result;

  }


  Eigen::MatrixXd sparseMF::implicit_UpwardPass(const Eigen::MatrixXd &inputRHS)
  {
    Eigen::MatrixXd modifiedRHS = inputRHS;
    for (int i = 1; i <  matrixElmTreePtr->numLevels; i++){
      int currLevel = matrixElmTreePtr->numLevels - i ;
      //std::cout<<"Upward pass solving nodes at level "<<currLevel<<std::endl;
	std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
	for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	  eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	  implicit_UpwardPass_Update(currNodePtr,modifiedRHS);
	}  
    }
    return modifiedRHS;
  }


  void sparseMF::implicit_UpwardPass_Update(eliminationTree::node* root,
					    Eigen::MatrixXd &modifiedRHS)
  {
    int nodeSize = root->max_Col - root->min_Col + 1;
    std::vector<int> nodeIdxVec = createSequentialVec(root->min_Col,root->max_Col-root->min_Col+1);  
    Eigen::MatrixXd node_RHS = modifiedRHS.block(root->min_Col,0,nodeSize,modifiedRHS.cols());
  
    // Update RHS
    Eigen::MatrixXd RHS_UpdateSoln  = (root->nodeMatrix_LU).solve(node_RHS);
    Eigen::MatrixXd multiply = root->updateToNode_U * RHS_UpdateSoln;
    for (unsigned int i = 0; i < root->updateIdxVector.size(); i++){
      int idx = root->updateIdxVector[i];
      modifiedRHS.row(idx) -= multiply.row(i); 
    }
  }


  Eigen::MatrixXd sparseMF::fast_UpwardPass(const Eigen::MatrixXd &inputRHS)
  {
    Eigen::MatrixXd modifiedRHS = inputRHS;
    for (int i = 1; i <  matrixElmTreePtr->numLevels; i++){
      int currLevel = matrixElmTreePtr->numLevels - i ;
      //std::cout<<"Upward pass solving nodes at level "<<currLevel<<std::endl;
	std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
	for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	  eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	  fast_UpwardPass_Update(currNodePtr,modifiedRHS);
	}  
    }
    return modifiedRHS;
  }
  

  void sparseMF::fast_UpwardPass_Update(eliminationTree::node* root,
					Eigen::MatrixXd &modifiedRHS)
  {
    int nodeSize = root->max_Col - root->min_Col + 1;
    std::vector<int> nodeIdxVec = createSequentialVec(root->min_Col,root->max_Col-root->min_Col+1);
    Eigen::MatrixXd node_RHS = modifiedRHS.block(root->min_Col,0,nodeSize,modifiedRHS.cols());
    
    // Update RHS
    Eigen::MatrixXd RHS_UpdateSoln = fast_NodeSolve(root,node_RHS);
    Eigen::MatrixXd fastMultiply = fast_UpdateToNodeMultiply(root,RHS_UpdateSoln);
    for (unsigned int i = 0; i < root->updateIdxVector.size(); i++){
      int idx = root->updateIdxVector[i];
      modifiedRHS.row(idx) -= fastMultiply.row(i); 
    }
  }


  Eigen::MatrixXd sparseMF::implicit_DownwardPass(const Eigen::MatrixXd& upwardPassRHS)
  {
    eliminationTree::node* root = matrixElmTreePtr->root;
    Eigen::MatrixXd finalSoln = Eigen::MatrixXd::Zero(upwardPassRHS.rows(),upwardPassRHS.cols());
    implicit_DownwardPass(root,upwardPassRHS,finalSoln);
    return finalSoln;
  }


  void sparseMF::implicit_DownwardPass(eliminationTree::node* root,
				       const Eigen::MatrixXd& upwardPassRHS,
				       Eigen::MatrixXd& finalSoln)
  {
    //std::cout<<"Downward pass solving nodes at level "<<root->currLevel<<std::endl;
  
    // Solve for node matrix
    int nodeSize = root->max_Col - root->min_Col + 1;
    Eigen::MatrixXd upwardPassRHS_Node = upwardPassRHS.block(root->min_Col,0,nodeSize,upwardPassRHS.cols());;  
    assert(upwardPassRHS_Node.rows() == nodeSize);
    Eigen::MatrixXd node_RHS;
    if (root->currLevel == 0){
      node_RHS = upwardPassRHS_Node;
    }else{
      Eigen::MatrixXd parentUpdate = getRowBlkMatrix(finalSoln,root->updateIdxVector);
      node_RHS  = upwardPassRHS_Node - root->nodeToUpdate_U * parentUpdate;
    }
    Eigen::MatrixXd nodeMatrixSoln = (root->nodeMatrix_LU).solve(node_RHS);
    finalSoln.block(root->min_Col,0,nodeSize,upwardPassRHS.cols()) = nodeMatrixSoln;
    
    // Do nothing if leaf
    if (root->isLeaf == true)
      return;
    
    // Downward solve children
    std::vector<eliminationTree::node*> children = root->children;
    for (unsigned int i = 0; i < children.size(); i++){
      eliminationTree::node* currNode = children[i];
      implicit_DownwardPass(currNode,upwardPassRHS,finalSoln);
    }
  }


  Eigen::MatrixXd sparseMF::fast_DownwardPass(const Eigen::MatrixXd& upwardPassRHS)
  {
    eliminationTree::node* root = matrixElmTreePtr->root;
    Eigen::MatrixXd finalSoln = Eigen::MatrixXd::Zero(upwardPassRHS.rows(),upwardPassRHS.cols());
    fast_DownwardPass(root,upwardPassRHS,finalSoln);
    return finalSoln;
  }


  void sparseMF::fast_DownwardPass(eliminationTree::node* root,
				   const Eigen::MatrixXd& upwardPassRHS,
				   Eigen::MatrixXd& finalSoln)
  {
    //std::cout<<"Downward pass solving nodes at level "<<root->currLevel<<std::endl;
  
    // Solve for node matrix
    int nodeSize = root->max_Col - root->min_Col + 1;
    Eigen::MatrixXd upwardPassRHS_Node = upwardPassRHS.block(root->min_Col,0,nodeSize,upwardPassRHS.cols());;  
    assert(upwardPassRHS_Node.rows() == nodeSize);
    Eigen::MatrixXd node_RHS;
    if (root->currLevel == 0){
      node_RHS = upwardPassRHS_Node;
    }else{
      Eigen::MatrixXd parentUpdate = getRowBlkMatrix(finalSoln,root->updateIdxVector);
      node_RHS = upwardPassRHS_Node - fast_NodeToUpdateMultiply(root,parentUpdate);
    }
    Eigen::MatrixXd nodeMatrixSoln = fast_NodeSolve(root,node_RHS);
    finalSoln.block(root->min_Col,0,nodeSize,upwardPassRHS.cols()) = nodeMatrixSoln;
    // Do nothing if leaf
    if (root->isLeaf == true)
      return;
    
    // Downward solve children
    std::vector<eliminationTree::node*> children = root->children;
    for (unsigned int i = 0; i < children.size(); i++){
      eliminationTree::node* currNode = children[i];
      fast_DownwardPass(currNode,upwardPassRHS,finalSoln);
    }
  }
  

  Eigen::MatrixXd sparseMF::getRowBlkMatrix(const Eigen::MatrixXd& inputMatrix, 
					    const std::vector<int>& inputIndex)
  {
    int numBlkRows = inputIndex.size();
    int numBlkCols = inputMatrix.cols();
    assert (numBlkRows <= inputMatrix.rows());
    Eigen::MatrixXd blkMatrix(numBlkRows,numBlkCols);
    for (int i = 0; i < numBlkRows; i++)
      blkMatrix.row(i) = inputMatrix.row(inputIndex[i]);
    return blkMatrix;
  }  


  void sparseMF::setRowBlkMatrix(const Eigen::MatrixXd &srcMatrix, 
				 Eigen::MatrixXd &destMatrix, 
				 const std::vector<int> &destIndex)
  {
    int numRows = destIndex.size();
    assert(numRows == srcMatrix.rows());
    for (int i = 0; i< numRows; i++)
      destMatrix.row(destIndex[i]) = srcMatrix.row(i);
  }


  Eigen::MatrixXd sparseMF::fast_NodeToUpdateMultiply(eliminationTree::node* root,
						      const Eigen::MatrixXd& RHS)
  {
    if (!root->nodeToUpdate_LR)
      return root->nodeToUpdate_U * RHS;
    else
      return root->nodeToUpdate_U * ((root->nodeToUpdate_V.transpose()) * RHS);
  }


  Eigen::MatrixXd sparseMF::fast_UpdateToNodeMultiply(eliminationTree::node* root,
						      const Eigen::MatrixXd& RHS)
  {
    if (!root->updateToNode_LR)
      return root->updateToNode_U * RHS;
    else
      return root->updateToNode_U * ((root->updateToNode_V.transpose()) * RHS);
  }


  Eigen::MatrixXd sparseMF::fast_NodeSolve(eliminationTree::node* root,
					   const Eigen::MatrixXd & RHS)
  {
    Eigen::MatrixXd result;
    if (root->criterion)
      result = (root->fast_NodeMatrix_HODLR).recLU_Solve(RHS);
    else
      result = (root->nodeMatrix_LU).solve(RHS);
    
    return result;
  }

  
  Eigen_IML_Vector sparseMF::solve(const Eigen_IML_Vector & other)
  {
    // Need some assetion error checking here
    //assert(fast_Factorized == true);
    return fast_Solve(other);
  }

  
  Eigen::MatrixXd sparseMF::iterative_Solve(const Eigen::MatrixXd & inputRHS, 
					    int maxIterations, 
					    double stopTolerance,
					    double LR_Tolerance)
  {
    
    //assert(input_RHS.rows() == matrixSize);
    // double prev_LRTolerance = LR_Tolerance;
    if (LR_Tolerance != options.fast_LR_Tol){
      fast_Factorized = false;
      options.fast_LR_Tol = LR_Tolerance;
    }
    bool prev_printResultInfo = printResultInfo;
    printResultInfo = false;
    
    Timer t("totalIter_SolveTime");
    Eigen::MatrixXd init_Guess = fast_Solve(inputRHS);
    Eigen_IML_Vector x0(init_Guess);
    Eigen_IML_Vector RHS(inputRHS);
    double tol = stopTolerance;
    int maxit = maxIterations,restart = 32;
    Eigen::MatrixXd H =Eigen::MatrixXd::Zero(restart+1,restart);
    Eigen::SparseMatrix<double> origMatrix = permuteRows(reorderedMatrix,permVector,true);
    origMatrix = origMatrix.transpose();
    origMatrix = (permuteRows(origMatrix,permVector,true)).transpose();
    
    int GMRESResult = GMRES(origMatrix,x0,RHS,*this,H,restart,maxit,tol);
    int num_Iter = maxit;
    t.stop();
    
    Eigen::MatrixXd result = *(&x0);
    printResultInfo = prev_printResultInfo;   

    if (printResultInfo){
      std::cout<<"**************************************************"<<std::endl;
      std::cout<<"Solver Type                           = "<<"Iterative"<<std::endl;
      std::cout<<"Average Large Front Size              = "<<averageLargeFrontSize<<std::endl;
      std::cout<<"Number of Large Fronts                = "<<numLargeFronts<<std::endl;
      std::cout<<"Low-Rank Tolerance                    = "<<options.fast_LR_Tol<<std::endl;
      std::cout<<"Boundary Depth                        = "<<options.fast_BoundaryDepth<<std::endl;
      std::cout<<"Matrix Reordering Time                = "<<get_time("matrixReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"     Matrix Graph Conversion Time     = "<<get_time("matrixGraphConversionTime")<<" seconds"<<std::endl;
      std::cout<<"     SCOTCH Reordering Time           = "<<get_time("SCOTCH_ReorderingTime")<<" seconds"<<std::endl;
      std::cout<<"Factorization Time                    = "<<get_time("fast_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"     Extend Add Time                  = "<<get_time("fast_ExtendAddTime")<<" seconds"<<std::endl;   
      std::cout<<"Symbolic Factorization Time           = "<<get_time("symbolic_FactorizationTime")<<" seconds"<<std::endl;
      std::cout<<"Num Iterations                        = "<<num_Iter<<std::endl;
      std::cout<<"Total Solve Time                      = "<<t.elapsed()<<" seconds"<<std::endl;
      std::cout<<"Residual l2 Relative Error            = "<<tol<<std::endl;
    }
    return result;
  }

} // end namespace smf
