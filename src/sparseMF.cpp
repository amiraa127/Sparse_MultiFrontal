#include "sparseMF.hpp"

sparseMF::sparseMF(Eigen::SparseMatrix<double> & inputSpMatrix){
  /*
  Eigen::MatrixXd dummy = Eigen::MatrixXd::Zero(12,12);
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(12,12);
   
  for (int i = 0; i < 12; i++)
    dummy(i,i) = 10;
  dummy(0,1) = 2; dummy(0,4) = 4;
  dummy(1,2) = -1; dummy(1,5) = -3;
  dummy(2,3) = 2; dummy(2,6) = -1;
  dummy(3,7) = 5;
  dummy(4,5) = -3; dummy(4,8) = 2;
  dummy(5,6) = -2; dummy(5,9) = -1;
  dummy(6,7) =  4; dummy(6,10) = -2; dummy(6,11) = 3 ; 
  dummy(7,11) = -1 ;
  dummy(8,9) = -3;
  dummy(9,10) = 5;
  dummy(10,11) = 2;
  
  for (int i = 0; i < 12;i++)
    for (int j = i; j < 12; j++)
      dummy(j,i) = dummy(i,j);
  inputSpMatrix = dummy.sparseView();
  std::cout<<inputSpMatrix<<std::endl;
  */
  
  frontID = 0;
  matrixElmTreePtr = NULL;
  matrixReorderingTime = 0;
  SCOTCH_ReorderingTime = 0;
  matrixGraphConversionTime = 0;
  fast_ExtendAddTime = 0;
  fast_FactorizationTime = 0;
  fast_SolveTime = 0;
  fast_TotalTime = 0;
  fast_SymbolicFactorTime = 0;
  LU_FactorizationTime = 0;
  LU_SolveTime = 0;
  LU_TotalTime = 0;
  LU_ExtendAddTime = 0;
  LU_SymbolicFactorTime = 0;
  LU_AssemblyTime = 0;
  LU_Factorized = false;
  fast_Factorized = false;
  testResults = false;
  printResultInfo = false;
  int numRows = inputSpMatrix.rows();
  int numCols = inputSpMatrix.cols();
  assert (numCols == numRows);
  Sp_MatrixSize = numRows;
  LU_Permutation = Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1);
  fast_MatrixSizeThresh = 10000;
  fast_HODLR_LeafSize = 30;
  fast_LR_Tol = 1e-2;
  fast_MinValueACA = 0;
  fast_LR_Method = "partialPiv_ACA";
  inputSpMatrix.prune(1e-40);  

  double startTime = clock();
  reorderMatrix(inputSpMatrix);
  double endTime = clock();
  matrixReorderingTime = (endTime - startTime)/CLOCKS_PER_SEC;
  

}

sparseMF:: ~sparseMF(){
  std::cout<<"destructor"<<std::endl;
  if (matrixElmTreePtr != NULL)
    delete(matrixElmTreePtr);
}

void sparseMF::reorderMatrix(Eigen::SparseMatrix<double> & inputSpMatrix){
  
  // **********Convert Matrix into SCOTCH_Graph********                   
  double startTime = clock();
  
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
  if (graphPtr == NULL){
    std::cout<<"Error! Could not allocate graph."<<std::endl;
    exit(EXIT_FAILURE);
  }

  if (SCOTCH_graphInit(graphPtr) != 0){
    std::cout<<"Error! Could not initialize graph."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  if (SCOTCH_graphBuild(graphPtr,baseval,vertnbr,verttab,verttab + 1,NULL,NULL,edgenbr,edgetab,NULL) !=0){
    std::cout<<"Error! Failed to build graph."<<std::endl;
    exit(EXIT_FAILURE);
  }

  if (SCOTCH_graphCheck(graphPtr) !=0){
    std::cout<<"Error! Graph inconsistent."<<std::endl;
    exit(EXIT_FAILURE);
  } 
 
  double endTime = clock();
  matrixGraphConversionTime = (endTime - startTime)/CLOCKS_PER_SEC;

  
  // Find graph statistics                                                               
  SCOTCH_Num numVertices,numEdges;
  SCOTCH_graphSize(graphPtr,&numVertices,&numEdges);
  std::cout<<numVertices<<" "<<numEdges<<std::endl;

  //******************** Order graph**********************                                                                                                                                                                   
  // Initialize ordering strategy                                                         
  SCOTCH_Strat* orderingStratPtr = SCOTCH_stratAlloc() ;
  if(SCOTCH_stratInit(orderingStratPtr) != 0){
    std::cout<<"Error! Could not initialize ordering strategy."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  //std::string orderingStratStr = "n{sep=m{vert=50,low=h{pass=10},asc=f{bal=0.1}}|m{vert=50,low=h{pass=10},asc=f{bal=0.1}},ole=s,ose=g}";
  //std::string orderingStratStr = "n{sep=/(vert>2000)?m{vert=50,low=h{pass=10},asc=f{bal=0.1}}|m{vert=50,low=h{pass=10},asc=f{bal=0.1}};,ole=s,ose=g}"; 

  std::string orderingStratStr =  "c{rat=0.7,cpr=n{sep=/(vert>120)?m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}}|m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}};,ole=f{cmin=0,cmax=100000,frat=0.0},ose=g},unc=n{sep=/(vert>120)?(m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}})|m{rat=0.8,vert=100,low=h{pass=10},asc=f{bal=0.2}};,ole=f{cmin=15,cmax=100000,frat=0.08},ose=g}}";
  if(SCOTCH_stratGraphOrder(orderingStratPtr , orderingStratStr.c_str()) != 0){
    std::cout<<"Error! Could not set strategy string."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  
  // Initialize variables                                                           
  SCOTCH_Num* permtab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
  SCOTCH_Num* peritab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
  SCOTCH_Num* treetab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
  SCOTCH_Num* rangtab = (SCOTCH_Num*)calloc((numVertices+1),sizeof(SCOTCH_Num));
  SCOTCH_Num* cblknbr = (SCOTCH_Num*)calloc(1,sizeof(SCOTCH_Num));

  
  // Reorder graph            
  startTime = clock();
  std::cout<<"reordering graph ..."<<std::endl;
  if (SCOTCH_graphOrder(graphPtr, orderingStratPtr, permtab, peritab, cblknbr, rangtab, treetab) != 0){
    std::cout<<"Error! Graph ordering failed."<<std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout<<"reordering complete"<<std::endl;
  endTime = clock();
  SCOTCH_ReorderingTime = (endTime - startTime)/CLOCKS_PER_SEC;

  // Permute rows and columns of the original sparse matrix and RHS                   
  permVector = std::vector<int>(permtab, permtab + numVertices);
  reorderedMatrix =  permuteRowsCols(inputSpMatrix, permVector);
  
  // Create elimination tree  
  int numBlocks = *cblknbr;
  std::cout<<numBlocks<<std::endl;
  std::vector<int> rangVector(rangtab, rangtab + numBlocks + 1);
  std::vector<int> treeVector(treetab, treetab + numBlocks);
  std::cout<<"Creating elimination tree.."<<std::endl;
  int* col_Ptr = reorderedMatrix.outerIndexPtr();
  int* row_Ind = reorderedMatrix.innerIndexPtr();
 //matrixElmTreePtr = new eliminationTree(numVertices,numBlocks,rangVector,treeVector);     
  matrixElmTreePtr = new eliminationTree(col_Ptr,row_Ind,numVertices);

  std::cout<<"Elimination tree created successfully."<<std::endl;


  //eliminationTree test;
 
  
  //test.test(col_Ptr,row_Ind,numVertices);

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


void sparseMF::createFrontalAndUpdateMatrixFromNode(eliminationTree::node* root){

  int minIdx = root->min_Col;
  int maxIdx = root->max_Col;
  int blkSize = Sp_MatrixSize - minIdx;
  //std::cout<<"currLevel = "<<root->currLevel<<std::endl;
  //std::cout<<"minIdx = "<<minIdx<<" maxIdx = "<<maxIdx<<std::endl;
  int nodeSize = root->numCols;
  Eigen::SparseMatrix<double> colMatrix = reorderedMatrix.block(minIdx,minIdx,blkSize,nodeSize);
  Eigen::SparseMatrix<double> rowMatrix = reorderedMatrix.block(minIdx,minIdx,nodeSize,blkSize);
  Eigen::MatrixXd LU_Permutation_Blk = LU_Permutation.block(minIdx,0,nodeSize,1);
  std::set<int> idxSet;
  std::map<int,int> idxMap;
  // Find the set of connected indices                                       
                                                                            
  // Row connection                                                                        
  for (int k = 0; k < rowMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(rowMatrix,k); it; ++it)     
      idxSet.insert(it.col() + minIdx);
      
  // Col connection                                                                              
  for (int k = 0; k < colMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(colMatrix,k); it; ++it)
      idxSet.insert(it.row() + minIdx);
  double startTime = clock();    
  updateNodeIdxWithChildrenFillins(root,idxSet);
  double endTime = clock();
  LU_SymbolicFactorTime += (endTime - startTime)/CLOCKS_PER_SEC;
  for (int k = 0; k < rowMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(rowMatrix,k); it; ++it)     
      idxSet.insert(it.col() + minIdx);
  
  
  // Create Index Map                                                                   
  std::vector<int> mappingVector(idxSet.begin(),idxSet.end());
  std::sort(mappingVector.begin(),mappingVector.end());
  assert(mappingVector[0] == minIdx);
  assert(mappingVector[nodeSize - 1] == maxIdx);
  std::vector<int> updateIdxVector(mappingVector.begin() + nodeSize,mappingVector.end());
  int frontalMatrixSize = mappingVector.size();
  int updateMatrixSize = frontalMatrixSize - nodeSize;
  for(int i = 0; i < frontalMatrixSize; i++)
    idxMap[mappingVector[i]] = i; 
  Eigen::MatrixXd frontalMatrix = Eigen::MatrixXd::Zero(frontalMatrixSize,frontalMatrixSize);

  // Assemble frontal and update matrices                              

  // Row matrix entries                                                                  
  for (int k = 0; k < rowMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(rowMatrix,k); it; ++it)
      frontalMatrix(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx]) = it.value();
 
  // Col matrix entries                                            
  for (int k = 0; k < colMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(colMatrix,k); it; ++it)
      frontalMatrix(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx]) = it.value();

  // Update frontal matrix using updates from children
  startTime = clock();
  nodeExtendAddUpdate(root,frontalMatrix,mappingVector);
  endTime = clock();
  LU_ExtendAddTime += (endTime - startTime)/CLOCKS_PER_SEC;
  
  // Create update matrices
  Eigen::MatrixXd nodeMatrix = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
  Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
  Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);
  Eigen::PartialPivLU<Eigen::MatrixXd> nodeMatrix_LU(nodeMatrix);
  Eigen::MatrixXd updateMatrix = frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize) - updateToNode * nodeMatrix_LU.solve(nodeToUpdate);
  root->updateMatrix = updateMatrix;
  root->updateIdxVector = updateIdxVector;
  //std::cout<<frontalMatrixSize<<" "<<nodeSize<<" "<<mappingVector[0]<<std::endl;
  
  // Update L and U factors
  Eigen::MatrixXd nodeMatrix_LUMatrix = nodeMatrix_LU.matrixLU();
  Eigen::MatrixXd nodeMatrix_L = Eigen::MatrixXd::Identity(nodeSize,nodeSize);
  nodeMatrix_L.triangularView<Eigen::StrictlyLower>() = nodeMatrix_LUMatrix.triangularView<Eigen::StrictlyLower>();
  Eigen::MatrixXd nodeMatrix_U = nodeMatrix_LUMatrix.triangularView<Eigen::Upper>();
  Eigen::MatrixXd nodeMatrix_P = nodeMatrix_LU.permutationP();

  Eigen::MatrixXd update_U = nodeMatrix_L.triangularView<Eigen::UnitLower>().solve(nodeToUpdate);
  Eigen::MatrixXd update_L = ((nodeMatrix_U.transpose()).triangularView<Eigen::Lower>().solve(updateToNode.transpose())).transpose();

  // Update LU_Permutation 
  LU_Permutation.block(minIdx,0,nodeSize,1) = nodeMatrix_P * LU_Permutation_Blk;
  // Assemble L and U factors 
  assembleUFactor(nodeMatrix_U,update_U,mappingVector);
  assembleLFactor(nodeMatrix_L,update_L,mappingVector);
  
  //Special Operations :DD
  if (nodeSize >= fast_MatrixSizeThresh){
    std::stringstream ss;
    ss << frontID; 
    std::string outputFileName = "front_num" + ss.str();
    saveMatrixXdToBinary(nodeMatrix,outputFileName);
    frontID ++;
  }
  //std::cout<<"**********************"<<std::endl;
};


void sparseMF::ultra_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root){

  int minIdx = root->min_Col;
  int maxIdx = root->max_Col;
  int blkSize = Sp_MatrixSize - minIdx;
  //std::cout<<"currLevel = "<<root->currLevel<<std::endl;
  //std::cout<<"minIdx = "<<minIdx<<" maxIdx = "<<maxIdx<<std::endl;
  int nodeSize = root->numCols;
  Eigen::SparseMatrix<double> colMatrix = reorderedMatrix.block(minIdx,minIdx,blkSize,nodeSize);
  Eigen::SparseMatrix<double> rowMatrix = reorderedMatrix.block(minIdx,minIdx,nodeSize,blkSize);
  Eigen::MatrixXd LU_Permutation_Blk = LU_Permutation.block(minIdx,0,nodeSize,1);
  std::set<int> idxSet;
  std::map<int,int> idxMap;
  // Find the set of connected indices                                       
                                                                            
  // Row connection                                                                        
  for (int k = 0; k < rowMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(rowMatrix,k); it; ++it)     
      idxSet.insert(it.col() + minIdx);
      
  // Col connection                                                                              
  for (int k = 0; k < colMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(colMatrix,k); it; ++it)
      idxSet.insert(it.row() + minIdx);
  
  double startTime = clock();
  updateNodeIdxWithChildrenFillins(root,idxSet);
  double endTime = clock();
  fast_SymbolicFactorTime += (endTime - startTime)/CLOCKS_PER_SEC;

  // Create Index Map                                                                   
  std::vector<int> mappingVector(idxSet.begin(),idxSet.end());
  std::sort(mappingVector.begin(),mappingVector.end());
  assert(mappingVector[0] == minIdx);
  assert(mappingVector[nodeSize - 1] == maxIdx);
  std::vector<int> updateIdxVector(mappingVector.begin() + nodeSize,mappingVector.end());
  int frontalMatrixSize = mappingVector.size();
  int updateMatrixSize = frontalMatrixSize - nodeSize;
  for(int i = 0; i < frontalMatrixSize; i++)
    idxMap[mappingVector[i]] = i; 
  Eigen::MatrixXd frontalMatrix = Eigen::MatrixXd::Zero(frontalMatrixSize,frontalMatrixSize);
  Eigen::SparseMatrix<double> frontalMatrix_Sp(frontalMatrixSize,frontalMatrixSize); 
  std::vector<Eigen::Triplet<double,int> > Sp_TripletVec;
  Eigen::SparseMatrix<double> B = rowMatrix.block(0,nodeSize,nodeSize,updateMatrixSize);
  //std::cout<<B.nonZeros()<<std::endl;
  HODLR_Matrix B_HODLR(B);
  Eigen::MatrixXd W,V,K;
  int calculatedRank;
  B_HODLR.PS_LowRankApprox_Sp(W,V,K,0,B.rows() - 1,0, B.cols() - 1,fast_LR_Tol,calculatedRank);
  //std::cout<<"rank = "<<calculatedRank<<" "<<W.cols()<<" "<<V.cols()<<std::endl;
  // Assemble frontal and update matrices                              

  // Row matrix entries                                                                  
  for (int k = 0; k < rowMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(rowMatrix,k); it; ++it){
      frontalMatrix(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx]) = it.value();
      Eigen::Triplet<double,int> entryTriplet(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx],it.value());
      Sp_TripletVec.push_back(entryTriplet);
    }

  // Col matrix entries                                            
  for (int k = 0; k < colMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(colMatrix,k); it; ++it){
      frontalMatrix(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx]) = it.value();
      Eigen::Triplet<double,int> entryTriplet(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx],it.value());
      Sp_TripletVec.push_back(entryTriplet);
    }
  frontalMatrix_Sp.setFromTriplets(Sp_TripletVec.begin(),Sp_TripletVec.end());
  

  // Update frontal matrix using updates from children
  bool isHODLR =  false;
  if (frontalMatrixSize > 1000 /*fast_MatrixSizeThresh*/){
    HODLR_Matrix panelHODLR;
    if (root->currLevel != 0){  
      isHODLR = true;
      user_IndexTree usrTree;
      usrTree.rootNode = new user_IndexTree::node;
      usrTree.rootNode->splitIndex          = nodeSize - 1;
      usrTree.rootNode->topOffDiag_minRank  = -1;
      usrTree.rootNode->bottOffDiag_minRank = -1;
      usrTree.rootNode->LR_Method           = "PS_Sparse";
      usrTree.rootNode->left                = NULL;
      usrTree.rootNode->right               = NULL;
      panelHODLR = HODLR_Matrix(frontalMatrix_Sp,fast_HODLR_LeafSize,usrTree);
    }else{
      panelHODLR = HODLR_Matrix(frontalMatrix_Sp,fast_HODLR_LeafSize);
    }
    if (root->isLeaf == false){
      panelHODLR.set_LRTolerance(fast_LR_Tol);
      int sumChildRanks = 0;
      std::vector<Eigen::MatrixXd*>   LR_UpdateU_PtrVec;
      std::vector<Eigen::MatrixXd*>   LR_UpdateV_PtrVec;
      std::vector<std::vector<int>* > updateIdxPtrVec;
      // Go over all childern
      std::vector<eliminationTree::node*> nodeChildren = root->children;
      int numChildren = nodeChildren.size();
      for (int i = 0; i < numChildren; i++){
	eliminationTree::node* childNode = nodeChildren[i];
	updateIdxPtrVec.push_back(&(childNode->updateIdxVector));
	LR_UpdateU_PtrVec.push_back(&(childNode->updateU));
	LR_UpdateV_PtrVec.push_back(&(childNode->updateV));
	sumChildRanks += (childNode->updateU).cols(); 
	std::cout<<(childNode->updateU).cols()<<std::endl;
      }
      std::cout<<sumChildRanks<<std::endl;
      //Extend Add Update
      panelHODLR.extendAddUpdate(mappingVector,LR_UpdateU_PtrVec,LR_UpdateV_PtrVec,updateIdxPtrVec,sumChildRanks);
      
      if (root->currLevel != 0){
	Eigen::MatrixXd UB = panelHODLR.returnTopOffDiagU();
	Eigen::MatrixXd VB = panelHODLR.returnTopOffDiagV();
	Eigen::MatrixXd KB = panelHODLR.returnTopOffDiagK();
	Eigen::MatrixXd UC = panelHODLR.returnBottOffDiagU();
	Eigen::MatrixXd VC = panelHODLR.returnBottOffDiagV();
	Eigen::MatrixXd KC = panelHODLR.returnBottOffDiagK();
	root->updateU       = UC * KC;
	root->updateV       = VB * (KB.transpose() * UB.transpose() * VC);

      }
    }
  }
  


  startTime = clock();
  fast_NodeExtendAddUpdate(root,frontalMatrix,mappingVector);
  endTime = clock();
  fast_ExtendAddTime += (endTime - startTime)/CLOCKS_PER_SEC;
  
  Eigen::MatrixXd updateSoln,updateMatrix;
  Eigen::MatrixXd nodeToUpdate_U,nodeToUpdate_V;
  Eigen::MatrixXd updateToNode_U,updateToNode_V;

  // Create update matrices
  Eigen::MatrixXd nodeMatrix = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
  Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
  Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);

  if (nodeSize > fast_MatrixSizeThresh){
    // Factorize node matrix
    root->fast_NodeMatrix_HODLR = HODLR_Matrix(nodeMatrix,fast_HODLR_LeafSize);
    (root->fast_NodeMatrix_HODLR).set_def_LRMethod(fast_LR_Method);
    (root->fast_NodeMatrix_HODLR).set_LRTolerance(fast_LR_Tol);
    (root->fast_NodeMatrix_HODLR).set_MinValueACA(fast_MinValueACA);
    (root->fast_NodeMatrix_HODLR).set_FreeMatrixMemory(true);
    (root->fast_NodeMatrix_HODLR).recLU_Compute();
    //(root->fast_NodeMatrix_HODLR).printResultInfo = true;
    //(root->fast_NodeMatrix_HODLR).printLevelAccuracy = true;
    nodeMatrix.resize(0,0);
    if (root->currLevel != 0){
      // Decompose nodeToUpdate to LR
      int nodeToUpdate_Rank;
      fastSolve_LRApprox(nodeToUpdate,nodeToUpdate_U,nodeToUpdate_V,nodeToUpdate_Rank,fast_LR_Tol,fast_LR_Method);
      root->nodeToUpdate_LR = true;

      // Decompose updateToNode to LR
      int updateToNode_Rank;
      fastSolve_LRApprox(updateToNode,updateToNode_U,updateToNode_V,updateToNode_Rank,fast_LR_Tol,fast_LR_Method);
      root->updateToNode_LR = true;

      // Obtain Update  
      updateSoln =  (root->fast_NodeMatrix_HODLR).recLU_Solve(nodeToUpdate_U);
     
    }
  }else{
    Eigen::PartialPivLU<Eigen::MatrixXd> fast_NodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(nodeMatrix);
    (root->fast_NodeMatrix_LU) = (fast_NodeMatrix_LU).matrixLU();
    (root->fast_NodeMatrix_P) = (fast_NodeMatrix_LU).permutationP();
    if (root->currLevel != 0){
      updateSoln = (fast_NodeMatrix_LU).solve(nodeToUpdate);
      root->nodeToUpdate_LR = false;
      root->updateToNode_LR = false;
      nodeToUpdate_U = nodeToUpdate;
      updateToNode_U = updateToNode;
    }
  }

  root->updateIdxVector = updateIdxVector;
  root->nodeToUpdate_U  = nodeToUpdate_U;
  root->nodeToUpdate_V  = nodeToUpdate_V;
  root->updateToNode_U  = updateToNode_U;
  root->updateToNode_V  = updateToNode_V;
  
  if (isHODLR == false){
    root->updateU     = updateToNode_U;
    root->updateV     = updateSoln.transpose();  
  }
  
  fast_CreateUpdateMatrixForNode(root,updateSoln,frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize));
  
};

void sparseMF::fast_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root){

  int minIdx = root->min_Col;
  int maxIdx = root->max_Col;
  int blkSize = Sp_MatrixSize - minIdx;
  //std::cout<<"currLevel = "<<root->currLevel<<std::endl;
  //std::cout<<"minIdx = "<<minIdx<<" maxIdx = "<<maxIdx<<std::endl;
  int nodeSize = root->numCols;
  Eigen::SparseMatrix<double> colMatrix = reorderedMatrix.block(minIdx,minIdx,blkSize,nodeSize);
  Eigen::SparseMatrix<double> rowMatrix = reorderedMatrix.block(minIdx,minIdx,nodeSize,blkSize);
  Eigen::MatrixXd LU_Permutation_Blk = LU_Permutation.block(minIdx,0,nodeSize,1);
  std::set<int> idxSet;
  std::map<int,int> idxMap;
  // Find the set of connected indices                                       
                                                                            
  // Row connection                                                                        
  for (int k = 0; k < rowMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(rowMatrix,k); it; ++it)     
      idxSet.insert(it.col() + minIdx);
      
  // Col connection                                                                              
  for (int k = 0; k < colMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(colMatrix,k); it; ++it)
      idxSet.insert(it.row() + minIdx);
  
  double startTime = clock();
  updateNodeIdxWithChildrenFillins(root,idxSet);
  double endTime = clock();
  fast_SymbolicFactorTime += (endTime - startTime)/CLOCKS_PER_SEC;

  // Create Index Map                                                                   
  std::vector<int> mappingVector(idxSet.begin(),idxSet.end());
  std::sort(mappingVector.begin(),mappingVector.end());
  assert(mappingVector[0] == minIdx);
  assert(mappingVector[nodeSize - 1] == maxIdx);
  std::vector<int> updateIdxVector(mappingVector.begin() + nodeSize,mappingVector.end());
  int frontalMatrixSize = mappingVector.size();
  int updateMatrixSize = frontalMatrixSize - nodeSize;
  for(int i = 0; i < frontalMatrixSize; i++)
    idxMap[mappingVector[i]] = i; 
  Eigen::MatrixXd frontalMatrix = Eigen::MatrixXd::Zero(frontalMatrixSize,frontalMatrixSize);

  // Assemble frontal and update matrices                              

  // Row matrix entries                                                                  
  for (int k = 0; k < rowMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(rowMatrix,k); it; ++it)
      frontalMatrix(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx]) = it.value();


  // Col matrix entries                                            
  for (int k = 0; k < colMatrix.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(colMatrix,k); it; ++it)
      frontalMatrix(idxMap[it.row() + minIdx],idxMap[it.col() + minIdx]) = it.value();

  // Update frontal matrix using updates from children
  startTime = clock();
  fast_NodeExtendAddUpdate(root,frontalMatrix,mappingVector);
  endTime = clock();
  fast_ExtendAddTime += (endTime - startTime)/CLOCKS_PER_SEC;
  
  Eigen::MatrixXd updateSoln,updateMatrix;
  Eigen::MatrixXd nodeToUpdate_U,nodeToUpdate_V;
  Eigen::MatrixXd updateToNode_U,updateToNode_V;
  // Create update matrices
  Eigen::MatrixXd nodeMatrix = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
  Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
  Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);

  if (nodeSize > fast_MatrixSizeThresh){
    // Factorize node matrix
    root->fast_NodeMatrix_HODLR = HODLR_Matrix(nodeMatrix,fast_HODLR_LeafSize);
    (root->fast_NodeMatrix_HODLR).set_def_LRMethod(fast_LR_Method);
    (root->fast_NodeMatrix_HODLR).set_LRTolerance(fast_LR_Tol);
    (root->fast_NodeMatrix_HODLR).set_MinValueACA(fast_MinValueACA);
    (root->fast_NodeMatrix_HODLR).set_FreeMatrixMemory(true);
    (root->fast_NodeMatrix_HODLR).recLU_Compute();
    //(root->fast_NodeMatrix_HODLR).printResultInfo = true;
    //(root->fast_NodeMatrix_HODLR).printLevelAccuracy = true;
    nodeMatrix.resize(0,0);
    if (root->currLevel != 0){
      // Decompose nodeToUpdate to LR
      int nodeToUpdate_Rank;
      fastSolve_LRApprox(nodeToUpdate,nodeToUpdate_U,nodeToUpdate_V,nodeToUpdate_Rank,fast_LR_Tol,fast_LR_Method);
      root->nodeToUpdate_LR = true;

      // Decompose updateToNode to LR
      int updateToNode_Rank;
      fastSolve_LRApprox(updateToNode,updateToNode_U,updateToNode_V,updateToNode_Rank,fast_LR_Tol,fast_LR_Method);
      root->updateToNode_LR = true;

      // Obtain Update  
      updateSoln =  (root->fast_NodeMatrix_HODLR).recLU_Solve(nodeToUpdate_U);
     
    }
  }else{
    Eigen::PartialPivLU<Eigen::MatrixXd> fast_NodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(nodeMatrix);
    (root->fast_NodeMatrix_LU) = (fast_NodeMatrix_LU).matrixLU();
    (root->fast_NodeMatrix_P) = (fast_NodeMatrix_LU).permutationP();
    if (root->currLevel != 0){
      updateSoln = (fast_NodeMatrix_LU).solve(nodeToUpdate);
      root->nodeToUpdate_LR = false;
      root->updateToNode_LR = false;
      nodeToUpdate_U = nodeToUpdate;
      updateToNode_U = updateToNode;
    }
  }

  root->updateIdxVector = updateIdxVector;
  root->nodeToUpdate_U = nodeToUpdate_U;
  root->nodeToUpdate_V = nodeToUpdate_V;
  root->updateToNode_U = updateToNode_U;
  root->updateToNode_V = updateToNode_V;
  fast_CreateUpdateMatrixForNode(root,updateSoln,frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize));

  //std::cout<<frontalMatrixSize<<" "<<nodeSize<<" "<<mappingVector[0]<<std::endl;
  
  
  //std::cout<<"**********************"<<std::endl;
};

void sparseMF::fast_CreateUpdateMatrixForNode(eliminationTree::node* root,const Eigen::MatrixXd & nodeUpdateSoln,const Eigen::MatrixXd & bottomRightMatrix){
  Eigen::MatrixXd updateMatrix;
  if ((root->updateToNode_LR == false) && (root->nodeToUpdate_LR == false)){
    updateMatrix = bottomRightMatrix - (root->updateToNode_U) * nodeUpdateSoln;
  }else if ((root->updateToNode_LR == true) && (root->nodeToUpdate_LR == true)){
    updateMatrix = bottomRightMatrix - (root->updateToNode_U * root->updateToNode_V.transpose() * nodeUpdateSoln * root->nodeToUpdate_V.transpose());
  }else{
    std::cout<<"Error! Wrong combination of booleans."<<std::endl;
    exit(EXIT_FAILURE);
  }
  root->fast_UpdateMatrix = updateMatrix;
  return;
}


void sparseMF::updateNodeIdxWithChildrenFillins(eliminationTree::node* root,std::set<int> & idxSet){
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
    //std::cout<<root->min_Col<<" "<<childUpdateIdxVector[0]<<" "<<childNode->min_Col<<" "<<childNode->max_Col<<std::endl;
    for (int j = 0; j < updateIdxVectorSize; j++)
      idxSet.insert(childUpdateIdxVector[j]);
  }
};
  

void sparseMF::assembleUFactor(const Eigen::MatrixXd & nodeMatrix_U, const Eigen::MatrixXd & update_U, const std::vector<int> & mappingVector){
  int nodeSize = nodeMatrix_U.cols();
  int updateSize = update_U.cols();
  assert((nodeSize + updateSize) == mappingVector.size());
  
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

void sparseMF::assembleLFactor(const Eigen::MatrixXd & nodeMatrix_L, const Eigen::MatrixXd & update_L, const std::vector<int> & mappingVector){
  int nodeSize = nodeMatrix_L.rows();
  int updateSize = update_L.rows();
  assert((nodeSize + updateSize) == mappingVector.size());
  
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

void sparseMF::nodeExtendAddUpdate(eliminationTree::node* root,Eigen::MatrixXd & nodeFrontalMatrix,std::vector<int> & nodeMappingVector){
  if (root->isLeaf == true)
    return;
  // Go over all childern
  std::vector<eliminationTree::node*> nodeChildren = root->children;
  int numChildren = nodeChildren.size();
  for (int i = 0; i < numChildren; i++){
   
    eliminationTree::node* childNode = nodeChildren[i];
    Eigen::MatrixXd childUpdateMatrix = childNode->updateMatrix;
    std::vector<int> childUpdateIdxVector = childNode->updateIdxVector;
    int updateMatrixSize = childUpdateMatrix.rows();
      
    // Find update matrix extend add indices
    std::vector<int> childUpdateExtendVec(updateMatrixSize);
    for (int j = 0; j < updateMatrixSize; j++){
      std::vector<int>::iterator iter;
      iter = std::lower_bound(nodeMappingVector.begin(),nodeMappingVector.end(),childUpdateIdxVector[j]);
      int extendPos = iter - nodeMappingVector.begin();
      childUpdateExtendVec[j] = extendPos;
    }
    
    // Go over all rows and columns in the update matrix
    for (int j = 0; j < updateMatrixSize; j++){
      for (int k = 0; k < updateMatrixSize; k++){
	int rowIdx = childUpdateExtendVec[j];
	int colIdx = childUpdateExtendVec[k];
	nodeFrontalMatrix(rowIdx,colIdx) += childUpdateMatrix(j,k);	
      }
    }
    // Free Children Memory
    (childNode->updateMatrix).resize(0,0);
  }
}

void sparseMF::fast_NodeExtendAddUpdate(eliminationTree::node* root,Eigen::MatrixXd & nodeFrontalMatrix,std::vector<int> & nodeMappingVector){

  if (root->isLeaf == true)
    return;
  // Go over all childern
  std::vector<eliminationTree::node*> nodeChildren = root->children;
  int numChildren = nodeChildren.size();
  for (int i = 0; i < numChildren; i++){
   
    eliminationTree::node* childNode = nodeChildren[i];
    Eigen::MatrixXd childUpdateMatrix = childNode->fast_UpdateMatrix;
    std::vector<int> childUpdateIdxVector = childNode->updateIdxVector;
    int updateMatrixSize = childUpdateMatrix.rows();
  
    // Find update matrix extend add indices
    std::vector<int> childUpdateExtendVec(updateMatrixSize);
    for (int j = 0; j < updateMatrixSize; j++){
      std::vector<int>::iterator iter;
      iter = std::lower_bound(nodeMappingVector.begin(),nodeMappingVector.end(),childUpdateIdxVector[j]);
      int extendPos = iter - nodeMappingVector.begin();
      childUpdateExtendVec[j] = extendPos;
    }
    // Go over all rows and columns in the update matrix
    for (int j = 0; j < updateMatrixSize; j++){
      for (int k = 0; k < updateMatrixSize; k++){
	int rowIdx = childUpdateExtendVec[j];
	int colIdx = childUpdateExtendVec[k];
	nodeFrontalMatrix(rowIdx,colIdx) += childUpdateMatrix(j,k);
      }
    }
    // Free Children Memory
    (childNode->fast_UpdateMatrix).resize(0,0);
  }
}

void sparseMF::LU_FactorizeMatrix(){
  for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
      int currLevel = matrixElmTreePtr->numLevels - i ;
      std::cout<<"Eliminating nodes at level "<<currLevel<<std::endl;
      std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
      for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	createFrontalAndUpdateMatrixFromNode(currNodePtr);
      }  
    }
  double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
  assert(permuteErr == 0);
}

void sparseMF::fast_FactorizeMatrix(){
  double startTime = clock();
  for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
      int currLevel = matrixElmTreePtr->numLevels - i ;
      std::cout<<"Eliminating nodes at level "<<currLevel<<std::endl;
      std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
      for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	//fast_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
	ultra_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
	
      }  
      
  }
  double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
  assert(permuteErr == 0);
  fast_Factorized = true;
  double endTime = clock();
  fast_FactorizationTime = (endTime - startTime)/CLOCKS_PER_SEC;
}


void sparseMF::assembleLUMatrix(){
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
    testLUFactorization();

}

void sparseMF::testLUFactorization(){
  std::cout<<"Testing factorization...."<<std::endl;
  Eigen::SparseMatrix<double> reconstructedMatrix = (L_Matrix * U_Matrix).pruned(1e-20);
  double absError = (reconstructedMatrix - reorderedMatrix).norm();
  double relError = absError/reorderedMatrix.norm();
  std::cout<<"Absolute Error = "<<absError<<std::endl;
  std::cout<<"Relative Error = "<<relError<<std::endl;
}

void sparseMF::LU_compute(){
  double startTime = clock();
  LU_FactorizeMatrix();
  double endTime = clock();
  LU_FactorizationTime = (endTime - startTime)/CLOCKS_PER_SEC;

  startTime = clock();
  assembleLUMatrix();
  endTime = clock();
  LU_AssemblyTime = (endTime - startTime)/CLOCKS_PER_SEC;

  LU_Factorized = true;
}

Eigen::MatrixXd sparseMF::LU_ExactSolve(const Eigen::MatrixXd & inputRHS){
  if (LU_Factorized == false){
    LU_compute();
  }
  double permTime;
  double startTime = clock();
  Eigen::MatrixXd permutedRHS = permuteRows(inputRHS,permVector,false);
  double endTime = clock();
  permTime = (endTime - startTime)/CLOCKS_PER_SEC;

  startTime = clock();
  Eigen::MatrixXd L_Soln = L_Matrix.triangularView<Eigen::UnitLower>().solve(permutedRHS);
  Eigen::MatrixXd U_Soln = U_Matrix.triangularView<Eigen::Upper>().solve(L_Soln);
  endTime = clock();
  LU_SolveTime = (endTime - startTime)/CLOCKS_PER_SEC;

  startTime = clock();
  Eigen::MatrixXd result = permuteRows(U_Soln,permVector,true);
  endTime = clock();

  LU_TotalTime = matrixReorderingTime + LU_FactorizationTime + LU_AssemblyTime + LU_SolveTime + permTime;

  if (printResultInfo == true){
    std::cout<<"**************************************************"<<std::endl;
    std::cout<<"Solver Type                           = "<<"LU_Explicit"<<std::endl;
    std::cout<<"Matrix Reordering Time                = "<<matrixReorderingTime<<" seconds"<<std::endl;
    std::cout<<"     Matrix Graph Conversion Time     = "<<matrixGraphConversionTime<<" seconds"<<std::endl;
    std::cout<<"     SCOTCH Reordering Time           = "<<SCOTCH_ReorderingTime<<" seconds"<<std::endl;
    std::cout<<"Factorization Time                    = "<<LU_FactorizationTime<<" seconds"<<std::endl;
    std::cout<<"     Extend Add Time                  = "<<LU_ExtendAddTime<<" seconds"<<std::endl;   
    std::cout<<"     Symbolic Factorization Time      = "<<LU_SymbolicFactorTime<<" seconds"<<std::endl;
    std::cout<<"LU Assembly Time                      = "<<LU_AssemblyTime<<" seconds"<<std::endl;
    std::cout<<"Solve Time                            = "<<LU_SolveTime<<" seconds"<<std::endl;
    std::cout<<"Total Solve Time                      = "<<LU_TotalTime<<" seconds"<<std::endl;
    std::cout<<"Residula l2 Relative Error            = "<<((reorderedMatrix * U_Soln) - permutedRHS).norm()/permutedRHS.norm()<<std::endl;
  }
  return result;
}

Eigen::MatrixXd sparseMF::fastSolve(const Eigen::MatrixXd & inputRHS){
  if (fast_Factorized == false){
    fast_FactorizeMatrix();
  }
  double permTime;
  double startTime = clock();
  Eigen::MatrixXd permutedRHS = permuteRows(inputRHS,permVector,false);
  double endTime = clock();
  permTime = (endTime - startTime)/CLOCKS_PER_SEC;

  startTime = clock();
  Eigen::MatrixXd fastSolve_UpwardPass_Soln = fastSolve_UpwardPass(permutedRHS);
  std::cout<<"Upward pass completed. Attempting downward pass..."<<std::endl;
  Eigen::MatrixXd finalSoln = fastSolve_DownwardPass(fastSolve_UpwardPass_Soln);
  std::cout<<"Downward pass completed"<<std::endl;
  endTime = clock();
  fast_SolveTime = (endTime - startTime)/CLOCKS_PER_SEC;
 
  startTime = clock();
  Eigen::MatrixXd result = permuteRows(finalSoln,permVector,true);
  endTime = clock();
  permTime += (endTime - startTime)/CLOCKS_PER_SEC;
 
  fast_TotalTime = matrixReorderingTime + fast_FactorizationTime + fast_SolveTime + permTime; 
   
  if (printResultInfo == true){
    std::cout<<"**************************************************"<<std::endl;
    std::cout<<"Solver Type                           = "<<"Fast_Implicit"<<std::endl;
    std::cout<<"Low-Rank Tolerance                    = "<<fast_LR_Tol<<std::endl;
    std::cout<<"Matrix Reordering Time                = "<<matrixReorderingTime<<" seconds"<<std::endl;
    std::cout<<"     Matrix Graph Conversion Time     = "<<matrixGraphConversionTime<<" seconds"<<std::endl;
    std::cout<<"     SCOTCH Reordering Time           = "<<SCOTCH_ReorderingTime<<" seconds"<<std::endl;
    std::cout<<"Fast Factorization Time               = "<<fast_FactorizationTime<<" seconds"<<std::endl;
    std::cout<<"     Fast Extend Add Time             = "<<fast_ExtendAddTime<<" seconds"<<std::endl;   
    std::cout<<"     Fast Symbolic Factorization Time = "<<fast_SymbolicFactorTime<<" seconds"<<std::endl;
    std::cout<<"Fast Solve Time                       = "<<fast_SolveTime<<" seconds"<<std::endl;
    std::cout<<"Fast Total Solve Time                 = "<<fast_TotalTime<<" seconds"<<std::endl;
    std::cout<<"Residula l2 Relative Error            = "<<((reorderedMatrix * finalSoln) - permutedRHS).norm()/permutedRHS.norm()<<std::endl;
  }
  return result;
}


Eigen::MatrixXd sparseMF::fastSolve_UpwardPass(const Eigen::MatrixXd &inputRHS){
  Eigen::MatrixXd modifiedRHS = inputRHS;
  for (int i = 1; i <  matrixElmTreePtr->numLevels; i++){
    int currLevel = matrixElmTreePtr->numLevels - i ;
      std::cout<<"Upward pass solving nodes at level "<<currLevel<<std::endl;
      std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
      for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	fastSolve_UpwardPass_Update(currNodePtr,modifiedRHS);
      }  
  }
  return modifiedRHS;
}

void sparseMF::fastSolve_UpwardPass_Update(eliminationTree::node* root,Eigen::MatrixXd &modifiedRHS){
  std::vector<int> nodeIdxVec;
  for (int i = root->min_Col; i<= root->max_Col; i++)
    nodeIdxVec.push_back(i);
  Eigen::MatrixXd update_RHS = getRowBlkMatrix(modifiedRHS,root->updateIdxVector);
  Eigen::MatrixXd node_RHS = getRowBlkMatrix(modifiedRHS,nodeIdxVec);
  
  // Update RHS
  Eigen::MatrixXd RHS_UpdateSoln = fastSolve_NodeSolve(root,node_RHS);
  //Eigen::MatrixXd modifiedRHS_Blk = update_RHS - (root->updateToNode_U) * ((root->updateToNode_V).transpose() * RHS_UpdateSoln);
  Eigen::MatrixXd modifiedRHS_Blk = update_RHS - fast_UpdateToNodeMultiply(root,RHS_UpdateSoln);
  setRowBlkMatrix(modifiedRHS_Blk,modifiedRHS,(root->updateIdxVector));  
  
}

Eigen::MatrixXd sparseMF::getRowBlkMatrix(const Eigen::MatrixXd & inputMatrix, const std::vector<int> & inputIndex){
  int numBlkRows = inputIndex.size();
  int numBlkCols = inputMatrix.cols();
  assert (numBlkRows <= inputMatrix.rows());
  Eigen::MatrixXd blkMatrix(numBlkRows,numBlkCols);
  for (int i = 0; i < numBlkRows; i++)
    blkMatrix.row(i) = inputMatrix.row(inputIndex[i]);
  return blkMatrix;
}  

void sparseMF::setRowBlkMatrix(const Eigen::MatrixXd &srcMatrix, Eigen::MatrixXd &destMatrix, const std::vector<int> &destIndex){
  int numRows = destIndex.size();
  assert(numRows == srcMatrix.rows());
  for (int i = 0; i< numRows; i++)
    destMatrix.row(destIndex[i]) = srcMatrix.row(i);

}

Eigen::MatrixXd sparseMF::fastSolve_DownwardPass(const Eigen::MatrixXd & upwardPassRHS){
  eliminationTree::node* root = matrixElmTreePtr->root;
  Eigen::MatrixXd finalSoln = Eigen::MatrixXd::Zero(upwardPassRHS.rows(),upwardPassRHS.cols());
  fastSolve_DownwardPass(root,upwardPassRHS,finalSoln);
  return finalSoln;
}


void sparseMF::fastSolve_DownwardPass(eliminationTree::node* root,const Eigen::MatrixXd & upwardPassRHS,Eigen::MatrixXd & finalSoln){

  std::cout<<"Downward pass solving nodes at level "<<root->currLevel<<std::endl;
 
  // Solve for node matrix
  int nodeSize = root->max_Col - root->min_Col + 1;
  std::vector<int> nodeIdxVec;
  for (int i = root->min_Col; i<= root->max_Col; i++)
    nodeIdxVec.push_back(i);
  Eigen::MatrixXd upwardPassRHS_Node = getRowBlkMatrix(upwardPassRHS,nodeIdxVec);  
  assert(upwardPassRHS_Node.rows() == nodeSize);
  Eigen::MatrixXd node_RHS;
  if (root->currLevel == 0){
    node_RHS = upwardPassRHS_Node;
  }else{
    Eigen::MatrixXd parentUpdate = getRowBlkMatrix(finalSoln,root->updateIdxVector);
    //node_RHS = upwardPassRHS_Node - root->nodeToUpdate_U * ((root->nodeToUpdate_V.transpose()) * parentUpdate);
    node_RHS = upwardPassRHS_Node - fast_NodeToUpdateMultiply(root,parentUpdate);
  }
  Eigen::MatrixXd nodeMatrixSoln = fastSolve_NodeSolve(root,node_RHS);
  setRowBlkMatrix(nodeMatrixSoln,finalSoln,nodeIdxVec);
  
  // Do nothing if leaf
  if (root->isLeaf == true)
    return;
  
  // Downward solve children
  std::vector<eliminationTree::node*> children = root->children;
  for (unsigned int i = 0; i < children.size(); i++){
    eliminationTree::node* currNode = children[i];
    fastSolve_DownwardPass(currNode,upwardPassRHS,finalSoln);
  }
  
}
 
Eigen::MatrixXd sparseMF::fast_NodeToUpdateMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS){
  if (root->nodeToUpdate_LR == false){
    return root->nodeToUpdate_U * RHS;
  }else{
    return root->nodeToUpdate_U * ((root->nodeToUpdate_V.transpose()) * RHS);
  }
}

Eigen::MatrixXd sparseMF::fast_UpdateToNodeMultiply(eliminationTree::node* root,const Eigen::MatrixXd & RHS){
  if (root->updateToNode_LR == false){
    return root->updateToNode_U * RHS;
  }else{
    return root->updateToNode_U * ((root->updateToNode_V.transpose()) * RHS);
  }
}


Eigen::MatrixXd sparseMF::fastSolve_NodeSolve(eliminationTree::node* root,const Eigen::MatrixXd & RHS){
  int nodeMatrixSize = root->max_Col - root->min_Col + 1;
  Eigen::MatrixXd result;
  if (nodeMatrixSize > fast_MatrixSizeThresh)
    result = (root->fast_NodeMatrix_HODLR).recLU_Solve(RHS);
  else{
    Eigen::MatrixXd  L_soln = (root->fast_NodeMatrix_LU).triangularView<Eigen::UnitLower>().solve(root->fast_NodeMatrix_P * RHS);
    result = (root->fast_NodeMatrix_LU).triangularView<Eigen::Upper>().solve(L_soln);
  }
  return result;
}
 
 void sparseMF::fastSolve_LRApprox(Eigen::MatrixXd & inputMatrix,Eigen::MatrixXd & U, Eigen::MatrixXd & V,int & calculatedRank,const double LR_Tol,const std::string input_LRMethod){

  HODLR_Matrix inputMatrix_HODLR(inputMatrix);
  inputMatrix_HODLR.set_LRTolerance(LR_Tol);
  inputMatrix_HODLR.set_MinValueACA(fast_MinValueACA);
  int numRows = inputMatrix.rows();
  int numCols = inputMatrix.cols();
  if (input_LRMethod == "fullPiv_ACA"){
    inputMatrix_HODLR.fullPivACA_LowRankApprox(U,V,0,numRows - 1,0,numCols - 1,LR_Tol,calculatedRank);
  }else if (input_LRMethod == "partialPiv_ACA"){
    inputMatrix_HODLR.partialPivACA_LowRankApprox(U,V,0,numRows - 1,0,numCols - 1,LR_Tol,calculatedRank);
  }
  else{
    std::cout<<"Error! Invalid low-rank approximation scheme."<<std::endl;
    exit(EXIT_FAILURE);
  }

}
