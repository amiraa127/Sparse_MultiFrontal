#include "sparseMF.hpp"

sparseMF::sparseMF(Eigen::SparseMatrix<double> & inputSpMatrix){  
  frontID = 0;
  matrixElmTreePtr = NULL;

  matrixReorderingTime      = 0;
  SCOTCH_ReorderingTime     = 0;
  matrixGraphConversionTime = 0;


  implicit_ExtendAddTime       = 0;
  implicit_FactorizationTime   = 0;
  implicit_SolveTime           = 0;
  implicit_TotalTime           = 0;
  implicit_SymbolicFactorTime  = 0;

  ultra_ExtendAddTime      = 0;
  ultra_FactorizationTime  = 0;
  ultra_SolveTime          = 0;
  ultra_TotalTime          = 0;
  ultra_SymbolicFactorTime = 0;


  LU_FactorizationTime     = 0;
  LU_SolveTime             = 0;
  LU_TotalTime             = 0;
  LU_ExtendAddTime         = 0;
  LU_SymbolicFactorTime    = 0;
  LU_AssemblyTime          = 0;
  
  LU_Factorized           = false;
  implicit_Factorized     = false;
  ultra_Factorized        = false;

  testResults             = false;
  
  printResultInfo         = false;
  int numRows = inputSpMatrix.rows();
  int numCols = inputSpMatrix.cols();
  assert (numCols == numRows);
  Sp_MatrixSize = numRows;
  LU_Permutation = Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1);
  fast_MatrixSizeThresh = 10000;
  fast_HODLR_LeafSize = 30;
  fast_LR_Tol = 1e-5;
  fast_MinValueACA = 0;
  fast_LR_Method = "partialPiv_ACA";
  inputSpMatrix.prune(1e-40);  

  double startTime = clock();
  reorderMatrix(inputSpMatrix);
  double endTime = clock();
  matrixReorderingTime = (endTime - startTime)/CLOCKS_PER_SEC;
  

}

sparseMF:: ~sparseMF(){
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


void sparseMF::LU_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root){
  int minIdx = root->min_Col;
  int maxIdx = root->max_Col;
  int blkSize = Sp_MatrixSize - minIdx;
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
  /*
  if (nodeSize >= fast_MatrixSizeThresh){
    std::stringstream ss;
    ss << frontID; 
    std::string outputFileName = "front_num" + ss.str();
    saveMatrixXdToBinary(nodeMatrix,outputFileName);
    frontID ++;
  }
  */
  
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
  ultra_SymbolicFactorTime += (endTime - startTime)/CLOCKS_PER_SEC;

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
  //frontalMatrix_Sp.setFromTriplets(Sp_TripletVec.begin(),Sp_TripletVec.end());
  frontalMatrix_Sp = frontalMatrix.sparseView();
  
  
  // Update frontal matrix using updates from children
  root->D_UpdateDense = true;
  root->frontSize = frontalMatrixSize;
  root->criterion = (frontalMatrixSize >= 1000);
  if (root->criterion == true /*fast_MatrixSizeThresh*/){
    root->D_UpdateDense = false;
    HODLR_Matrix panelHODLR;
    if (root->currLevel != 0){  
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
    panelHODLR.set_LRTolerance(fast_LR_Tol);
    ultra_NodeExtendAddUpdate(root,panelHODLR,mappingVector);
    if (root->currLevel != 0){
      root->fast_NodeMatrix_HODLR = panelHODLR.topDiag();
      Eigen::MatrixXd UB    = panelHODLR.returnTopOffDiagU();
      Eigen::MatrixXd VB    = panelHODLR.returnTopOffDiagV();
      Eigen::MatrixXd KB    = panelHODLR.returnTopOffDiagK();
      Eigen::MatrixXd UC    = panelHODLR.returnBottOffDiagU();
      Eigen::MatrixXd VC    = panelHODLR.returnBottOffDiagV();
      Eigen::MatrixXd KC    = panelHODLR.returnBottOffDiagK();
      root->updateU         = -UC * KC;
      root->updateV         = (VC.transpose() * root->fast_NodeMatrix_HODLR.recLU_Solve(UB) * KB * VB.transpose()).transpose();
      root->D_HODLR         = panelHODLR.bottDiag();
      root->nodeToUpdate_U  = UB * KB;
      root->nodeToUpdate_V  = VB;
      root->updateToNode_U  = UC * KC;
      root->updateToNode_V  = VC;
      root->nodeToUpdate_LR = true;
      root->updateToNode_LR = true;
      
    }else{
      root->fast_NodeMatrix_HODLR = panelHODLR;
    }
    (root->fast_NodeMatrix_HODLR).recLU_Compute();
  }else{
    startTime = clock();
    nodeExtendAddUpdate(root,frontalMatrix,mappingVector);
    endTime = clock();
    ultra_ExtendAddTime += (endTime - startTime)/CLOCKS_PER_SEC;
    Eigen::MatrixXd updateSoln,updateMatrix;
    Eigen::MatrixXd nodeToUpdate_U;
    Eigen::MatrixXd updateToNode_U;
    
    // Create update matrices
    Eigen::MatrixXd nodeMatrix   = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
    Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
    Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);
    Eigen::PartialPivLU<Eigen::MatrixXd> fast_NodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(nodeMatrix);
    (root->fast_NodeMatrix_LU) = (fast_NodeMatrix_LU).matrixLU();
    (root->fast_NodeMatrix_P ) = (fast_NodeMatrix_LU).permutationP();
    if (root->currLevel != 0){
      updateSoln = (fast_NodeMatrix_LU).solve(nodeToUpdate);
      root->nodeToUpdate_LR = false;
      root->updateToNode_LR = false;
      nodeToUpdate_U = nodeToUpdate;
      updateToNode_U = updateToNode;
      root->nodeToUpdate_U  = nodeToUpdate_U;
      root->updateToNode_U  = updateToNode_U;
      ultra_CreateUpdateMatrixForNode(root,updateSoln,frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize));
    }
  }
  root->updateIdxVector = updateIdxVector;
};

void sparseMF::implicit_CreateFrontalAndUpdateMatrixFromNode(eliminationTree::node* root){

  int minIdx = root->min_Col;
  int maxIdx = root->max_Col;
  int blkSize = Sp_MatrixSize - minIdx;
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
  implicit_SymbolicFactorTime += (endTime - startTime)/CLOCKS_PER_SEC;

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
  implicit_ExtendAddTime += (endTime - startTime)/CLOCKS_PER_SEC;
  
  Eigen::MatrixXd updateSoln,updateMatrix;
  Eigen::MatrixXd nodeToUpdate_U,nodeToUpdate_V;
  Eigen::MatrixXd updateToNode_U,updateToNode_V;
  // Create update matrices
  Eigen::MatrixXd nodeMatrix = frontalMatrix.topLeftCorner(nodeSize,nodeSize);
  Eigen::MatrixXd nodeToUpdate = frontalMatrix.topRightCorner(nodeSize,updateMatrixSize);
  Eigen::MatrixXd updateToNode = frontalMatrix.bottomLeftCorner(updateMatrixSize,nodeSize);

  //root->criterion = false;

  Eigen::PartialPivLU<Eigen::MatrixXd> fast_NodeMatrix_LU = Eigen::PartialPivLU<Eigen::MatrixXd>(nodeMatrix);
  (root->fast_NodeMatrix_LU) = (fast_NodeMatrix_LU).matrixLU();
  (root->fast_NodeMatrix_P) = (fast_NodeMatrix_LU).permutationP();
  if (root->currLevel != 0){
    updateSoln = (fast_NodeMatrix_LU).solve(nodeToUpdate);
    nodeToUpdate_U = nodeToUpdate;
    updateToNode_U = updateToNode;
  }
  
  root->updateIdxVector = updateIdxVector;
  root->nodeToUpdate_U = nodeToUpdate_U;
  root->updateToNode_U = updateToNode_U;
  root->updateMatrix =  frontalMatrix.bottomRightCorner(updateMatrixSize,updateMatrixSize) - (root->updateToNode_U) * updateSoln;
     
};

void sparseMF::ultra_CreateUpdateMatrixForNode(eliminationTree::node* root,const Eigen::MatrixXd & nodeUpdateSoln,const Eigen::MatrixXd & bottomRightMatrix){
  Eigen::MatrixXd updateMatrix;
  if ((root->updateToNode_LR == false) && (root->nodeToUpdate_LR == false)){
    updateMatrix = bottomRightMatrix - (root->updateToNode_U) * nodeUpdateSoln;
  }else if ((root->updateToNode_LR == true) && (root->nodeToUpdate_LR == true)){
    updateMatrix = bottomRightMatrix - (root->updateToNode_U * root->updateToNode_V.transpose() * nodeUpdateSoln * root->nodeToUpdate_V.transpose());
  }else{
    std::cout<<"Error! Wrong combination of booleans."<<std::endl;
    exit(EXIT_FAILURE);
  }
  root->updateMatrix = updateMatrix;
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

void sparseMF::nodeExtendAddUpdate(eliminationTree::node* root,Eigen::MatrixXd & nodeFrontalMatrix,std::vector<int> & parentIdxVec){
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
    std::vector<int> childUpdateExtendVec = extendIdxVec(childNode->updateIdxVector,parentIdxVec);
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

void sparseMF::ultra_NodeExtendAddUpdate(eliminationTree::node* root,HODLR_Matrix & panelHODLR,std::vector<int> & parentIdxVec){
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
      std::cout<<"dense D"<<std::endl;
      panelHODLR.extendAddUpdate(childNode->updateMatrix,childUpdateExtendVec,fast_LR_Tol,"Compress_LU");
    }else{
      std::cout<<"HODLR D"<<std::endl;
      panelHODLR.extendAddUpdate(childNode->updateU,childNode->updateV,childUpdateExtendVec,fast_LR_Tol,"Compress_LU");
      panelHODLR.extendAddUpdate(childNode->D_HODLR,childUpdateExtendVec,fast_LR_Tol,"Compress_LU");
    }
  }
}

std::vector<int> sparseMF::extendIdxVec(std::vector<int> & childIdxVec, std::vector<int> & parentIdxVec){
  int updateMatrixSize = childIdxVec.size();
  std::vector<int> extendIdxVec(updateMatrixSize);
  for (int i = 0; i < updateMatrixSize; i++){
    std::vector<int>::iterator iter;
    iter = std::lower_bound(parentIdxVec.begin(),parentIdxVec.end(),childIdxVec[i]);
    int extendPos = iter - parentIdxVec.begin();
    extendIdxVec[i] = extendPos;
  }
  return extendIdxVec;
}

void sparseMF::LU_FactorizeMatrix(){
  for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
      int currLevel = matrixElmTreePtr->numLevels - i ;
      std::cout<<"Eliminating nodes at level "<<currLevel<<std::endl;
      std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
      for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	LU_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
      }  
    }
  double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
  assert(permuteErr == 0);
}


void sparseMF::implicit_FactorizeMatrix(){
  double startTime = clock();
  for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
      int currLevel = matrixElmTreePtr->numLevels - i ;
      std::cout<<"Eliminating nodes at level "<<currLevel<<std::endl;
      std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
      for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	implicit_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
      }   
  }
  double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
  assert(permuteErr == 0);
  implicit_Factorized = true;
  double endTime = clock();
  implicit_FactorizationTime = (endTime - startTime)/CLOCKS_PER_SEC;
}

void sparseMF::ultra_FactorizeMatrix(){
  double startTime = clock();
  //bool criterion = false;
  for (int i = 1; i <=  matrixElmTreePtr->numLevels; i++){
    int currLevel = matrixElmTreePtr->numLevels - i ;
    std::cout<<"Eliminating nodes at level "<<currLevel<<std::endl;
    std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
    for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
      eliminationTree::node* currNodePtr = currLevelNodesVec[j];
      //currNodePtr->criterion = criterion; 
      ultra_CreateFrontalAndUpdateMatrixFromNode(currNodePtr);
      //if (currNodePtr->frontSize > 500)
      //	criterion = true;
    }   
  }
  double permuteErr = (LU_Permutation -  Eigen::VectorXd::LinSpaced(Eigen::Sequential,Sp_MatrixSize,0,Sp_MatrixSize - 1)).norm();
  assert(permuteErr == 0);
  ultra_Factorized = true;
  double endTime = clock();
  ultra_FactorizationTime = (endTime - startTime)/CLOCKS_PER_SEC;
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
    test_LU_Factorization();

}

void sparseMF::test_LU_Factorization(){
  std::cout<<"Testing factorization...."<<std::endl;
  Eigen::SparseMatrix<double> reconstructedMatrix = (L_Matrix * U_Matrix).pruned(1e-20);
  double absError = (reconstructedMatrix - reorderedMatrix).norm();
  double relError = absError/reorderedMatrix.norm();
  std::cout<<"Absolute Error = "<<absError<<std::endl;
  std::cout<<"Relative Error = "<<relError<<std::endl;
}

void sparseMF::LU_Compute(){
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

Eigen::MatrixXd sparseMF::LU_Solve(const Eigen::MatrixXd & inputRHS){
  if (LU_Factorized == false){
    LU_Compute();
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
    std::cout<<"Solver Type                           = "<<"LU"<<std::endl;
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

Eigen::MatrixXd sparseMF::implicit_Solve(const Eigen::MatrixXd & inputRHS){
  if (implicit_Factorized == false){
    implicit_FactorizeMatrix();
  }
  double permTime;
  double startTime = clock();
  Eigen::MatrixXd permutedRHS = permuteRows(inputRHS,permVector,false);
  double endTime = clock();
  permTime = (endTime - startTime)/CLOCKS_PER_SEC;

  startTime = clock();
  Eigen::MatrixXd fast_UpwardPass_Soln = implicit_UpwardPass(permutedRHS);
  std::cout<<"Upward pass completed. Attempting downward pass..."<<std::endl;
  Eigen::MatrixXd finalSoln = implicit_DownwardPass(fast_UpwardPass_Soln);
  std::cout<<"Downward pass completed"<<std::endl;
  endTime = clock();
  implicit_SolveTime = (endTime - startTime)/CLOCKS_PER_SEC;
 
  startTime = clock();
  Eigen::MatrixXd result = permuteRows(finalSoln,permVector,true);
  endTime = clock();
  permTime += (endTime - startTime)/CLOCKS_PER_SEC;
 
  implicit_TotalTime = matrixReorderingTime + implicit_FactorizationTime + implicit_SolveTime + permTime; 
   
  if (printResultInfo == true){
    std::cout<<"**************************************************"<<std::endl;
    std::cout<<"Solver Type                           = "<<"Implicit"<<std::endl;
    std::cout<<"Low-Rank Tolerance                    = "<<fast_LR_Tol<<std::endl;
    std::cout<<"Matrix Reordering Time                = "<<matrixReorderingTime<<" seconds"<<std::endl;
    std::cout<<"     Matrix Graph Conversion Time     = "<<matrixGraphConversionTime<<" seconds"<<std::endl;
    std::cout<<"     SCOTCH Reordering Time           = "<<SCOTCH_ReorderingTime<<" seconds"<<std::endl;
    std::cout<<"Fast Factorization Time               = "<<implicit_FactorizationTime<<" seconds"<<std::endl;
    std::cout<<"     Fast Extend Add Time             = "<<implicit_ExtendAddTime<<" seconds"<<std::endl;   
    std::cout<<"     Fast Symbolic Factorization Time = "<<implicit_SymbolicFactorTime<<" seconds"<<std::endl;
    std::cout<<"Fast Solve Time                       = "<<implicit_SolveTime<<" seconds"<<std::endl;
    std::cout<<"Fast Total Solve Time                 = "<<implicit_TotalTime<<" seconds"<<std::endl;
    std::cout<<"Residula l2 Relative Error            = "<<((reorderedMatrix * finalSoln) - permutedRHS).norm()/permutedRHS.norm()<<std::endl;
  }
  return result;
}

Eigen::MatrixXd sparseMF::ultra_Solve(const Eigen::MatrixXd & inputRHS){
  if (ultra_Factorized == false){
    ultra_FactorizeMatrix();
  }
  double permTime;
  double startTime = clock();
  Eigen::MatrixXd permutedRHS = permuteRows(inputRHS,permVector,false);
  double endTime = clock();
  permTime = (endTime - startTime)/CLOCKS_PER_SEC;
  
  startTime = clock();
  Eigen::MatrixXd ultraSolve_UpwardPass_Soln = ultra_UpwardPass(permutedRHS);
  std::cout<<"Upward pass completed. Attempting downward pass..."<<std::endl;
  Eigen::MatrixXd finalSoln = ultra_DownwardPass(ultraSolve_UpwardPass_Soln);
  std::cout<<"Downward pass completed"<<std::endl;
  endTime = clock();
  ultra_SolveTime = (endTime - startTime)/CLOCKS_PER_SEC;
 
  startTime = clock();
  Eigen::MatrixXd result = permuteRows(finalSoln,permVector,true);
  endTime = clock();
  permTime += (endTime - startTime)/CLOCKS_PER_SEC;
 
  ultra_TotalTime = matrixReorderingTime + ultra_FactorizationTime + ultra_SolveTime + permTime; 
   
  if (printResultInfo == true){
    std::cout<<"**************************************************"<<std::endl;
    std::cout<<"Solver Type                           = "<<"Ultra Fast"<<std::endl;
    std::cout<<"Low-Rank Tolerance                    = "<<fast_LR_Tol<<std::endl;
    std::cout<<"Matrix Reordering Time                = "<<matrixReorderingTime<<" seconds"<<std::endl;
    std::cout<<"     Matrix Graph Conversion Time     = "<<matrixGraphConversionTime<<" seconds"<<std::endl;
    std::cout<<"     SCOTCH Reordering Time           = "<<SCOTCH_ReorderingTime<<" seconds"<<std::endl;
    std::cout<<"Fast Factorization Time               = "<<ultra_FactorizationTime<<" seconds"<<std::endl;
    std::cout<<"     Fast Extend Add Time             = "<<ultra_ExtendAddTime<<" seconds"<<std::endl;   
    std::cout<<"     Fast Symbolic Factorization Time = "<<ultra_SymbolicFactorTime<<" seconds"<<std::endl;
    std::cout<<"Fast Solve Time                       = "<<ultra_SolveTime<<" seconds"<<std::endl;
    std::cout<<"Fast Total Solve Time                 = "<<ultra_TotalTime<<" seconds"<<std::endl;
    std::cout<<"Residula l2 Relative Error            = "<<((reorderedMatrix * finalSoln) - permutedRHS).norm()/permutedRHS.norm()<<std::endl;
  }
  return result;
}


Eigen::MatrixXd sparseMF::implicit_UpwardPass(const Eigen::MatrixXd &inputRHS){
  Eigen::MatrixXd modifiedRHS = inputRHS;
  for (int i = 1; i <  matrixElmTreePtr->numLevels; i++){
    int currLevel = matrixElmTreePtr->numLevels - i ;
      std::cout<<"Upward pass solving nodes at level "<<currLevel<<std::endl;
      std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
      for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	implicit_UpwardPass_Update(currNodePtr,modifiedRHS);
      }  
  }
  return modifiedRHS;
}

void sparseMF::implicit_UpwardPass_Update(eliminationTree::node* root,Eigen::MatrixXd &modifiedRHS){
  std::vector<int> nodeIdxVec;
  for (int i = root->min_Col; i<= root->max_Col; i++)
    nodeIdxVec.push_back(i);
  Eigen::MatrixXd update_RHS = getRowBlkMatrix(modifiedRHS,root->updateIdxVector);
  Eigen::MatrixXd node_RHS = getRowBlkMatrix(modifiedRHS,nodeIdxVec);
  
  // Update RHS
  //Eigen::MatrixXd RHS_UpdateSoln = fast_NodeSolve(root,node_RHS);
  //Eigen::MatrixXd modifiedRHS_Blk = update_RHS - (root->updateToNode_U) * ((root->updateToNode_V).transpose() * RHS_UpdateSoln);
   Eigen::MatrixXd  L_soln = (root->fast_NodeMatrix_LU).triangularView<Eigen::UnitLower>().solve(root->fast_NodeMatrix_P * node_RHS);
  Eigen::MatrixXd RHS_UpdateSoln  = (root->fast_NodeMatrix_LU).triangularView<Eigen::Upper>().solve(L_soln);

  // Eigen::MatrixXd modifiedRHS_Blk = update_RHS - fast_UpdateToNodeMultiply(root,RHS_UpdateSoln);
  Eigen::MatrixXd modifiedRHS_Blk = update_RHS - root->updateToNode_U * RHS_UpdateSoln;

  setRowBlkMatrix(modifiedRHS_Blk,modifiedRHS,(root->updateIdxVector));  
  
}



Eigen::MatrixXd sparseMF::ultra_UpwardPass(const Eigen::MatrixXd &inputRHS){
  Eigen::MatrixXd modifiedRHS = inputRHS;
  for (int i = 1; i <  matrixElmTreePtr->numLevels; i++){
    int currLevel = matrixElmTreePtr->numLevels - i ;
      std::cout<<"Upward pass solving nodes at level "<<currLevel<<std::endl;
      std::vector<eliminationTree::node*> currLevelNodesVec = matrixElmTreePtr->nodeLevelVec[currLevel];
      for (unsigned int j = 0; j < currLevelNodesVec.size(); j++){
	eliminationTree::node* currNodePtr = currLevelNodesVec[j];
	ultra_UpwardPass_Update(currNodePtr,modifiedRHS);
      }  
  }
  return modifiedRHS;
}



void sparseMF::ultra_UpwardPass_Update(eliminationTree::node* root,Eigen::MatrixXd &modifiedRHS){
  std::vector<int> nodeIdxVec;
  for (int i = root->min_Col; i<= root->max_Col; i++)
    nodeIdxVec.push_back(i);
  Eigen::MatrixXd update_RHS = getRowBlkMatrix(modifiedRHS,root->updateIdxVector);
  Eigen::MatrixXd node_RHS = getRowBlkMatrix(modifiedRHS,nodeIdxVec);
  
  // Update RHS
  Eigen::MatrixXd RHS_UpdateSoln = fast_NodeSolve(root,node_RHS);
  //Eigen::MatrixXd modifiedRHS_Blk = update_RHS - (root->updateToNode_U) * ((root->updateToNode_V).transpose() * RHS_UpdateSoln);
  Eigen::MatrixXd modifiedRHS_Blk = update_RHS - fast_UpdateToNodeMultiply(root,RHS_UpdateSoln);
  setRowBlkMatrix(modifiedRHS_Blk,modifiedRHS,(root->updateIdxVector));  
  
}




Eigen::MatrixXd sparseMF::implicit_DownwardPass(const Eigen::MatrixXd & upwardPassRHS){
  eliminationTree::node* root = matrixElmTreePtr->root;
  Eigen::MatrixXd finalSoln = Eigen::MatrixXd::Zero(upwardPassRHS.rows(),upwardPassRHS.cols());
  implicit_DownwardPass(root,upwardPassRHS,finalSoln);
  return finalSoln;
}


void sparseMF::implicit_DownwardPass(eliminationTree::node* root,const Eigen::MatrixXd & upwardPassRHS,Eigen::MatrixXd & finalSoln){

  std::cout<<"Downward pass solving nodes at level "<<root->currLevel<<std::endl;
 
  // Solve for node matrix
  int nodeSize = root->max_Col - root->min_Col + 1;
  Eigen::MatrixXd upwardPassRHS_Node = upwardPassRHS.block(root->min_Col,0,nodeSize,upwardPassRHS.cols());;  
  assert(upwardPassRHS_Node.rows() == nodeSize);
  Eigen::MatrixXd node_RHS;
  if (root->currLevel == 0){
    node_RHS = upwardPassRHS_Node;
  }else{
    Eigen::MatrixXd parentUpdate = getRowBlkMatrix(finalSoln,root->updateIdxVector);
    //node_RHS = upwardPassRHS_Node - root->nodeToUpdate_U * ((root->nodeToUpdate_V.transpose()) * parentUpdate);
    //node_RHS = upwardPassRHS_Node - fast_NodeToUpdateMultiply(root,parentUpdate);
    node_RHS  = upwardPassRHS_Node - root->nodeToUpdate_U * parentUpdate;
  }
  //Eigen::MatrixXd nodeMatrixSoln = fast_NodeSolve(root,node_RHS);
  Eigen::MatrixXd  L_soln = (root->fast_NodeMatrix_LU).triangularView<Eigen::UnitLower>().solve(root->fast_NodeMatrix_P * node_RHS);
  Eigen::MatrixXd nodeMatrixSoln  = (root->fast_NodeMatrix_LU).triangularView<Eigen::Upper>().solve(L_soln);


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


Eigen::MatrixXd sparseMF::ultra_DownwardPass(const Eigen::MatrixXd & upwardPassRHS){
  eliminationTree::node* root = matrixElmTreePtr->root;
  Eigen::MatrixXd finalSoln = Eigen::MatrixXd::Zero(upwardPassRHS.rows(),upwardPassRHS.cols());
  ultra_DownwardPass(root,upwardPassRHS,finalSoln);
  return finalSoln;
}


void sparseMF::ultra_DownwardPass(eliminationTree::node* root,const Eigen::MatrixXd & upwardPassRHS,Eigen::MatrixXd & finalSoln){

  std::cout<<"Downward pass solving nodes at level "<<root->currLevel<<std::endl;
 
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
    ultra_DownwardPass(currNode,upwardPassRHS,finalSoln);
  }
  
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


Eigen::MatrixXd sparseMF::fast_NodeSolve(eliminationTree::node* root,const Eigen::MatrixXd & RHS){
  //int nodeMatrixSize = root->max_Col - root->min_Col + 1;
  Eigen::MatrixXd result;
  //if (nodeMatrixSize > fast_MatrixSizeThresh)
  if (root->criterion == true){
    std::cout<<root->fast_NodeMatrix_HODLR.get_MatrixSize()<<" "<<RHS.rows()<<std::endl;
    result = (root->fast_NodeMatrix_HODLR).recLU_Solve(RHS);
  }else{
    Eigen::MatrixXd  L_soln = (root->fast_NodeMatrix_LU).triangularView<Eigen::UnitLower>().solve(root->fast_NodeMatrix_P * RHS);
    result = (root->fast_NodeMatrix_LU).triangularView<Eigen::Upper>().solve(L_soln);
  }
  return result;
}
