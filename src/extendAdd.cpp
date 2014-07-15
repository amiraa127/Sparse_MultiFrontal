#include "extendAdd.hpp"



void extend(HODLR_Tree::node* resultRoot,HODLR_Tree::node* HODLR_Root,std::vector<int> & extendIdxVec,int parentSize);
void extendAddLRinTree(HODLR_Tree::node* HODLR_Root,const Eigen::MatrixXd & updateExtendU,const Eigen::MatrixXd & updateExtendV,double tol,std::string mode);
void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,HODLR_Matrix & extendD_HODLR,HODLR_Matrix & D_HODLR,std::vector<int> & updateIdxVec,double tol,std::string mode);
void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,Eigen::MatrixXd & extendD,Eigen::MatrixXd & D,std::vector<int> & updateIdxVec,double tol);



HODLR_Matrix extend(std::vector<int> & extendIdxVec, int parentSize, HODLR_Matrix & childHODLR){
  assert(childHODLR.get_MatrixSize() == (int)extendIdxVec.size()); 
  //storeLRinTree();
  //freeDenseMatMem();
  //freeSparseMatMem();
  //extend(indexTree.rootNode,extendIdxVec,parentSize);
  //matrixSize = parentSize;
  
  childHODLR.storeLRinTree();
  HODLR_Matrix result = childHODLR;
  extend(result.get_TreeRootNode(),childHODLR.get_TreeRootNode(),extendIdxVec,parentSize);
  result.recalculateSize();
  result.freeMatrixData();
  return result;
}


void extendAddUpdate(HODLR_Matrix & parentHODLR,HODLR_Matrix & D_HODLR,std::vector<int> & updateIdxVec,double tol,std::string mode){
  parentHODLR.storeLRinTree();
  int parentSize = parentHODLR.get_MatrixSize();
  //D_HODLR.extend(updateIdxVec,matrixSize);
  HODLR_Matrix extendD_HODLR;// = extend(updateIdxVec,parentSize,D_HODLR);
  //assert(extendD_HODLR.get_MatrixSize() == parentSize);
  assert(updateIdxVec.size() == D_HODLR.get_MatrixSize());
  //extendAddLRinTree(indexTree.rootNode,D_HODLR,updateIdxVec,tol,mode);
  //freeDenseMatMem();
  //freeSparseMatMem();  
  extendAddLRinTree(parentHODLR,parentHODLR.get_TreeRootNode(),extendD_HODLR,D_HODLR,updateIdxVec,tol,mode);  
  parentHODLR.freeMatrixData();
}


void extendAddUpdate(HODLR_Matrix & parentHODLR,Eigen::MatrixXd & D,std::vector<int> & updateIdxVec,double tol,std::string mode){
  
  parentHODLR.storeLRinTree(); 
  if (mode == "Compress_LU"){
    Eigen::MatrixXd U_D,V_D;
    int D_Rank;
    ::fullPivACA_LowRankApprox(D,U_D,V_D,0,0,D.rows(),D.cols(),tol,D_Rank);
    extendAddUpdate(parentHODLR,U_D,V_D,updateIdxVec,tol,"Compress_LU");
  }else if (mode == "PS_Boundary"){
    Eigen::MatrixXd extendD;// = ::extend(updateIdxVec,matrixSize,D,0,0,D.rows(),D.cols(),"RowsCols");
    //assert(extendD.rows() == matrixSize);
    extendAddLRinTree(parentHODLR,parentHODLR.get_TreeRootNode(),extendD,D,updateIdxVec,tol);
  }else{
    std::cout<<"Error! Unknown operation mode!"<<std::endl;
    exit(EXIT_FAILURE);
  }
  // freeDenseMatMem();
  //freeSparseMatMem();
  parentHODLR.freeMatrixData();
}

void extendAddUpdate(HODLR_Matrix & parentHODLR,Eigen::MatrixXd & updateU,Eigen::MatrixXd & updateV,std::vector<int>& updateIdxVec,double tol,std::string mode){
  parentHODLR.storeLRinTree();
  int parentSize = parentHODLR.get_MatrixSize();
  Eigen::MatrixXd updateExtendU = extend(updateIdxVec,parentSize,updateU,0,0,updateU.rows(),updateU.cols(),"Rows");
  Eigen::MatrixXd updateExtendV = extend(updateIdxVec,parentSize,updateV,0,0,updateV.rows(),updateV.cols(),"Rows");
  extendAddLRinTree(parentHODLR.get_TreeRootNode(), updateExtendU,updateExtendV,tol,mode);
  //freeDenseMatMem();
  //freeSparseMatMem();
  parentHODLR.freeMatrixData();
}





 
std::vector<int> extractBlocks(const std::vector<int> & inputVec){
  std::vector<int> blkVec;
  if (inputVec.size() < 1)
    return blkVec;
  int blkIdx = 0;
  blkVec.push_back(blkIdx);
  while (blkIdx <= ((int)inputVec.size() - 2)){
    if (inputVec[blkIdx + 1] != (inputVec[blkIdx] + 1))
      blkVec.push_back(blkIdx + 1);
    blkIdx++;
  }
  if (blkVec[blkVec.size() - 1] != inputVec.size())
    blkVec.push_back(inputVec.size());
  //std::cout<<blkVec.size()<<" "<<inputVec.size()<<std::endl;
  return blkVec;
}


std::vector<int> createChildExractVec(std::vector<int> & parentRowColIdxVec, std::vector<int> updateIdxVec,int offset){
  std::vector<int> result;
  for (int i = 0; i < (int)parentRowColIdxVec.size(); i++)
    if (std::binary_search (updateIdxVec.begin(), updateIdxVec.end(), parentRowColIdxVec[i] + offset)){
      std::vector<int>::iterator low = std::lower_bound (updateIdxVec.begin(), updateIdxVec.end(), parentRowColIdxVec[i] + offset);
      result.push_back(low - updateIdxVec.begin());
    }
  return result;
}


Eigen::MatrixXd extend(std::vector<int> & extendIdxVec,int parentSize,Eigen::MatrixXd & child,int min_i,int min_j,int numRows,int numCols,std::string mode){
  Eigen::MatrixXd result;
  int max_i = min_i + numRows - 1;
  int max_j = min_j + numCols - 1;
  int child_NumRows = child.rows();
  int child_NumCols = child.cols();
  assert(max_i < child_NumRows);
  assert(max_j < child_NumCols);
  if (mode == "RowsCols"){
    result = Eigen::MatrixXd::Zero(parentSize,parentSize);
    // Go over all rows and columns in the child matrix     
    for (int i = min_i; i <= max_i; i++){
      for (int j = min_j; j <= max_j; j++){
	int rowIdx = extendIdxVec[i];
	int colIdx = extendIdxVec[j];
	result(rowIdx,colIdx) = child(i,j);
      }
    }
  }else if (mode == "Cols"){
    result = Eigen::MatrixXd::Zero(child.rows(),parentSize);
    for (int j = min_j; j <= max_j; j++){
      int colIdx = extendIdxVec[j];
      result.col(colIdx) = child.col(j);
    }
  }else if (mode == "Rows"){
    result = Eigen::MatrixXd::Zero(parentSize,child.cols());
    for (int i = min_i; i <= max_i; i++){
      int rowIdx = extendIdxVec[i];
      result.row(rowIdx) = child.row(i);
    }
  }else{
    std::cout<<"Error! Unknown operation mode."<<std::endl;
    exit(EXIT_FAILURE);
  }
  return result;
}

void extend(HODLR_Tree::node* resultRoot,HODLR_Tree::node* HODLR_Root,std::vector<int> & extendIdxVec,int parentSize){
  int min_i,min_j,max_i,max_j;
  int childSize = extendIdxVec.size();
  assert(HODLR_Root->max_i < childSize);
  assert(HODLR_Root->max_j < childSize);
 
  // Modify Indices
  if (HODLR_Root->min_i == 0)
    min_i = 0;
  else{
    min_i = extendIdxVec[HODLR_Root->min_i - 1] + 1;
  }
  if (HODLR_Root->min_j == 0)
    min_j = 0;
  else{
    min_j = extendIdxVec[HODLR_Root->min_j - 1] + 1;
  }
  
  if (HODLR_Root->max_i == childSize - 1)
    max_i = parentSize - 1;
  else{
    max_i = extendIdxVec[HODLR_Root->max_i];
  }

  if (HODLR_Root->max_j == childSize - 1)
    max_j = parentSize - 1;
  else{
    max_j = extendIdxVec[HODLR_Root->max_j];
  }
    
  int numRows = max_i - min_i + 1;
  int numCols = max_j - min_j + 1;
  Eigen::MatrixXd leafMatrix = Eigen::MatrixXd::Zero(numRows,numCols);
  if (HODLR_Root->isLeaf == true){
    for (int i = 0; i < (HODLR_Root->leafMatrix).rows(); i++){
      for (int j = 0; j < (HODLR_Root->leafMatrix).cols(); j++){
	int rowIdx = extendIdxVec[i + HODLR_Root->min_i] - min_i;
	int colIdx = extendIdxVec[j + HODLR_Root->min_j] - min_j;
	leafMatrix(rowIdx,colIdx) = (HODLR_Root->leafMatrix)(i,j);
      } 
    }
    /*
    HODLR_Root->min_i         = min_i;
    HODLR_Root->min_j         = min_j;
    HODLR_Root->max_i         = max_i;
    HODLR_Root->max_j         = max_j;
    HODLR_Root->leafMatrix    = leafMatrix;
    */
    
    resultRoot->min_i         = min_i;
    resultRoot->min_j         = min_j;
    resultRoot->max_i         = max_i;
    resultRoot->max_j         = max_j;
    resultRoot->leafMatrix    = leafMatrix;
   
    return;
  }
  
  int splitIndex_i  = extendIdxVec[HODLR_Root->splitIndex_i];
  int splitIndex_j  = extendIdxVec[HODLR_Root->splitIndex_j];
  
  // Modify Matrices
  int numRows_TopOffDiag  = splitIndex_i - min_i + 1; 
  int numRows_BottOffDiag = max_i - splitIndex_i;
  int numCols_TopOffDiag  = numRows_BottOffDiag;
  int numCols_BottOffDiag = numRows_TopOffDiag; 
  int topRank             = (HODLR_Root->topOffDiagU).cols();
  int bottRank            = (HODLR_Root->bottOffDiagU).cols();
  Eigen::MatrixXd topOffDiagU  = Eigen::MatrixXd::Zero(numRows_TopOffDiag,topRank);
  Eigen::MatrixXd topOffDiagV  = Eigen::MatrixXd::Zero(numCols_TopOffDiag,topRank);
  Eigen::MatrixXd bottOffDiagU = Eigen::MatrixXd::Zero(numRows_BottOffDiag,bottRank);
  Eigen::MatrixXd bottOffDiagV = Eigen::MatrixXd::Zero(numCols_BottOffDiag,bottRank);
  
  for (int i = 0; i < (HODLR_Root->topOffDiagU).rows(); i++){
    int rowIdx = extendIdxVec[i + HODLR_Root->min_i] - min_i;
    topOffDiagU.row(rowIdx) = (HODLR_Root->topOffDiagU).row(i);
  } 
    
  for (int i = 0; i < (HODLR_Root->topOffDiagV).rows(); i++){
    int rowIdx = extendIdxVec[i + HODLR_Root->splitIndex_j + 1] - splitIndex_j - 1;
    topOffDiagV.row(rowIdx) = (HODLR_Root->topOffDiagV).row(i);
  } 

  for (int i = 0; i < (HODLR_Root->bottOffDiagU).rows(); i++){
    int rowIdx = extendIdxVec[i + HODLR_Root->splitIndex_i + 1] - splitIndex_i - 1;
    bottOffDiagU.row(rowIdx) = (HODLR_Root->bottOffDiagU).row(i);
  } 

  for (int i = 0; i < (HODLR_Root->bottOffDiagV).rows(); i++){
    int rowIdx = extendIdxVec[i + HODLR_Root->min_j] - min_j;
    bottOffDiagV.row(rowIdx) = (HODLR_Root->bottOffDiagV).row(i);
  } 
  /*
  HODLR_Root->min_i         = min_i;
  HODLR_Root->min_j         = min_j;
  HODLR_Root->max_i         = max_i;
  HODLR_Root->max_j         = max_j;
  HODLR_Root->splitIndex_i  = splitIndex_i;
  HODLR_Root->splitIndex_j  = splitIndex_j;
  HODLR_Root->topOffDiagU   = topOffDiagU;
  HODLR_Root->topOffDiagV   = topOffDiagV;
  HODLR_Root->bottOffDiagU  = bottOffDiagU;
  HODLR_Root->bottOffDiagV  = bottOffDiagV;
  extend( HODLR_Root->left ,extendIdxVec,parentSize);
  extend( HODLR_Root->right,extendIdxVec,parentSize);
  */
  
  resultRoot->min_i         = min_i;
  resultRoot->min_j         = min_j;
  resultRoot->max_i         = max_i;
  resultRoot->max_j         = max_j;
  resultRoot->splitIndex_i  = splitIndex_i;
  resultRoot->splitIndex_j  = splitIndex_j;
  resultRoot->topOffDiagU   = topOffDiagU;
  resultRoot->topOffDiagV   = topOffDiagV;
  resultRoot->bottOffDiagU  = bottOffDiagU;
  resultRoot->bottOffDiagV  = bottOffDiagV;
  extend( resultRoot->left,HODLR_Root->left ,extendIdxVec,parentSize);
  extend( resultRoot->right,HODLR_Root->right,extendIdxVec,parentSize);
 
  
}

template <typename T>
void extractParentChildRowsCols(HODLR_Matrix & parentHODLR,const int min_i,const int min_j,const int numRows,const int numCols,T & D,std::vector<int> & parentRowColIdxVec,std::vector<int> & updateIdxVec,const std::string mode,Eigen::MatrixXd & parentExtract,Eigen::MatrixXd & childExtract){
  // Stage 1: extract parent block
  if ((parentExtract.rows() > 0) && (parentRowColIdxVec.size() > 1)) {
    std::vector<int> parentRowColBlk  = extractBlocks(parentRowColIdxVec);
    for (int i = 0; i < ((int)parentRowColBlk.size() - 1); i++){
      int currBlkStartIdx = parentRowColBlk[i];
      int currBlkEndIdx   = parentRowColBlk[i + 1] - 1;
      int blkSize         = currBlkEndIdx - currBlkStartIdx + 1;
      if (mode == "Cols")
	parentExtract.block(0,currBlkStartIdx,numRows,blkSize) = parentHODLR.block(min_i,min_j + parentRowColIdxVec[currBlkStartIdx],numRows,blkSize);
      else if (mode == "Rows")
	parentExtract.block(0,currBlkStartIdx,numCols,blkSize) = parentHODLR.block(min_i + parentRowColIdxVec[currBlkStartIdx],min_j,blkSize,numCols).transpose();
      else{
	std::cout<<"Error! Unknown Operation mode."<<std::endl;
	exit(EXIT_FAILURE);
      }
    }
  }
  // Stage 2: extract child block
  if ((childExtract.rows() > 0) && (parentRowColIdxVec.size() > 1)){
    int offset;  
    if (mode == "Cols")
      offset = min_j;
    else if (mode == "Rows")
      offset = min_i;
    else{
      std::cout<<"Error! Unknown Operation mode."<<std::endl;
      exit(EXIT_FAILURE);
    }

    std::vector<int> childRowColIdxVec = createChildExractVec(parentRowColIdxVec,updateIdxVec,offset);
    std::vector<int> childRowColBlkVec = extractBlocks(childRowColIdxVec);
    
    // Extract child rows and columns in blocks
    for (int i = 0; i < ((int)childRowColBlkVec.size() - 1); i++){
      int currChildBlkStartIdx  = childRowColBlkVec[i];
      int currChildBlkEndIdx    = childRowColBlkVec[i + 1] - 1;
      int childBlkSize          = currChildBlkEndIdx - currChildBlkStartIdx + 1;
      int currParentStartIdx    = updateIdxVec[childRowColIdxVec[currChildBlkStartIdx]];
      int currParentEndIdx      = updateIdxVec[childRowColIdxVec[currChildBlkEndIdx]];
      std::vector<int>::iterator startIdxIter = std::lower_bound (parentRowColIdxVec.begin(),parentRowColIdxVec.end(),currParentStartIdx - offset);
      std::vector<int>::iterator endIdxIter   = std::lower_bound (parentRowColIdxVec.begin(),parentRowColIdxVec.end(),currParentEndIdx - offset);
      int localStartIdx  = startIdxIter - parentRowColIdxVec.begin();
      int localEndIdx    = endIdxIter - parentRowColIdxVec.begin();
      int parentBlkSize  = localEndIdx - localStartIdx + 1;
      std::vector<int> blkExtendVec(childBlkSize,0);
      
      for (int j = 0; j < childBlkSize; j++){
	std::vector<int>::iterator childExtractIdxIter = std::lower_bound (parentRowColIdxVec.begin(),parentRowColIdxVec.end(),updateIdxVec[childRowColIdxVec[currChildBlkStartIdx + j]] - offset);
	blkExtendVec[j] = childExtractIdxIter - parentRowColIdxVec.begin() - localStartIdx;
      }
      int parentSize = parentHODLR.get_MatrixSize();
      if (mode == "Cols"){
	Eigen::MatrixXd childExtractCols = D.block(0,childRowColIdxVec[currChildBlkStartIdx],D.rows(),childBlkSize);
	Eigen::MatrixXd childExtractCols_RowExtend = ::extend(updateIdxVec,parentSize,childExtractCols,0,0,childExtractCols.rows(),childExtractCols.cols(),"Rows").block(min_i,0,numRows,childBlkSize);
	Eigen::MatrixXd childExtractCols_ColExtend = ::extend(blkExtendVec,parentBlkSize,childExtractCols_RowExtend,0,0,childExtractCols_RowExtend.rows(),childExtractCols_RowExtend.cols(),"Cols");
	childExtract.block(0,localStartIdx,numRows,parentBlkSize) = childExtractCols_ColExtend;
      }else if (mode == "Rows"){
	Eigen::MatrixXd childExtractRows = D.block(childRowColIdxVec[currChildBlkStartIdx],0,childBlkSize,D.cols());
	Eigen::MatrixXd childExtractRows_ColExtend = ::extend(updateIdxVec,parentSize,childExtractRows,0,0,childExtractRows.rows(),childExtractRows.cols(),"Cols").block(0,min_j,childBlkSize,numCols);
	Eigen::MatrixXd childExtractRows_RowExtend = ::extend(blkExtendVec,parentBlkSize,childExtractRows_ColExtend,0,0,childExtractRows_ColExtend.rows(),childExtractRows_ColExtend.cols(),"Rows");
	childExtract.block(0,localStartIdx,numCols,parentBlkSize) = childExtractRows_RowExtend.transpose();	
      }else{
	std::cout<<"Error! Unknown Operation mode."<<std::endl;
	exit(EXIT_FAILURE);
      }      
    }
  }
}


void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,Eigen::MatrixXd & extendD,Eigen::MatrixXd & D,std::vector<int> & updateIdxVec,double tol){
  if (HODLR_Root->isLeaf == true){
    int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
    int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;  
    std::vector<int> leafIdxVec(numCols);
    for (int i = 0; i < numCols; i++)
      leafIdxVec[i] = i;
    Eigen::MatrixXd childExtract = Eigen::MatrixXd::Zero(numRows,numCols);
    Eigen::MatrixXd parentExtract;
    //extractParentChildRowsCols(*this,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,D,leafIdxVec,updateIdxVec,"Cols",parentExtract,childExtract);
    extractParentChildRowsCols(parentHODLR,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,D,leafIdxVec,updateIdxVec,"Cols",parentExtract,childExtract);
    HODLR_Root->leafMatrix += childExtract;
    return;
  }

  int numRows_TopOffDiag  = HODLR_Root->splitIndex_i - HODLR_Root->min_i + 1; 
  int numRows_BottOffDiag = HODLR_Root->max_i - HODLR_Root->splitIndex_i;
  int numCols_TopOffDiag  = HODLR_Root->max_j - HODLR_Root->splitIndex_j;
  int numCols_BottOffDiag = HODLR_Root->splitIndex_j - HODLR_Root->min_j + 1;
  
  Eigen::MatrixXd U1_TopOffDiag,U1_BottOffDiag;
  Eigen::MatrixXd V1_TopOffDiag,V1_BottOffDiag;  
  Eigen::MatrixXd U2_TopOffDiag,U2_BottOffDiag;
  Eigen::MatrixXd V2_TopOffDiag,V2_BottOffDiag;
 
  // Create randomVectors
  std::vector<int> topColIdxVec  = HODLR_Root->topOffDiagColIdx;
  std::vector<int> topRowIdxVec  = HODLR_Root->topOffDiagRowIdx;
  std::vector<int> bottColIdxVec = HODLR_Root->bottOffDiagColIdx;
  std::vector<int> bottRowIdxVec = HODLR_Root->bottOffDiagRowIdx;

  int numPointsTop  = std::max(topColIdxVec.size() ,topRowIdxVec.size());
  int numPointsBott = std::max(bottColIdxVec.size(),bottRowIdxVec.size());
  numPointsTop  = std::max(numPointsTop,1);
  numPointsBott = std::max(numPointsBott,1);
  U1_TopOffDiag   = Eigen::MatrixXd::Zero(numRows_TopOffDiag,numPointsTop);
  V1_TopOffDiag   = Eigen::MatrixXd::Zero(numCols_TopOffDiag,numPointsTop);
  U1_BottOffDiag  = Eigen::MatrixXd::Zero(numRows_BottOffDiag,numPointsBott);
  V1_BottOffDiag  = Eigen::MatrixXd::Zero(numCols_BottOffDiag,numPointsBott);
  U2_TopOffDiag   = Eigen::MatrixXd::Zero(numRows_TopOffDiag,numPointsTop);
  V2_TopOffDiag   = Eigen::MatrixXd::Zero(numCols_TopOffDiag,numPointsTop);
  U2_BottOffDiag  = Eigen::MatrixXd::Zero(numRows_BottOffDiag,numPointsBott);
  V2_BottOffDiag  = Eigen::MatrixXd::Zero(numCols_BottOffDiag,numPointsBott);
  
  int min_i = HODLR_Root->min_i;
  int min_j = HODLR_Root->splitIndex_j + 1;
  
  //extractParentChildRowsCols(*this,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topColIdxVec,updateIdxVec,"Cols",U1_TopOffDiag,U2_TopOffDiag);
  //extractParentChildRowsCols(*this,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topRowIdxVec,updateIdxVec,"Rows",V1_TopOffDiag,V2_TopOffDiag);
  extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topColIdxVec,updateIdxVec,"Cols",U1_TopOffDiag,U2_TopOffDiag);
  extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topRowIdxVec,updateIdxVec,"Rows",V1_TopOffDiag,V2_TopOffDiag);

  min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
  min_j = HODLR_Root->min_j;
  //extractParentChildRowsCols(*this,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottColIdxVec,updateIdxVec,"Cols",U1_BottOffDiag,U2_BottOffDiag);
  //extractParentChildRowsCols(*this,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottRowIdxVec,updateIdxVec,"Rows",V1_BottOffDiag,V2_BottOffDiag);
  
  extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottColIdxVec,updateIdxVec,"Cols",U1_BottOffDiag,U2_BottOffDiag);
  extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottRowIdxVec,updateIdxVec,"Rows",V1_BottOffDiag,V2_BottOffDiag);
 
  // Crete U,K and V
  Eigen::MatrixXd U_TopOffDiag,K_TopOffDiag,V_TopOffDiag;
  Eigen::MatrixXd U_BottOffDiag,K_BottOffDiag,V_BottOffDiag;
  Eigen::MatrixXd tempU,tempV,tempK;
  
  tempU = U1_TopOffDiag + U2_TopOffDiag;
  tempV = V1_TopOffDiag + V2_TopOffDiag;
  tempK = Eigen::MatrixXd::Zero(numPointsTop,numPointsTop);
  for (unsigned int i = 0; i < numPointsTop; i++)
    for (unsigned int j = 0; j < numPointsTop; j++)
      if (i < topRowIdxVec.size())
	tempK(i,j) = tempU(topRowIdxVec[i],j);

  Eigen::FullPivLU<Eigen::MatrixXd> lu(tempK);
  lu.setThreshold(tol);
  int rank = lu.rank();
  if (rank > 0){
    V_TopOffDiag = ((lu.permutationP() * tempV.transpose()).transpose()).leftCols(rank);
    Eigen::MatrixXd L_Soln = lu.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::UnitLower>().solve(V_TopOffDiag.transpose());
    V_TopOffDiag = lu.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::Upper>().solve(L_Soln).transpose();
    K_TopOffDiag = Eigen::MatrixXd::Identity(rank,rank);
    U_TopOffDiag = (tempU * lu.permutationQ()).leftCols(rank);
  }else{
    U_TopOffDiag = Eigen::MatrixXd::Zero(numRows_TopOffDiag,1);
    V_TopOffDiag = Eigen::MatrixXd::Zero(numCols_TopOffDiag,1);
    K_TopOffDiag = Eigen::MatrixXd::Zero(1,1);
  }

  tempU = U1_BottOffDiag + U2_BottOffDiag;
  tempV = V1_BottOffDiag + V2_BottOffDiag;
  tempK = Eigen::MatrixXd::Zero(numPointsBott,numPointsBott);
  for (unsigned int i = 0; i < numPointsBott; i++)
    for (unsigned int j = 0; j < numPointsBott; j++)
      if (i < bottRowIdxVec.size())
	tempK(i,j) = tempU(bottRowIdxVec[i],j);

  Eigen::FullPivLU<Eigen::MatrixXd> lu2(tempK);
  lu2.setThreshold(tol);
  rank = lu2.rank();
  
  if (rank > 0){
    V_BottOffDiag = ((lu2.permutationP() * tempV.transpose()).transpose()).leftCols(rank);
    Eigen::MatrixXd L_Soln = lu2.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::UnitLower>().solve(V_BottOffDiag.transpose());
    V_BottOffDiag = lu2.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::Upper>().solve(L_Soln).transpose();
    K_BottOffDiag = Eigen::MatrixXd::Identity(rank,rank);
    U_BottOffDiag = (tempU * lu2.permutationQ()).leftCols(rank);
  }else{
    U_BottOffDiag = Eigen::MatrixXd::Zero(numRows_BottOffDiag,1);
    V_BottOffDiag = Eigen::MatrixXd::Zero(numCols_BottOffDiag,1);
    K_BottOffDiag = Eigen::MatrixXd::Zero(1,1);
  }

  HODLR_Root->topOffDiagRank = U_TopOffDiag.cols();
  HODLR_Root->topOffDiagU    = U_TopOffDiag;
  HODLR_Root->topOffDiagK    = K_TopOffDiag;
  HODLR_Root->topOffDiagV    = V_TopOffDiag;
  
  HODLR_Root->bottOffDiagRank = U_BottOffDiag.cols();
  HODLR_Root->bottOffDiagU    = U_BottOffDiag;
  HODLR_Root->bottOffDiagK    = K_BottOffDiag;
  HODLR_Root->bottOffDiagV    = V_BottOffDiag;

  extendAddLRinTree(parentHODLR,HODLR_Root->left ,extendD,D,updateIdxVec,tol);
  extendAddLRinTree(parentHODLR,HODLR_Root->right,extendD,D,updateIdxVec,tol);
  
}




void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,HODLR_Matrix & extendD_HODLR,HODLR_Matrix & D_HODLR, std::vector<int> & updateIdxVec,double tol,std::string mode){
  if (HODLR_Root->isLeaf == true){

    int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
    int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;  
    if (mode == "PS_Boundary"){
      std::vector<int> leafIdxVec(numCols);
      for (int i = 0; i < numCols; i++)
	leafIdxVec[i] = i;
      Eigen::MatrixXd childExtract = Eigen::MatrixXd::Zero(numRows,numCols);
      Eigen::MatrixXd parentExtract;
      //extractParentChildRowsCols(*this,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,D,leafIdxVec,updateIdxVec,"Cols",parentExtract,childExtract);
      extractParentChildRowsCols(parentHODLR,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,D_HODLR,leafIdxVec,updateIdxVec,"Cols",parentExtract,childExtract);
      HODLR_Root->leafMatrix += childExtract;
      return;
    }


    HODLR_Root->leafMatrix += extendD_HODLR.block(HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols);
    return;
  }

  int numRows_TopOffDiag  = HODLR_Root->splitIndex_i - HODLR_Root->min_i + 1; 
  int numRows_BottOffDiag = HODLR_Root->max_i - HODLR_Root->splitIndex_i;
  int numCols_TopOffDiag  = HODLR_Root->max_j - HODLR_Root->splitIndex_j;
  int numCols_BottOffDiag = HODLR_Root->splitIndex_j - HODLR_Root->min_j + 1;
  
  Eigen::MatrixXd U1_TopOffDiag,U1_BottOffDiag;
  Eigen::MatrixXd V1_TopOffDiag,V1_BottOffDiag;  
  Eigen::MatrixXd U2_TopOffDiag,U2_BottOffDiag;
  Eigen::MatrixXd V2_TopOffDiag,V2_BottOffDiag;
  Eigen::MatrixXd U_TopOffDiag,K_TopOffDiag,V_TopOffDiag;
  Eigen::MatrixXd U_BottOffDiag,K_BottOffDiag,V_BottOffDiag;
  int topRank,bottRank;
  
  if (mode == "PS_Boundary"){
    std::vector<int> topColIdxVec  = HODLR_Root->topOffDiagColIdx;
    std::vector<int> topRowIdxVec  = HODLR_Root->topOffDiagRowIdx;
    std::vector<int> bottColIdxVec = HODLR_Root->bottOffDiagColIdx;
    std::vector<int> bottRowIdxVec = HODLR_Root->bottOffDiagRowIdx;
    std::vector<int> topColBlk,topRowBlk,bottColBlk,bottRowBlk;
    topColBlk  = extractBlocks(topColIdxVec );
    topRowBlk  = extractBlocks(topRowIdxVec );
    bottColBlk = extractBlocks(bottColIdxVec);
    bottRowBlk = extractBlocks(bottRowIdxVec);

    int numPointsTop  = std::max(topColIdxVec.size() ,topRowIdxVec.size());
    int numPointsBott = std::max(bottColIdxVec.size(),bottRowIdxVec.size());
    numPointsTop    = std::max(numPointsTop,1);
    numPointsBott   = std::max(numPointsBott,1);
    U1_TopOffDiag   = Eigen::MatrixXd::Zero(numRows_TopOffDiag,numPointsTop);
    V1_TopOffDiag   = Eigen::MatrixXd::Zero(numCols_TopOffDiag,numPointsTop);
    U1_BottOffDiag  = Eigen::MatrixXd::Zero(numRows_BottOffDiag,numPointsBott);
    V1_BottOffDiag  = Eigen::MatrixXd::Zero(numCols_BottOffDiag,numPointsBott);
    U2_TopOffDiag   = Eigen::MatrixXd::Zero(numRows_TopOffDiag,numPointsTop);
    V2_TopOffDiag   = Eigen::MatrixXd::Zero(numCols_TopOffDiag,numPointsTop);
    U2_BottOffDiag  = Eigen::MatrixXd::Zero(numRows_BottOffDiag,numPointsBott);
    V2_BottOffDiag  = Eigen::MatrixXd::Zero(numCols_BottOffDiag,numPointsBott);
    
      
    int min_i = HODLR_Root->min_i;
    int min_j = HODLR_Root->splitIndex_j + 1;
    
    //extractParentChildRowsCols(*this,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topColIdxVec,updateIdxVec,"Cols",U1_TopOffDiag,U2_TopOffDiag);
    //extractParentChildRowsCols(*this,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topRowIdxVec,updateIdxVec,"Rows",V1_TopOffDiag,V2_TopOffDiag);
    extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D_HODLR,topColIdxVec,updateIdxVec,"Cols",U1_TopOffDiag,U2_TopOffDiag);
    extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D_HODLR,topRowIdxVec,updateIdxVec,"Rows",V1_TopOffDiag,V2_TopOffDiag);
    
    min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
    min_j = HODLR_Root->min_j;
    //extractParentChildRowsCols(*this,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottColIdxVec,updateIdxVec,"Cols",U1_BottOffDiag,U2_BottOffDiag);
    //extractParentChildRowsCols(*this,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottRowIdxVec,updateIdxVec,"Rows",V1_BottOffDiag,V2_BottOffDiag);
    
    extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D_HODLR,bottColIdxVec,updateIdxVec,"Cols",U1_BottOffDiag,U2_BottOffDiag);
    extractParentChildRowsCols(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D_HODLR,bottRowIdxVec,updateIdxVec,"Rows",V1_BottOffDiag,V2_BottOffDiag);



    /*
    int min_i = HODLR_Root->min_i;
    int min_j = HODLR_Root->splitIndex_j + 1;
    
    if (topColBlk.size() > 1)
      for (unsigned int i = 0; i < (topColBlk.size() - 1); i++){
	int currBlkStartIdx = topColBlk[i];
	int currBlkEndIdx   = topColBlk[i + 1] - 1;
	int numBlkCols      = currBlkEndIdx - currBlkStartIdx + 1;
	U1_TopOffDiag.block(0,currBlkStartIdx,numRows_TopOffDiag,numBlkCols) = parentHODLR.block(min_i,min_j +topColIdxVec[currBlkStartIdx],numRows_TopOffDiag,numBlkCols);
	U2_TopOffDiag.block(0,currBlkStartIdx,numRows_TopOffDiag,numBlkCols) = extendD_HODLR.block(min_i,min_j +topColIdxVec[currBlkStartIdx],numRows_TopOffDiag,numBlkCols);
      }
    if (topRowBlk.size() > 1)
      for (unsigned int i = 0; i < (topRowBlk.size() - 1); i++){
        int currBlkStartIdx = topRowBlk[i];
        int currBlkEndIdx   = topRowBlk[i + 1] - 1;
        int numBlkRows      = currBlkEndIdx - currBlkStartIdx + 1;
	V1_TopOffDiag.block(0,currBlkStartIdx,numCols_TopOffDiag,numBlkRows) = parentHODLR.block(min_i + topRowIdxVec[currBlkStartIdx],min_j,numBlkRows,numCols_TopOffDiag).transpose();
	V2_TopOffDiag.block(0,currBlkStartIdx,numCols_TopOffDiag,numBlkRows) = extendD_HODLR.block(min_i + topRowIdxVec[currBlkStartIdx],min_j,numBlkRows,numCols_TopOffDiag).transpose();
      } 
    
    min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
    min_j = HODLR_Root->min_j;

    if (bottColBlk.size() > 1)
      for (unsigned int i = 0; i < (bottColBlk.size() - 1); i++){
	int currBlkStartIdx = bottColBlk[i];
	int currBlkEndIdx   = bottColBlk[i + 1] - 1;
	int numBlkCols      = currBlkEndIdx - currBlkStartIdx + 1;
	U1_BottOffDiag.block(0,currBlkStartIdx,numRows_BottOffDiag,numBlkCols) = parentHODLR.block(min_i,min_j +bottColIdxVec[currBlkStartIdx],numRows_BottOffDiag,numBlkCols);
	U2_BottOffDiag.block(0,currBlkStartIdx,numRows_BottOffDiag,numBlkCols) = extendD_HODLR.block(min_i,min_j +bottColIdxVec[currBlkStartIdx],numRows_BottOffDiag,numBlkCols);
      }
    if (bottRowBlk.size() > 1)
      for (unsigned int i = 0; i < (bottRowBlk.size() - 1); i++){
        int currBlkStartIdx = bottRowBlk[i];
        int currBlkEndIdx   = bottRowBlk[i + 1] - 1;
        int numBlkRows      = currBlkEndIdx - currBlkStartIdx + 1;
	V1_BottOffDiag.block(0,currBlkStartIdx,numCols_BottOffDiag,numBlkRows) = parentHODLR.block(min_i + bottRowIdxVec[currBlkStartIdx],min_j,numBlkRows,numCols_BottOffDiag).transpose();
        V2_BottOffDiag.block(0,currBlkStartIdx,numCols_BottOffDiag,numBlkRows) = extendD_HODLR.block(min_i + bottRowIdxVec[currBlkStartIdx],min_j,numBlkRows,numCols_BottOffDiag).transpose();
      }
    */
    /*
    for (unsigned int i = 0; i < numPointsTop; i++){
      int min_i = HODLR_Root->min_i;
      int min_j = HODLR_Root->splitIndex_j + 1;
      if (i < topColIdxVec.size()){
	U1_TopOffDiag.col(i) = block(min_i,min_j + topColIdxVec[i],numRows_TopOffDiag,1);
	U2_TopOffDiag.col(i) = extendD_HODLR.block(min_i,min_j + topColIdxVec[i],numRows_TopOffDiag,1);   
      }   
      if (i < topRowIdxVec.size()){
	V1_TopOffDiag.col(i) = block(min_i + topRowIdxVec[i],min_j,1,numCols_TopOffDiag).transpose();
	V2_TopOffDiag.col(i) = extendD_HODLR.block(min_i + topRowIdxVec[i],min_j,1,numCols_TopOffDiag).transpose();
      }
    }
    
    for (unsigned int i = 0; i < numPointsBott; i++){
      int min_i = HODLR_Root->splitIndex_i + 1;
      int min_j = HODLR_Root->min_j;
      if (i < bottColIdxVec.size()){
	U1_BottOffDiag.col(i) = block(min_i,min_j + bottColIdxVec[i],numRows_BottOffDiag,1); 
	U2_BottOffDiag.col(i) = extendD_HODLR.block(min_i,min_j + bottColIdxVec[i],numRows_BottOffDiag,1);
      }
      if (i < bottRowIdxVec.size()){
	V1_BottOffDiag.col(i) = block(min_i + bottRowIdxVec[i],min_j,1,numCols_BottOffDiag).transpose();
	V2_BottOffDiag.col(i) = extendD_HODLR.block(min_i + bottRowIdxVec[i],min_j,1,numCols_BottOffDiag).transpose();
      }
    }
    */
    // Crete U,K and V
    Eigen::MatrixXd tempU,tempV,tempK;
    
    tempU = U1_TopOffDiag + U2_TopOffDiag;
    tempV = V1_TopOffDiag + V2_TopOffDiag;
    tempK = Eigen::MatrixXd::Zero(numPointsTop,numPointsTop);
    for (unsigned int i = 0; i < numPointsTop; i++)
      for (unsigned int j = 0; j < numPointsTop; j++)
	if (i < topRowIdxVec.size())
	  tempK(i,j) = tempU(topRowIdxVec[i],j);

    Eigen::FullPivLU<Eigen::MatrixXd> lu(tempK);
    lu.setThreshold(tol);
    int rank = lu.rank();
    if (rank > 0){
      V_TopOffDiag = ((lu.permutationP() * tempV.transpose()).transpose()).leftCols(rank);
      Eigen::MatrixXd L_Soln = lu.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::UnitLower>().solve(V_TopOffDiag.transpose());
      V_TopOffDiag = lu.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::Upper>().solve(L_Soln).transpose();
      K_TopOffDiag = Eigen::MatrixXd::Identity(rank,rank);
      U_TopOffDiag = (tempU * lu.permutationQ()).leftCols(rank);
    }else{
      U_TopOffDiag = Eigen::MatrixXd::Zero(numRows_TopOffDiag,1);
      V_TopOffDiag = Eigen::MatrixXd::Zero(numCols_TopOffDiag,1);
      K_TopOffDiag = Eigen::MatrixXd::Zero(1,1);
    }
    

    tempU = U1_BottOffDiag + U2_BottOffDiag;
    tempV = V1_BottOffDiag + V2_BottOffDiag;
    tempK = Eigen::MatrixXd::Zero(numPointsBott,numPointsBott);
    for (unsigned int i = 0; i < numPointsBott; i++)
      for (unsigned int j = 0; j < numPointsBott; j++)
	if (i < bottRowIdxVec.size())
	  tempK(i,j) = tempU(bottRowIdxVec[i],j);
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu2(tempK);
    lu2.setThreshold(tol);
    rank = lu2.rank();
    
    if (rank > 0){
      V_BottOffDiag = ((lu2.permutationP() * tempV.transpose()).transpose()).leftCols(rank);
      Eigen::MatrixXd L_Soln = lu2.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::UnitLower>().solve(V_BottOffDiag.transpose());
      V_BottOffDiag = lu2.matrixLU().topLeftCorner(rank,rank).triangularView<Eigen::Upper>().solve(L_Soln).transpose();
      K_BottOffDiag = Eigen::MatrixXd::Identity(rank,rank);
      U_BottOffDiag = (tempU * lu2.permutationQ()).leftCols(rank);
    }else{
      U_BottOffDiag = Eigen::MatrixXd::Zero(numRows_BottOffDiag,1);
      V_BottOffDiag = Eigen::MatrixXd::Zero(numCols_BottOffDiag,1);
      K_BottOffDiag = Eigen::MatrixXd::Zero(1,1);
    }
    
  }else if (mode == "Compress_LU"){
    int min_i_Top = HODLR_Root->min_i;
    int min_j_Top = HODLR_Root->splitIndex_j + 1;
    Eigen::MatrixXd addedMatrix_Top = parentHODLR.block(min_i_Top,min_j_Top,numRows_TopOffDiag,numCols_TopOffDiag) + extendD_HODLR.block(min_i_Top,min_j_Top,numRows_TopOffDiag,numCols_TopOffDiag);
    int min_i_Bott = HODLR_Root->splitIndex_i + 1;
    int min_j_Bott = HODLR_Root->min_j;
    Eigen::MatrixXd addedMatrix_Bott = parentHODLR.block(min_i_Bott,min_j_Bott,numRows_BottOffDiag,numCols_BottOffDiag) + extendD_HODLR.block(min_i_Bott,min_j_Bott,numRows_BottOffDiag,numCols_BottOffDiag);
    ::fullPivACA_LowRankApprox(addedMatrix_Top ,U_TopOffDiag ,V_TopOffDiag ,0,0,addedMatrix_Top.rows(),addedMatrix_Top.cols(),tol,topRank); 
    ::fullPivACA_LowRankApprox(addedMatrix_Bott,U_BottOffDiag,V_BottOffDiag,0,0,addedMatrix_Bott.rows(),addedMatrix_Bott.cols(),tol,bottRank);
    K_TopOffDiag  = Eigen::MatrixXd::Identity(topRank,topRank);
    K_BottOffDiag = Eigen::MatrixXd::Identity(bottRank,bottRank);
    
  }else{
    std::cout<<"Error! Unkown operation mode."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  HODLR_Root->topOffDiagRank = U_TopOffDiag.cols();
  HODLR_Root->topOffDiagU    = U_TopOffDiag;
  HODLR_Root->topOffDiagK    = K_TopOffDiag;
  HODLR_Root->topOffDiagV    = V_TopOffDiag;
  
  HODLR_Root->bottOffDiagRank = U_BottOffDiag.cols();
  HODLR_Root->bottOffDiagU    = U_BottOffDiag;
  HODLR_Root->bottOffDiagK    = K_BottOffDiag;
  HODLR_Root->bottOffDiagV    = V_BottOffDiag;
 
  extendAddLRinTree(parentHODLR,HODLR_Root->left ,extendD_HODLR,D_HODLR,updateIdxVec,tol,mode);
  extendAddLRinTree(parentHODLR,HODLR_Root->right,extendD_HODLR,D_HODLR,updateIdxVec,tol,mode);

}


void extendAddLRinTree(HODLR_Tree::node* HODLR_Root,const Eigen::MatrixXd & updateExtendU,const Eigen::MatrixXd & updateExtendV,double tol,std::string mode){
  assert(updateExtendV.cols() == updateExtendU.cols());
  int sumChildRanks = updateExtendV.cols();
  if (HODLR_Root->isLeaf == true){
    int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
    int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;  
    HODLR_Root->leafMatrix += (updateExtendU).block(HODLR_Root->min_i,0,numRows,sumChildRanks) * (updateExtendV).block(HODLR_Root->min_j,0,numCols,sumChildRanks).transpose();       
    return;
  }
  int numRows_TopOffDiag  = HODLR_Root->splitIndex_i - HODLR_Root->min_i + 1; 
  int numRows_BottOffDiag = HODLR_Root->max_i - HODLR_Root->splitIndex_i;
  int numCols_TopOffDiag  = HODLR_Root->max_j - HODLR_Root->splitIndex_j;
  int numCols_BottOffDiag = HODLR_Root->splitIndex_j - HODLR_Root->min_j + 1; 
  Eigen::MatrixXd U2_TopOffDiag,U2_BottOffDiag;
  Eigen::MatrixXd V2_TopOffDiag,V2_BottOffDiag;
  // Create topDiag U2s
  U2_TopOffDiag  = (updateExtendU).block(HODLR_Root->min_i,0,numRows_TopOffDiag,sumChildRanks);
  U2_BottOffDiag = (updateExtendU).block(HODLR_Root->splitIndex_i + 1,0,numRows_BottOffDiag,sumChildRanks);
  // Create V2s
  V2_TopOffDiag  = (updateExtendV).block(HODLR_Root->splitIndex_j + 1,0,numCols_TopOffDiag,sumChildRanks);
  V2_BottOffDiag = (updateExtendV).block(HODLR_Root->min_j,0,numCols_BottOffDiag,sumChildRanks);
  // Update current LR
  Eigen::MatrixXd result_U,result_V,result_K;
  HODLR_Root->topOffDiagRank  = add_LR(HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagK,HODLR_Root->topOffDiagV,HODLR_Root->topOffDiagU * HODLR_Root->topOffDiagK,HODLR_Root->topOffDiagV,U2_TopOffDiag,V2_TopOffDiag,tol,mode);
  HODLR_Root->bottOffDiagRank = add_LR(HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagK,HODLR_Root->bottOffDiagV,HODLR_Root->bottOffDiagU * HODLR_Root->bottOffDiagK,HODLR_Root->bottOffDiagV,U2_BottOffDiag,V2_BottOffDiag,tol,mode);
  // Do the same for children
  extendAddLRinTree(HODLR_Root->left ,updateExtendU,updateExtendV,tol,mode);
  extendAddLRinTree(HODLR_Root->right,updateExtendU,updateExtendV,tol,mode);
  
}


int add_LR(Eigen::MatrixXd & result_U,Eigen::MatrixXd & result_K,Eigen::MatrixXd & result_V,const Eigen::MatrixXd & U1, const Eigen::MatrixXd & V1, const Eigen::MatrixXd & U2, const Eigen::MatrixXd & V2,double tol,std::string mode){
  assert(U1.rows() == U2.rows());
  assert(V1.rows() == V2.rows());
  Eigen::MatrixXd Utot(U1.rows(),U1.cols() + U2.cols());
  Eigen::MatrixXd Vtot(V1.rows(),V1.cols() + V2.cols());
  Utot.leftCols(U1.cols())  = U1;
  Utot.rightCols(U2.cols()) = U2;
  Vtot.leftCols(V1.cols())  = V1;
  Vtot.rightCols(V2.cols()) = V2;
  
  if (mode == "Compress_QR"){
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_U(Utot);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_V(Vtot);
    Eigen::MatrixXd thinQ_U,thinQ_V;
    thinQ_U.setIdentity(Utot.rows(),Utot.cols());
    thinQ_V.setIdentity(Vtot.rows(),Vtot.cols());
    qr_U.householderQ().applyThisOnTheLeft(thinQ_U);
    qr_V.householderQ().applyThisOnTheLeft(thinQ_V);
    int rank_U = qr_U.rank();
    int rank_V = qr_V.rank();
    rank_U = std::max(rank_U,1);
    rank_V = std::max(rank_V,1);
    
    Eigen::MatrixXd Q_U = thinQ_U.leftCols(rank_U);
    Eigen::MatrixXd Q_V = thinQ_V.leftCols(rank_V);
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix_U = qr_U.colsPermutation();
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix_V = qr_V.colsPermutation();
    Eigen::MatrixXd sigma = (Q_U.transpose() * Utot * Vtot.transpose() * Q_V);
    Eigen::MatrixXd sigma_W,sigma_V,sigma_K;
    assert(sigma.rows() * sigma.cols() > 0);
    ::SVD_LowRankApprox(sigma,tol,&sigma_W,&sigma_V,&sigma_K);
    result_U = Q_U * sigma_W;
    result_K = sigma_K;
    result_V = Q_V * sigma_V;
    return sigma_K.rows();
  }else if (mode == "Compress_LU"){
    Eigen::MatrixXd U_U,V_U;
    int rank_U;
    ::fullPivACA_LowRankApprox(Utot,U_U,V_U,0,0,Utot.rows(),Utot.cols(),tol,rank_U);
    Eigen::MatrixXd U_V,V_V;
    int rank_V;
    ::fullPivACA_LowRankApprox(Vtot,U_V,V_V,0,0,Vtot.rows(),Vtot.cols(),tol,rank_V);
    Eigen::MatrixXd sigma = V_U.transpose() * V_V;
    Eigen::MatrixXd sigma_W,sigma_V,sigma_K;
    ::SVD_LowRankApprox(sigma,tol,&sigma_W,&sigma_V,&sigma_K);
    result_U = U_U * sigma_W;
    result_K = sigma_K;
    result_V = U_V * sigma_V;
    return sigma_K.rows();
  }else if (mode == "Exact"){
    int totRank = U1.cols() + U2.cols();
    result_U = Utot;
    result_V = Vtot;
    result_K = Eigen::MatrixXd::Identity(totRank,totRank);
    return totRank;
  }else{
    std::cout<<"Error! Unknown operation mode"<<std::endl;
    exit(EXIT_FAILURE);
  }
}
