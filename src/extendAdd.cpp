#include "extendAdd.hpp"



void extend(HODLR_Tree::node* resultRoot,HODLR_Tree::node* HODLR_Root,std::vector<int> & extendIdxVec,int parentSize);
void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,Eigen::MatrixXd & extendU,Eigen::MatrixXd & extendV,std::vector<int> & updateIdxVec,double tol,std::string mode);
template<typename T>
void extendAddinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,T & extendD,T & D,std::vector<int> & updateIdxVec,double tol,std::string mode);



HODLR_Matrix extend(std::vector<int> & extendIdxVec, int parentSize, HODLR_Matrix & childHODLR){
  assert(childHODLR.get_MatrixSize() == (int)extendIdxVec.size()); 
  childHODLR.storeLRinTree();
  HODLR_Matrix result = childHODLR;
  extend(result.get_TreeRootNode(),childHODLR.get_TreeRootNode(),extendIdxVec,parentSize);
  result.recalculateSize();
  result.freeMatrixData();
  return result;
}


void extendAddUpdate(HODLR_Matrix & parentHODLR,HODLR_Matrix & D_HODLR,std::vector<int> & updateIdxVec,double tol,std::string mode){
  
  parentHODLR.storeLRinTree(); 
  if (mode == "Compress_LU"){
    int parentSize = parentHODLR.get_MatrixSize();
    HODLR_Matrix extendD_HODLR = extend(updateIdxVec,parentSize,D_HODLR);
    assert(extendD_HODLR.rows() == parentSize);
    extendAddinTree(parentHODLR,parentHODLR.get_TreeRootNode(),extendD_HODLR,D_HODLR,updateIdxVec,tol,mode);
  }else if (mode == "PS_Boundary"){
    HODLR_Matrix extendD_HODLR;
    extendAddinTree(parentHODLR,parentHODLR.get_TreeRootNode(),extendD_HODLR,D_HODLR,updateIdxVec,tol,mode);
  }else{
    std::cout<<"Error! Unknown operation mode!"<<std::endl;
    exit(EXIT_FAILURE);
  }
  parentHODLR.freeMatrixData();
}


void extendAddUpdate(HODLR_Matrix & parentHODLR,Eigen::MatrixXd & D,std::vector<int> & updateIdxVec,double tol,std::string mode){
  
  parentHODLR.storeLRinTree(); 
  if (mode == "Compress_LU"){
    int parentSize = parentHODLR.get_MatrixSize();
    Eigen::MatrixXd extendD = extend(updateIdxVec,parentSize,D,0,0,D.rows(),D.cols(),"RowsCols");
    assert(extendD.rows() == parentSize);
    extendAddinTree(parentHODLR,parentHODLR.get_TreeRootNode(),extendD,D,updateIdxVec,tol,mode);
  }else if (mode == "PS_Boundary"){
    Eigen::MatrixXd extendD;
    extendAddinTree(parentHODLR,parentHODLR.get_TreeRootNode(),extendD,D,updateIdxVec,tol,mode);
  }else{
    std::cout<<"Error! Unknown operation mode!"<<std::endl;
    exit(EXIT_FAILURE);
  }
  parentHODLR.freeMatrixData();
}

void extendAddUpdate(HODLR_Matrix & parentHODLR,Eigen::MatrixXd & U,Eigen::MatrixXd & V,std::vector<int>& updateIdxVec,double tol,std::string mode){
  parentHODLR.storeLRinTree();
  int parentSize = parentHODLR.get_MatrixSize();
  Eigen::MatrixXd extendU = extend(updateIdxVec,parentSize,U,0,0,U.rows(),U.cols(),"Rows");
  Eigen::MatrixXd extendV = extend(updateIdxVec,parentSize,V,0,0,V.rows(),V.cols(),"Rows");
  extendAddLRinTree(parentHODLR,parentHODLR.get_TreeRootNode(),extendU,extendV,updateIdxVec,tol,mode);
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
void extractFromBlock(T & parentMatrix,const int min_i,const int min_j,const int numRows,const int numCols,std::vector<int> & parentRowColIdxVec,const std::string mode,Eigen::MatrixXd & parentExtract){
  
  if ((parentRowColIdxVec.size() > 1)) {
    std::vector<int> parentRowColBlk  = extractBlocks(parentRowColIdxVec);
    for (int i = 0; i < ((int)parentRowColBlk.size() - 1); i++){
      int currBlkStartIdx = parentRowColBlk[i];
      int currBlkEndIdx   = parentRowColBlk[i + 1] - 1;
      int blkSize         = currBlkEndIdx - currBlkStartIdx + 1;
      if (mode == "Cols")
	parentExtract.block(0,currBlkStartIdx,numRows,blkSize) = parentMatrix.block(min_i,min_j + parentRowColIdxVec[currBlkStartIdx],numRows,blkSize);
      else if (mode == "Rows")
	parentExtract.block(0,currBlkStartIdx,numCols,blkSize) = parentMatrix.block(min_i + parentRowColIdxVec[currBlkStartIdx],min_j,blkSize,numCols).transpose();
      else{
	std::cout<<"Error! Unknown Operation mode."<<std::endl;
	exit(EXIT_FAILURE);
      }
    }
  }
  
}


template <typename T>
void extractFromChild(HODLR_Matrix & parentHODLR,const int min_i,const int min_j,const int numRows,const int numCols,T & D,std::vector<int> & parentRowColIdxVec,std::vector<int> & updateIdxVec,const std::string mode,Eigen::MatrixXd & childExtract){
  // Stage 1: extract parent block
  
  /*
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
    
    extractFromBlock(parentHODLR,min_i,min_j,numRows,numCols,parentRowColIdxVec,mode,parentExtract);
  }
  */
  // Stage 2: extract child block
  if ((parentRowColIdxVec.size() > 1)){
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


Eigen::MatrixXd extractFromLR(Eigen::MatrixXd & U,Eigen::MatrixXd & V,const int min_i,const int min_j, const int numRows,const int numCols,std::vector<int> & rowColIdxVec,const std::string mode,const int numPoints){
  
  if (mode == "Cols"){
    Eigen::MatrixXd extractV = Eigen::MatrixXd::Zero(V.cols(),numPoints);
    extractFromBlock(V,min_j,0,numCols,V.cols(),rowColIdxVec,"Rows",extractV);
    return U.block(min_i,0,numRows,U.cols()) * extractV;
  }else if (mode == "Rows"){
    Eigen::MatrixXd extractU = Eigen::MatrixXd::Zero(U.cols(),numPoints);
    extractFromBlock(U,min_i,0,numRows,U.cols(),rowColIdxVec,"Rows",extractU);
    return V.block(min_j,0,numCols,V.cols()) * extractU;
  }else{
    std::cout<<"Error! Unknown Operation mode."<<std::endl;
    exit(EXIT_FAILURE);
  }
} 



template<typename T>
void extendAddinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,T & extendD,T & D, std::vector<int> & updateIdxVec,double tol,std::string mode){

  if (HODLR_Root->isLeaf == true){
    int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
    int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;  
    if (mode == "PS_Boundary"){
      std::vector<int> leafIdxVec(numCols);
      for (int i = 0; i < numCols; i++)
	leafIdxVec[i] = i;
      Eigen::MatrixXd childExtract = Eigen::MatrixXd::Zero(numRows,numCols);
      //extractParentChildRowsCols(parentHODLR,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,D,leafIdxVec,updateIdxVec,"Cols",parentExtract,childExtract);
      extractFromChild(parentHODLR,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,D,leafIdxVec,updateIdxVec,"Cols",childExtract);
     
      HODLR_Root->leafMatrix += childExtract;
      return;
    }
    HODLR_Root->leafMatrix += extendD.block(HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols);
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
    
    
    extractFromBlock(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topColIdxVec,"Cols",U1_TopOffDiag);
    extractFromBlock(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topRowIdxVec,"Rows",V1_TopOffDiag);
    
    extractFromChild(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topColIdxVec,updateIdxVec,"Cols",U2_TopOffDiag);
    extractFromChild(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topRowIdxVec,updateIdxVec,"Rows",V2_TopOffDiag);
   
    
    min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
    min_j = HODLR_Root->min_j;
    
    
    extractFromBlock(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottColIdxVec,"Cols",U1_BottOffDiag);
    extractFromBlock(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottRowIdxVec,"Rows",V1_BottOffDiag);

    extractFromChild(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottColIdxVec,updateIdxVec,"Cols",U2_BottOffDiag);
    extractFromChild(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottRowIdxVec,updateIdxVec,"Rows",V2_BottOffDiag);

    // Crete U,K and V
    Eigen::MatrixXd tempU,tempV,tempK;
  
    tempU = U1_TopOffDiag + U2_TopOffDiag;
    tempV = V1_TopOffDiag + V2_TopOffDiag;

    //std::cout<<"1"<<std::endl;
    HODLR_Root->topOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagV,HODLR_Root->topOffDiagK,topRowIdxVec,tol,"fullPivLU");

    /*
    tempK = Eigen::MatrixXd::Zero(numPointsTop,numPointsTop);
   
    for (unsigned int i = 0; i < numPointsTop; i++)
      for (unsigned int j = 0; j < numPointsTop; j++)
	if (i < topRowIdxVec.size())
	  tempK(i,j) = tempU(topRowIdxVec[i],j);

    Eigen::FullPivLU<Eigen::MatrixXd> lu(tempK);
    lu.setThreshold(tol);
    rank = lu.rank();
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
    
    
    std::cout<<(U_TopOffDiag - HODLR_Root->topOffDiagU).norm()<<std::endl;
    std::cout<<(V_TopOffDiag - HODLR_Root->topOffDiagV).norm()<<std::endl;
    std::cout<<(K_TopOffDiag - HODLR_Root->topOffDiagK).norm()<<std::endl;
    std::cout<<topRank - U_TopOffDiag.cols()<<std::endl;
    */
    tempU = U1_BottOffDiag + U2_BottOffDiag;
    tempV = V1_BottOffDiag + V2_BottOffDiag;

    //std::cout<<2<<std::endl; 
    HODLR_Root->bottOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagV,HODLR_Root->bottOffDiagK,bottRowIdxVec,tol,"fullPivLU");

    /*
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
    */

    //std::cout<<3<<std::endl;
    //HODLR_Root->topOffDiagRank  = topRank;
    //HODLR_Root->bottOffDiagRank = bottRank;
    extendAddinTree(parentHODLR,HODLR_Root->left ,extendD,D,updateIdxVec,tol,mode);
    extendAddinTree(parentHODLR,HODLR_Root->right,extendD,D,updateIdxVec,tol,mode);
    
  }else if (mode == "Compress_LU"){
    int min_i_Top = HODLR_Root->min_i;
    int min_j_Top = HODLR_Root->splitIndex_j + 1;
    Eigen::MatrixXd addedMatrix_Top = parentHODLR.block(min_i_Top,min_j_Top,numRows_TopOffDiag,numCols_TopOffDiag) + extendD.block(min_i_Top,min_j_Top,numRows_TopOffDiag,numCols_TopOffDiag);
    int min_i_Bott = HODLR_Root->splitIndex_i + 1;
    int min_j_Bott = HODLR_Root->min_j;
    Eigen::MatrixXd addedMatrix_Bott = parentHODLR.block(min_i_Bott,min_j_Bott,numRows_BottOffDiag,numCols_BottOffDiag) + extendD.block(min_i_Bott,min_j_Bott,numRows_BottOffDiag,numCols_BottOffDiag);
    fullPivACA_LowRankApprox(addedMatrix_Top ,U_TopOffDiag ,V_TopOffDiag ,0,0,addedMatrix_Top.rows(),addedMatrix_Top.cols(),tol,topRank); 
    fullPivACA_LowRankApprox(addedMatrix_Bott,U_BottOffDiag,V_BottOffDiag,0,0,addedMatrix_Bott.rows(),addedMatrix_Bott.cols(),tol,bottRank);
    K_TopOffDiag  = Eigen::MatrixXd::Identity(topRank,topRank);
    K_BottOffDiag = Eigen::MatrixXd::Identity(bottRank,bottRank);
    
    
    HODLR_Root->topOffDiagRank = U_TopOffDiag.cols();
    HODLR_Root->topOffDiagU    = U_TopOffDiag;
    HODLR_Root->topOffDiagK    = K_TopOffDiag;
    HODLR_Root->topOffDiagV    = V_TopOffDiag;
    
    HODLR_Root->bottOffDiagRank = U_BottOffDiag.cols();
    HODLR_Root->bottOffDiagU    = U_BottOffDiag;
    HODLR_Root->bottOffDiagK    = K_BottOffDiag;
    HODLR_Root->bottOffDiagV    = V_BottOffDiag;
    
    extendAddinTree(parentHODLR,HODLR_Root->left ,extendD,D,updateIdxVec,tol,mode);
    extendAddinTree(parentHODLR,HODLR_Root->right,extendD,D,updateIdxVec,tol,mode);
    
  }else{
    std::cout<<"Error! Unkown operation mode."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
}


void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,Eigen::MatrixXd & extendU,Eigen::MatrixXd & extendV,std::vector<int> & updateIdxVec,double tol,std::string mode){
   assert(extendV.cols() == extendU.cols());
   
  if (HODLR_Root->isLeaf == true){
    int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
    int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;
    HODLR_Root->leafMatrix += (extendU).block(HODLR_Root->min_i,0,numRows,extendU.cols()) * (extendV).block(HODLR_Root->min_j,0,numCols,extendV.cols()).transpose();
    return;
  }
  
  int numRows_TopOffDiag  = HODLR_Root->splitIndex_i - HODLR_Root->min_i + 1; 
  int numRows_BottOffDiag = HODLR_Root->max_i - HODLR_Root->splitIndex_i;
  int numCols_TopOffDiag  = HODLR_Root->max_j - HODLR_Root->splitIndex_j;
  int numCols_BottOffDiag = HODLR_Root->splitIndex_j - HODLR_Root->min_j + 1;
   
  if ((mode == "Compress_LU") || (mode == "Compress_QR")){
  
    Eigen::MatrixXd U2_TopOffDiag,U2_BottOffDiag;
    Eigen::MatrixXd V2_TopOffDiag,V2_BottOffDiag;
    // Create topDiag U2s
    U2_TopOffDiag  = (extendU).block(HODLR_Root->min_i,0,numRows_TopOffDiag,extendU.cols());
    U2_BottOffDiag = (extendU).block(HODLR_Root->splitIndex_i + 1,0,numRows_BottOffDiag,extendU.cols());
    // Create V2s
    V2_TopOffDiag  = (extendV).block(HODLR_Root->splitIndex_j + 1,0,numCols_TopOffDiag,extendV.cols());
    V2_BottOffDiag = (extendV).block(HODLR_Root->min_j,0,numCols_BottOffDiag,extendV.cols());
    // Update current LR
    Eigen::MatrixXd result_U,result_V,result_K;
    HODLR_Root->topOffDiagRank  = add_LR(HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagK,HODLR_Root->topOffDiagV,HODLR_Root->topOffDiagU * HODLR_Root->topOffDiagK,HODLR_Root->topOffDiagV,U2_TopOffDiag,V2_TopOffDiag,tol,mode);
    HODLR_Root->bottOffDiagRank = add_LR(HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagK,HODLR_Root->bottOffDiagV,HODLR_Root->bottOffDiagU * HODLR_Root->bottOffDiagK,HODLR_Root->bottOffDiagV,U2_BottOffDiag,V2_BottOffDiag,tol,mode);
    // Do the same for children
    extendAddLRinTree(parentHODLR,HODLR_Root->left ,extendU,extendV,updateIdxVec,tol,mode);
    extendAddLRinTree(parentHODLR,HODLR_Root->right,extendU,extendV,updateIdxVec,tol,mode);
  
  }else if (mode == "PS_Boundary"){
    
    Eigen::MatrixXd U1_TopOffDiag,U1_BottOffDiag;
    Eigen::MatrixXd V1_TopOffDiag,V1_BottOffDiag;  
    Eigen::MatrixXd U2_TopOffDiag,U2_BottOffDiag;
    Eigen::MatrixXd V2_TopOffDiag,V2_BottOffDiag;
    Eigen::MatrixXd U_TopOffDiag,K_TopOffDiag,V_TopOffDiag;
    Eigen::MatrixXd U_BottOffDiag,K_BottOffDiag,V_BottOffDiag;
   
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

    
    int min_i = HODLR_Root->min_i;
    int min_j = HODLR_Root->splitIndex_j + 1;
    
    extractFromBlock(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topColIdxVec,"Cols",U1_TopOffDiag);
    extractFromBlock(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topRowIdxVec,"Rows",V1_TopOffDiag);
  
    /*
    Eigen::MatrixXd extractV = Eigen::MatrixXd::Zero(extendV.cols(),numPointsTop);
    extractFromBlock(extendV,min_j,0,numCols_TopOffDiag,extendV.cols(),topColIdxVec,"Rows",extractV);
    U2_TopOffDiag = extendU.block(min_i,0,numRows_TopOffDiag,extendU.cols()) * extractV;
    */
   
    U2_TopOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topColIdxVec,"Cols",numPointsTop);
 
    /*
    Eigen::MatrixXd extractU = Eigen::MatrixXd::Zero(extendU.cols(),numPointsTop);
    extractFromBlock(extendU,min_i,0,numRows_TopOffDiag,extendU.cols(),topRowIdxVec,"Rows",extractU);
    V2_TopOffDiag = extendV.block(min_j,0,numCols_TopOffDiag,extendV.cols()) * extractU;
    */

        
    V2_TopOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topRowIdxVec,"Rows",numPointsTop);
 
    min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
    min_j = HODLR_Root->min_j;
       
    extractFromBlock(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottColIdxVec,"Cols",U1_BottOffDiag);
    extractFromBlock(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottRowIdxVec,"Rows",V1_BottOffDiag);
    
    /*
    extractV = Eigen::MatrixXd::Zero(extendV.cols(),numPointsBott);
    extractFromBlock(extendV,min_j,0,numCols_BottOffDiag,extendV.cols(),bottColIdxVec,"Rows",extractV);
    U2_BottOffDiag = extendU.block(min_i,0,numRows_BottOffDiag,extendU.cols()) * extractV;
    */
        
    U2_BottOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottColIdxVec,"Cols",numPointsBott);
   
    /*
    extractU = Eigen::MatrixXd::Zero(extendU.cols(),numPointsBott);
    extractFromBlock(extendU,min_i,0,numRows_BottOffDiag,extendU.cols(),bottRowIdxVec,"Rows",extractU);
    V2_BottOffDiag = extendV.block(min_j,0,numCols_BottOffDiag,extendV.cols()) * extractU;
    */

            
    V2_BottOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottRowIdxVec,"Rows",numPointsBott);
   
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
      
    HODLR_Root->topOffDiagRank = U_TopOffDiag.cols();
    HODLR_Root->topOffDiagU    = U_TopOffDiag;
    HODLR_Root->topOffDiagK    = K_TopOffDiag;
    HODLR_Root->topOffDiagV    = V_TopOffDiag;
    
    HODLR_Root->bottOffDiagRank = U_BottOffDiag.cols();
    HODLR_Root->bottOffDiagU    = U_BottOffDiag;
    HODLR_Root->bottOffDiagK    = K_BottOffDiag;
    HODLR_Root->bottOffDiagV    = V_BottOffDiag;
    
    extendAddLRinTree(parentHODLR,HODLR_Root->left ,extendU,extendV,updateIdxVec,tol,mode);
    extendAddLRinTree(parentHODLR,HODLR_Root->right,extendU,extendV,updateIdxVec,tol,mode);
    
  }else{
    std::cout<<"Error! Unkown operation mode."<<std::endl;
    exit(EXIT_FAILURE);
  }
}

