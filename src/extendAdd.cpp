#include "extendAdd.hpp"

namespace smf
{

  // some forward declarations
  
  void extend(HODLR_Tree::node* resultRoot,HODLR_Tree::node* HODLR_Root,std::vector<int> & extendIdxVec,int parentSize);
  void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,Eigen::MatrixXd & extendU,Eigen::MatrixXd & extendV,std::vector<int> & updateIdxVec,double tol,std::string mode);
  template<typename T>
  void extendAddinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,T & extendD,T & D,std::vector<int> & updateIdxVec,double tol,std::string mode);
  void storeParentContribution(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,std::string mode);
  void calcPseudoInvInTree(HODLR_Tree::node* HODLR_Root,double tol,int maxRank = - 1);
  template<typename T>
  void extendAddinTree(int parentSize,HODLR_Tree::node* HODLR_Root,std::vector<T*> D_Array,std::vector<std::vector<int> > & updateIdxVec_Array,double tol,std::string mode);
  void extendAddLRinTree(HODLR_Matrix & parentHODLR,HODLR_Tree::node* HODLR_Root,std::vector<Eigen::MatrixXd*> extendU_Array,std::vector<Eigen::MatrixXd*> extendV_Array,double tol,std::string mode);


  
  HODLR_Matrix extend(std::vector<int>& extendIdxVec, int parentSize, HODLR_Matrix& childHODLR)
  {
    assert(childHODLR.get_MatrixSize() == (int)extendIdxVec.size()); 
    childHODLR.storeLRinTree();
    HODLR_Matrix result = childHODLR;
    extend(result.get_indexTree().rootNode, childHODLR.get_indexTree().rootNode, extendIdxVec,parentSize);
    result.recalculateSize();
    result.freeMatrixData();
    return result;
  }
  

  void extendAddUpdate(HODLR_Matrix& parentHODLR, 
		      std::vector<Eigen::MatrixXd*> D_Array,
		      std::vector<HODLR_Matrix*> D_HODLR_Array,
		      std::vector<Eigen::MatrixXd*> U_Array,
		      std::vector<Eigen::MatrixXd*> V_Array,
		      std::vector<std::vector<int> >& updateIdxVec_Array_D,
		      std::vector<std::vector<int> >& updateIdxVec_Array_D_HODLR,
		      double tol, std::string mode, int maxRank)
  {
    parentHODLR.storeLRinTree();
    int parentSize = parentHODLR.get_MatrixSize();
    if (mode == "PS_Boundary"){
      storeParentContribution(parentHODLR,parentHODLR.get_indexTree().rootNode,mode);    
      parentHODLR.freeMatrixData();
      if (D_Array.size() > 0)
	extendAddinTree(parentSize,parentHODLR.get_indexTree().rootNode,D_Array,updateIdxVec_Array_D,tol,mode);
      if (D_HODLR_Array.size() > 0){
	extendAddinTree(parentSize,parentHODLR.get_indexTree().rootNode,D_HODLR_Array,updateIdxVec_Array_D_HODLR,tol,mode);
	// Extend Us and Vs
	std::vector<Eigen::MatrixXd *> extendU_Array(U_Array.size()),extendV_Array(V_Array.size());
	for (int i = 0; i < (int)U_Array.size();i++){
	  extendU_Array[i] = new Eigen::MatrixXd;
	  extendV_Array[i] = new Eigen::MatrixXd;
	  *extendU_Array[i] = extend(updateIdxVec_Array_D_HODLR[i],parentSize,*U_Array[i],0,0,U_Array[i]->rows(),U_Array[i]->cols(),"Rows");
	  *extendV_Array[i] = extend(updateIdxVec_Array_D_HODLR[i],parentSize,*V_Array[i],0,0,V_Array[i]->rows(),V_Array[i]->cols(),"Rows");
	}
	extendAddLRinTree(parentHODLR,parentHODLR.get_indexTree().rootNode,extendU_Array,extendV_Array,tol,mode); 
	for (int i = 0; i < (int)U_Array.size();i++){
	  delete extendU_Array[i];
	  delete extendV_Array[i];
	}
      }
      calcPseudoInvInTree(parentHODLR.get_indexTree().rootNode,tol,maxRank);
    }else{    
      std::cout<<"Error! Unknown operation mode!"<<std::endl;
      exit(EXIT_FAILURE);
    }
  }


  void extendAddUpdate(HODLR_Matrix& parentHODLR,
		      HODLR_Matrix& D_HODLR,
		      std::vector<int>& updateIdxVec,
		      double tol, std::string mode)
  {
    parentHODLR.storeLRinTree(); 
    if (mode == "Compress_LU"){ 
      int parentSize = parentHODLR.get_MatrixSize();
      HODLR_Matrix extendD_HODLR = extend(updateIdxVec,parentSize,D_HODLR);
      assert(extendD_HODLR.rows() == parentSize);
      extendAddinTree(parentHODLR,parentHODLR.get_indexTree().rootNode,extendD_HODLR,D_HODLR,updateIdxVec,tol,mode);
    }else if (mode == "PS_Boundary"){
      HODLR_Matrix extendD_HODLR;
      extendAddinTree(parentHODLR,parentHODLR.get_indexTree().rootNode,extendD_HODLR,D_HODLR,updateIdxVec,tol,mode);  
    }else{
      std::cout<<"Error! Unknown operation mode!"<<std::endl;
      exit(EXIT_FAILURE);
    }
    parentHODLR.freeMatrixData();
  }


  void extendAddUpdate(HODLR_Matrix& parentHODLR,
		      Eigen::MatrixXd& D,
		      std::vector<int>& updateIdxVec,
		      double tol, std::string mode)
  {
    parentHODLR.storeLRinTree(); 
    if (mode == "Compress_LU"){
      int parentSize = parentHODLR.get_MatrixSize();
      Eigen::MatrixXd extendD = extend(updateIdxVec,parentSize,D,0,0,D.rows(),D.cols(),"RowsCols");
      assert(extendD.rows() == parentSize);
      extendAddinTree(parentHODLR,parentHODLR.get_indexTree().rootNode,extendD,D,updateIdxVec,tol,mode);
      
    }else if (mode == "PS_Boundary"){
      Eigen::MatrixXd extendD;
      extendAddinTree(parentHODLR,parentHODLR.get_indexTree().rootNode,extendD,D,updateIdxVec,tol,mode);
    }else{
      std::cout<<"Error! Unknown operation mode!"<<std::endl;
      exit(EXIT_FAILURE);
    }
    parentHODLR.freeMatrixData();
  }
  

  void extendAddUpdate(HODLR_Matrix& parentHODLR,
		      Eigen::MatrixXd& U,
		      Eigen::MatrixXd& V,
		      std::vector<int>& updateIdxVec,
		      double tol,std::string mode)
  {
    parentHODLR.storeLRinTree();
    int parentSize = parentHODLR.get_MatrixSize();
    Eigen::MatrixXd extendU = extend(updateIdxVec,parentSize,U,0,0,U.rows(),U.cols(),"Rows");
    Eigen::MatrixXd extendV = extend(updateIdxVec,parentSize,V,0,0,V.rows(),V.cols(),"Rows");
    extendAddLRinTree(parentHODLR,parentHODLR.get_indexTree().rootNode,extendU,extendV,updateIdxVec,tol,mode);
    
    parentHODLR.freeMatrixData();
  }
  

  std::vector<int> extractBlocks(const std::vector<int>& inputVec)
  {
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
    if (blkVec[blkVec.size() - 1] != (int)inputVec.size())
      blkVec.push_back(inputVec.size());
    return blkVec;
  }


  std::vector<int> createChildExractVec(std::vector<int>& parentRowColIdxVec, 
					std::vector<int> updateIdxVec,
					int offset)
  {
    std::vector<int> result;
    for (int i = 0; i < (int)parentRowColIdxVec.size(); i++)
      if (std::binary_search (updateIdxVec.begin(), updateIdxVec.end(), parentRowColIdxVec[i] + offset)){
	std::vector<int>::iterator low = std::lower_bound (updateIdxVec.begin(), updateIdxVec.end(), parentRowColIdxVec[i] + offset);
	result.push_back(low - updateIdxVec.begin());
      }
    return result;
  }


  Eigen::MatrixXd extend(std::vector<int>& extendIdxVec,
			int parentSize,
			Eigen::MatrixXd& child,
			int min_i, int min_j,
			int numRows, int numCols,
			std::string mode)
  {
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
      for (int j = min_j; j <= max_j; j++){
	for (int i = min_i; i <= max_i; i++){
	  int rowIdx = extendIdxVec[i];
	  int colIdx = extendIdxVec[j];
	  result(rowIdx,colIdx) = child(i,j);
	}
      }
    }else if (mode == "Cols"){
      std::vector<int> extractBlk = extractBlocks(std::vector<int>(extendIdxVec.begin() + min_j,extendIdxVec.begin() + max_j + 1));
      result = Eigen::MatrixXd::Zero(child_NumRows,parentSize);
      for (int i = 0; i < (int)extractBlk.size() - 1; i++){
	int numBlkCols = extractBlk[i + 1] - extractBlk[i];
	result.block(0,extendIdxVec[extractBlk[i]],child_NumRows,numBlkCols) = child.block(0,extractBlk[i],child_NumRows,numBlkCols);  
      }
    }else if (mode == "Rows"){
      std::vector<int> extractBlk = extractBlocks(std::vector<int>(extendIdxVec.begin() + min_i,extendIdxVec.begin() + max_i + 1));
      result = Eigen::MatrixXd::Zero(parentSize,child_NumCols);
      for (int i = 0; i < (int)extractBlk.size() - 1; i++){
	int numBlkRows = extractBlk[i + 1] - extractBlk[i];
	result.block(extendIdxVec[extractBlk[i]],0,numBlkRows,child_NumCols) = child.block(extractBlk[i],0,numBlkRows,child_NumCols);
      }
    }else{
      std::cout<<"Error! Unknown operation mode."<<std::endl;
      exit(EXIT_FAILURE);
    }
    return result;
  }


  void extend(HODLR_Tree::node* resultRoot,
	      HODLR_Tree::node* HODLR_Root,
	      std::vector<int>& extendIdxVec,
	      int parentSize)
  {
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
  void extractFromMatrixBlk(T& parentMatrix,
			    int min_i, int min_j,
			    int numRows, int numCols,
			    std::vector<int>& parentRowColIdxVec,
			    std::string mode,
			    Eigen::MatrixXd& parentExtract)
  {
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
  void extractFromChild(int parentSize,
			int min_i, int min_j,
			int numRows, int numCols,
			T& D,
			std::vector<int>& parentRowColIdxVec,
			std::vector<int>& updateIdxVec,
			std::string mode,
			Eigen::MatrixXd& childExtract)
  {
    // Extract child block
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
	//int parentSize = parentHODLR.get_MatrixSize();
	if (mode == "Cols"){	
	  Eigen::MatrixXd childExtractCols = D.block(0,childRowColIdxVec[currChildBlkStartIdx],D.rows(),childBlkSize);
	  Eigen::MatrixXd childExtractCols_RowExtend = extend(updateIdxVec,parentSize,childExtractCols,0,0,childExtractCols.rows(),childExtractCols.cols(),"Rows").block(min_i,0,numRows,childBlkSize);
	  Eigen::MatrixXd childExtractCols_ColExtend = extend(blkExtendVec,parentBlkSize,childExtractCols_RowExtend,0,0,childExtractCols_RowExtend.rows(),childExtractCols_RowExtend.cols(),"Cols");
	  childExtract.block(0,localStartIdx,numRows,parentBlkSize) = childExtractCols_ColExtend;
	}else if (mode == "Rows"){
	  Eigen::MatrixXd childExtractRows = D.block(childRowColIdxVec[currChildBlkStartIdx],0,childBlkSize,D.cols());
	  Eigen::MatrixXd childExtractRows_ColExtend = extend(updateIdxVec,parentSize,childExtractRows,0,0,childExtractRows.rows(),childExtractRows.cols(),"Cols").block(0,min_j,childBlkSize,numCols);
	  Eigen::MatrixXd childExtractRows_RowExtend = extend(blkExtendVec,parentBlkSize,childExtractRows_ColExtend,0,0,childExtractRows_ColExtend.rows(),childExtractRows_ColExtend.cols(),"Rows");
	  childExtract.block(0,localStartIdx,numCols,parentBlkSize) = childExtractRows_RowExtend.transpose();	
	}else{
	  std::cout<<"Error! Unknown Operation mode."<<std::endl;
	  exit(EXIT_FAILURE);
	}      
      }
    }
  }


  Eigen::MatrixXd extractFromLR(Eigen::MatrixXd& U,
				Eigen::MatrixXd& V,
				int min_i, int min_j, 
				int numRows, int numCols,
				std::vector<int>& rowColIdxVec,
				std::string mode,
				int numPoints)
  {
    if (mode == "Cols"){
      Eigen::MatrixXd extractV = Eigen::MatrixXd::Zero(V.cols(),numPoints);
      extractFromMatrixBlk(V,min_j,0,numCols,V.cols(),rowColIdxVec,"Rows",extractV);
      return U.block(min_i,0,numRows,U.cols()) * extractV;
    }else if (mode == "Rows"){
      Eigen::MatrixXd extractU = Eigen::MatrixXd::Zero(U.cols(),numPoints);
      extractFromMatrixBlk(U,min_i,0,numRows,U.cols(),rowColIdxVec,"Rows",extractU);
      return V.block(min_j,0,numCols,V.cols()) * extractU;
    }else{
      std::cout<<"Error! Unknown Operation mode."<<std::endl;
      exit(EXIT_FAILURE);
    }
  } 

  
  void storeParentContribution(HODLR_Matrix& parentHODLR,
			      HODLR_Tree::node* HODLR_Root,
			      std::string mode)
  {
    if (mode == "PS_Boundary"){
      
      if (HODLR_Root->isLeaf == true){
	return;
      }

      int numRows_TopOffDiag  = HODLR_Root->splitIndex_i - HODLR_Root->min_i + 1; 
      int numRows_BottOffDiag = HODLR_Root->max_i - HODLR_Root->splitIndex_i;
      int numCols_TopOffDiag  = HODLR_Root->max_j - HODLR_Root->splitIndex_j;
      int numCols_BottOffDiag = HODLR_Root->splitIndex_j - HODLR_Root->min_j + 1;
      
      int numPointsTop  = std::max(HODLR_Root->topOffDiagColIdx.size() ,HODLR_Root->topOffDiagRowIdx.size());
      int numPointsBott = std::max(HODLR_Root->bottOffDiagColIdx.size(),HODLR_Root->bottOffDiagRowIdx.size());
      
      numPointsTop      = std::max(numPointsTop,1);
      numPointsBott     = std::max(numPointsBott,1);
      
      HODLR_Root->topOffDiagU     = Eigen::MatrixXd::Zero(numRows_TopOffDiag,numPointsTop);
      HODLR_Root->topOffDiagV     = Eigen::MatrixXd::Zero(numCols_TopOffDiag,numPointsTop);
      HODLR_Root->bottOffDiagU    = Eigen::MatrixXd::Zero(numRows_BottOffDiag,numPointsBott);
      HODLR_Root->bottOffDiagV    = Eigen::MatrixXd::Zero(numCols_BottOffDiag,numPointsBott);

      HODLR_Root->topOffDiagRank  = numPointsTop;
      HODLR_Root->bottOffDiagRank = numPointsBott;
      
      int min_i = HODLR_Root->min_i;
      int min_j = HODLR_Root->splitIndex_j + 1;
      
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,HODLR_Root->topOffDiagColIdx,"Cols",HODLR_Root->topOffDiagU);
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,HODLR_Root->topOffDiagRowIdx,"Rows",HODLR_Root->topOffDiagV);

      min_i = HODLR_Root->splitIndex_i + 1; 
      min_j = HODLR_Root->min_j;
		    
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,HODLR_Root->bottOffDiagColIdx,"Cols",HODLR_Root->bottOffDiagU);
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,HODLR_Root->bottOffDiagRowIdx,"Rows",HODLR_Root->bottOffDiagV);

      storeParentContribution(parentHODLR,HODLR_Root->left,mode);
      storeParentContribution(parentHODLR,HODLR_Root->right,mode);

    }else{
      std::cout<<"Error! Unkown operation mode."<<std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  
  void calcPseudoInvInTree(HODLR_Tree::node* HODLR_Root,
			  double tol, int maxRank)
  {
    if (HODLR_Root->isLeaf == true)
      return;

    Eigen::MatrixXd tempU,tempV;
    Eigen::MatrixXd U,K,V;
    tempU = HODLR_Root->topOffDiagU;
    tempV = HODLR_Root->topOffDiagV; 
    HODLR_Root->topOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagV,HODLR_Root->topOffDiagRowIdx,tol,"fullPivLU",maxRank);
  
    tempU = HODLR_Root->bottOffDiagU;
    tempV = HODLR_Root->bottOffDiagV;
    HODLR_Root->bottOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagV,HODLR_Root->bottOffDiagRowIdx,tol,"fullPivLU",maxRank);
    
    calcPseudoInvInTree(HODLR_Root->left,tol,maxRank);
    calcPseudoInvInTree(HODLR_Root->right,tol,maxRank);
  }


  template<typename T>
  void extendAddinTree(int parentSize,
		      HODLR_Tree::node* HODLR_Root,
		      std::vector<T*> D_Array,
		      std::vector<std::vector<int> >& updateIdxVec_Array,
		      double tol, std::string mode)
  {
    if (mode == "PS_Boundary"){
      if (HODLR_Root->isLeaf == true){
	int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
	int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;  
	std::vector<int> leafIdxVec(numCols);
	for (int i = 0; i < numCols; i++)
	  leafIdxVec[i] = i;
	for (int i = 0; i < (int)D_Array.size();i++){
	  Eigen::MatrixXd childExtract = Eigen::MatrixXd::Zero(numRows,numCols);
	  extractFromChild(parentSize,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,*(D_Array[i]),leafIdxVec,updateIdxVec_Array[i],"Cols",childExtract); 
	  HODLR_Root->leafMatrix += childExtract;
	}
	return;
      }
      
      int numRows_TopOffDiag  = HODLR_Root->splitIndex_i - HODLR_Root->min_i + 1; 
      int numRows_BottOffDiag = HODLR_Root->max_i - HODLR_Root->splitIndex_i;
      int numCols_TopOffDiag  = HODLR_Root->max_j - HODLR_Root->splitIndex_j;
      int numCols_BottOffDiag = HODLR_Root->splitIndex_j - HODLR_Root->min_j + 1;
      
      int numPointsTop  = std::max(HODLR_Root->topOffDiagColIdx.size() ,HODLR_Root->topOffDiagRowIdx.size());
      int numPointsBott = std::max(HODLR_Root->bottOffDiagColIdx.size(),HODLR_Root->bottOffDiagRowIdx.size());
      numPointsTop    = std::max(numPointsTop,1);
      numPointsBott   = std::max(numPointsBott,1);
      int min_i,min_j;
      
      for (int i = 0; i < (int)D_Array.size();i++){
	
	Eigen::MatrixXd U2_TopOffDiag   = Eigen::MatrixXd::Zero(numRows_TopOffDiag,numPointsTop);
	Eigen::MatrixXd V2_TopOffDiag   = Eigen::MatrixXd::Zero(numCols_TopOffDiag,numPointsTop);
	Eigen::MatrixXd U2_BottOffDiag  = Eigen::MatrixXd::Zero(numRows_BottOffDiag,numPointsBott);
	Eigen::MatrixXd V2_BottOffDiag  = Eigen::MatrixXd::Zero(numCols_BottOffDiag,numPointsBott);
	
	min_i = HODLR_Root->min_i;
	min_j = HODLR_Root->splitIndex_j + 1;
	
	extractFromChild(parentSize,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,*D_Array[i],HODLR_Root->topOffDiagColIdx,updateIdxVec_Array[i],"Cols",U2_TopOffDiag);
	extractFromChild(parentSize,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,*D_Array[i],HODLR_Root->topOffDiagRowIdx,updateIdxVec_Array[i],"Rows",V2_TopOffDiag);

	min_i = HODLR_Root->splitIndex_i + 1;  
	min_j = HODLR_Root->min_j;
      
	extractFromChild(parentSize,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,*D_Array[i],HODLR_Root->bottOffDiagColIdx,updateIdxVec_Array[i],"Cols",U2_BottOffDiag);
	extractFromChild(parentSize,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,*D_Array[i],HODLR_Root->bottOffDiagRowIdx,updateIdxVec_Array[i],"Rows",V2_BottOffDiag);
      
	HODLR_Root->topOffDiagU  += U2_TopOffDiag;
	HODLR_Root->topOffDiagV  += V2_TopOffDiag;
	HODLR_Root->bottOffDiagU += U2_BottOffDiag;
	HODLR_Root->bottOffDiagV += V2_BottOffDiag;
      }
      
      extendAddinTree(parentSize,HODLR_Root->left, D_Array,updateIdxVec_Array,tol,mode);
      extendAddinTree(parentSize,HODLR_Root->right,D_Array,updateIdxVec_Array,tol,mode);
    }
  }


  void extendAddLRinTree(HODLR_Matrix& parentHODLR,
			HODLR_Tree::node* HODLR_Root,
			std::vector<Eigen::MatrixXd*> extendU_Array,
			std::vector<Eigen::MatrixXd*> extendV_Array,
			double tol, std::string mode)
  {
    assert(extendV_Array.size() == extendU_Array.size());
    
    if (HODLR_Root->isLeaf == true){
      int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
      int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;
      for (int i = 0; i < (int)extendV_Array.size(); i++)
	HODLR_Root->leafMatrix += (extendU_Array[i])->block(HODLR_Root->min_i,0,numRows,extendU_Array[i]->cols()) * (extendV_Array[i])->block(HODLR_Root->min_j,0,numCols,extendV_Array[i]->cols()).transpose();
      return;
    }
    
    int numRows_TopOffDiag  = HODLR_Root->splitIndex_i - HODLR_Root->min_i + 1; 
    int numRows_BottOffDiag = HODLR_Root->max_i - HODLR_Root->splitIndex_i;
    int numCols_TopOffDiag  = HODLR_Root->max_j - HODLR_Root->splitIndex_j;
    int numCols_BottOffDiag = HODLR_Root->splitIndex_j - HODLR_Root->min_j + 1;
    if (mode == "PS_Boundary"){
    
      int numPointsTop  = std::max(HODLR_Root->topOffDiagColIdx.size() ,HODLR_Root->topOffDiagRowIdx.size());
      int numPointsBott = std::max(HODLR_Root->bottOffDiagColIdx.size(),HODLR_Root->bottOffDiagRowIdx.size());
      numPointsTop    = std::max(numPointsTop,1);
      numPointsBott   = std::max(numPointsBott,1);
      int min_i,min_j;
    
      for (int i = 0; i < (int)extendV_Array.size();i++){
	min_i = HODLR_Root->min_i;
	min_j = HODLR_Root->splitIndex_j + 1;
	HODLR_Root->topOffDiagU += extractFromLR(*extendU_Array[i],*extendV_Array[i],min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,HODLR_Root->topOffDiagColIdx,"Cols",numPointsTop);      
	HODLR_Root->topOffDiagV += extractFromLR(*extendU_Array[i],*extendV_Array[i],min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,HODLR_Root->topOffDiagRowIdx,"Rows",numPointsTop);
	
	min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
	min_j = HODLR_Root->min_j;
	HODLR_Root->bottOffDiagU += extractFromLR(*extendU_Array[i],*extendV_Array[i],min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,HODLR_Root->bottOffDiagColIdx,"Cols",numPointsBott);
	HODLR_Root->bottOffDiagV += extractFromLR(*extendU_Array[i],*extendV_Array[i],min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,HODLR_Root->bottOffDiagRowIdx,"Rows",numPointsBott);
	
      }
      extendAddLRinTree(parentHODLR,HODLR_Root->left ,extendU_Array,extendV_Array,tol,mode);
      extendAddLRinTree(parentHODLR,HODLR_Root->right,extendU_Array,extendV_Array,tol,mode);
    
    }else{
      std::cout<<"Error! Unkown operation mode."<<std::endl;
      exit(EXIT_FAILURE);
    }
  }


  template<typename T>
  void extendAddinTree(HODLR_Matrix& parentHODLR,
		      HODLR_Tree::node* HODLR_Root,
		      T& extendD,
		      T& D, 
		      std::vector<int>& updateIdxVec,
		      double tol, std::string mode)
  {
    int parentSize = parentHODLR.get_MatrixSize();
    if (HODLR_Root->isLeaf == true){
      int numRows = HODLR_Root->max_i - HODLR_Root->min_i + 1;
      int numCols = HODLR_Root->max_j - HODLR_Root->min_j + 1;  
      if (mode == "PS_Boundary"){
	std::vector<int> leafIdxVec(numCols);
	for (int i = 0; i < numCols; i++)
	  leafIdxVec[i] = i;
	Eigen::MatrixXd childExtract = Eigen::MatrixXd::Zero(numRows,numCols);
	extractFromChild(parentSize,HODLR_Root->min_i,HODLR_Root->min_j,numRows,numCols,D,leafIdxVec,updateIdxVec,"Cols",childExtract); 
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
      
      
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topColIdxVec,"Cols",U1_TopOffDiag);
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topRowIdxVec,"Rows",V1_TopOffDiag);
      
      extractFromChild(parentSize,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topColIdxVec,updateIdxVec,"Cols",U2_TopOffDiag);
      extractFromChild(parentSize,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,D,topRowIdxVec,updateIdxVec,"Rows",V2_TopOffDiag);
    
      
      min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
      min_j = HODLR_Root->min_j;
      
      
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottColIdxVec,"Cols",U1_BottOffDiag);
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottRowIdxVec,"Rows",V1_BottOffDiag);

      extractFromChild(parentSize,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottColIdxVec,updateIdxVec,"Cols",U2_BottOffDiag);
      extractFromChild(parentSize,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,D,bottRowIdxVec,updateIdxVec,"Rows",V2_BottOffDiag);

      // Crete U,K and V
      Eigen::MatrixXd tempU,tempV;
    
      tempU = U1_TopOffDiag + U2_TopOffDiag;
      tempV = V1_TopOffDiag + V2_TopOffDiag;   
      HODLR_Root->topOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagV,HODLR_Root->topOffDiagRowIdx,tol,"fullPivLU");
    
      tempU = U1_BottOffDiag + U2_BottOffDiag;
      tempV = V1_BottOffDiag + V2_BottOffDiag;
    
      HODLR_Root->bottOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagV,HODLR_Root->bottOffDiagRowIdx,tol,"fullPivLU");
    
      extendAddinTree(parentHODLR,HODLR_Root->left ,extendD,D,updateIdxVec,tol,mode);
      extendAddinTree(parentHODLR,HODLR_Root->right,extendD,D,updateIdxVec,tol,mode);
      
    }else if (mode == "Compress_LU"){
      int topRank,bottRank;
      int min_i_Top = HODLR_Root->min_i;
      int min_j_Top = HODLR_Root->splitIndex_j + 1;
      Eigen::MatrixXd addedMatrix_Top = parentHODLR.block(min_i_Top,min_j_Top,numRows_TopOffDiag,numCols_TopOffDiag) + extendD.block(min_i_Top,min_j_Top,numRows_TopOffDiag,numCols_TopOffDiag);
      int min_i_Bott = HODLR_Root->splitIndex_i + 1;
      int min_j_Bott = HODLR_Root->min_j;
      Eigen::MatrixXd addedMatrix_Bott = parentHODLR.block(min_i_Bott,min_j_Bott,numRows_BottOffDiag,numCols_BottOffDiag) + extendD.block(min_i_Bott,min_j_Bott,numRows_BottOffDiag,numCols_BottOffDiag);
      fullPivACA_LowRankApprox(addedMatrix_Top ,U_TopOffDiag ,V_TopOffDiag ,0,0,addedMatrix_Top.rows(),addedMatrix_Top.cols(),tol,topRank); 
      fullPivACA_LowRankApprox(addedMatrix_Bott,U_BottOffDiag,V_BottOffDiag,0,0,addedMatrix_Bott.rows(),addedMatrix_Bott.cols(),tol,bottRank);
      
      HODLR_Root->topOffDiagRank = U_TopOffDiag.cols();
      HODLR_Root->topOffDiagU    = U_TopOffDiag;
      HODLR_Root->topOffDiagV    = V_TopOffDiag;
      
      HODLR_Root->bottOffDiagRank = U_BottOffDiag.cols();
      HODLR_Root->bottOffDiagU    = U_BottOffDiag;
      HODLR_Root->bottOffDiagV    = V_BottOffDiag;
      
      extendAddinTree(parentHODLR,HODLR_Root->left ,extendD,D,updateIdxVec,tol,mode);
      extendAddinTree(parentHODLR,HODLR_Root->right,extendD,D,updateIdxVec,tol,mode);
      
    }else{
      std::cout<<"Error! Unkown operation mode."<<std::endl;
      exit(EXIT_FAILURE);
    }
  }


  void extendAddLRinTree(HODLR_Matrix& parentHODLR,
			HODLR_Tree::node* HODLR_Root,
			Eigen::MatrixXd& extendU,
			Eigen::MatrixXd& extendV,
			std::vector<int>& updateIdxVec,
			double tol, std::string mode)
  {
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
      
      HODLR_Root->topOffDiagRank  = add_LR(HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagV,HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagV,U2_TopOffDiag,V2_TopOffDiag,tol,mode);
      HODLR_Root->bottOffDiagRank = add_LR(HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagV,HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagV,U2_BottOffDiag,V2_BottOffDiag,tol,mode);
      
      // Do the same for children
      extendAddLRinTree(parentHODLR,HODLR_Root->left ,extendU,extendV,updateIdxVec,tol,mode);
      extendAddLRinTree(parentHODLR,HODLR_Root->right,extendU,extendV,updateIdxVec,tol,mode);
    
    } else if (mode == "PS_Boundary"){
      
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
      
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topColIdxVec,"Cols",U1_TopOffDiag);
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topRowIdxVec,"Rows",V1_TopOffDiag);
      
      U2_TopOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topColIdxVec,"Cols",numPointsTop);      
      V2_TopOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_TopOffDiag,numCols_TopOffDiag,topRowIdxVec,"Rows",numPointsTop);
  
      min_i = HODLR_Root->splitIndex_i + 1;                                                                                                                                                           
      min_j = HODLR_Root->min_j;
	
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottColIdxVec,"Cols",U1_BottOffDiag);
      extractFromMatrixBlk(parentHODLR,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottRowIdxVec,"Rows",V1_BottOffDiag);
	      
      U2_BottOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottColIdxVec,"Cols",numPointsBott);
      V2_BottOffDiag = extractFromLR(extendU,extendV,min_i,min_j,numRows_BottOffDiag,numCols_BottOffDiag,bottRowIdxVec,"Rows",numPointsBott);
    
      // Crete U,K and V
      Eigen::MatrixXd tempU,tempV;
      
      tempU = U1_TopOffDiag + U2_TopOffDiag;
      tempV = V1_TopOffDiag + V2_TopOffDiag;
      
      HODLR_Root->topOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->topOffDiagU,HODLR_Root->topOffDiagV,HODLR_Root->topOffDiagRowIdx,tol,"fullPivLU");
    
      tempU = U1_BottOffDiag + U2_BottOffDiag;
      tempV = V1_BottOffDiag + V2_BottOffDiag;
    
      HODLR_Root->bottOffDiagRank = PS_PseudoInverse(tempU,tempV,HODLR_Root->bottOffDiagU,HODLR_Root->bottOffDiagV,HODLR_Root->bottOffDiagRowIdx,tol,"fullPivLU");
    
      extendAddLRinTree(parentHODLR,HODLR_Root->left ,extendU,extendV,updateIdxVec,tol,mode);
      extendAddLRinTree(parentHODLR,HODLR_Root->right,extendU,extendV,updateIdxVec,tol,mode);
    
    }else{
      std::cout<<"Error! Unkown operation mode."<<std::endl;
      exit(EXIT_FAILURE);
    }
  }

} // end namespace smf
