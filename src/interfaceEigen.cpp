#include "interfaceEigen.hpp"

#include <fstream>
#include <algorithm>
#include <set>

#include "output.hpp"

namespace smf
{

  Eigen::SparseMatrix<double> rowScaling(Eigen::SparseMatrix<double> &originalMatrix)
  {
    Eigen::SparseMatrix<double> transMatrix = originalMatrix.transpose();
    Eigen::VectorXd diagElm(transMatrix.rows());
    for (int k=0; k < transMatrix.outerSize(); ++k){
      bool firstElm = true;
      double maxElm = 0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(transMatrix,k); it; ++it){
	if (firstElm){
	  maxElm = fabs(it.value());
	  firstElm = false;
	}else if (fabs(it.value()) > maxElm){
	  maxElm = fabs(it.value());
	} 
      }
      diagElm(k) = 1.0/maxElm;
    }
    return diagElm.asDiagonal() * originalMatrix;
  }  


  void  convertSparseMatrixIntoGraph(const Eigen::SparseMatrix<double> &inputMatrix,
				     SCOTCH_Graph* graphPtr,
				     const std::string fileName)
  {
    FILE* filePtr;
    // save sparse matrix as mtx file
    std::string matrixFileName;
    std::string graphFileName;
    if (fileName == "default"){
      matrixFileName = "data/tmp/tempMatrix.mtx";
      graphFileName = "data/tmp/tempGraph.grf";
    }else{
      matrixFileName = "data/"+fileName+".mtx";
      graphFileName = "data/"+fileName+".grf";
    }
    saveSparseMatrixIntoMtx(inputMatrix,matrixFileName);
    // Generate graph of sparse matrix
    std::string graphGenCommand = "gcv -im " + matrixFileName + " -os " + graphFileName;
    
    assert_msg(system(graphGenCommand.c_str()) == 0, 
	      "Could not generate graph from sparse matrix.");
    
    // Initialize graph
    assert_msg(SCOTCH_graphInit(graphPtr) == 0, 
	      "Could not initialize graph.");
    
    // Load graph from file
    assert_msg((filePtr = fopen(graphFileName.c_str(), "r")), 
	      "Could not open file.");
    assert_msg(SCOTCH_graphLoad(graphPtr, filePtr,0,0) == 0, 
	      "Could not load graph.");
    fclose(filePtr);
  }
  

  struct cmpArrValue{
    const std::vector<SCOTCH_Num> & valueVector;
    cmpArrValue(const std::vector<SCOTCH_Num> & valVec):
      valueVector(valVec){}
    bool operator()(int i, int j){
      return valueVector[i]<valueVector[j];
    }
  };

  
  std::vector<int> convertBinPartArrayIntoPermVector(SCOTCH_Num* parttab, int arrSize)
  {
    std::vector<SCOTCH_Num> partVec(parttab,parttab + arrSize);
    std::vector<int> permVecInv(arrSize);
    std::vector<int> permVec(arrSize);
    // Initialize permVectorInv
    for (int i = 0; i < arrSize ; i++)
      permVecInv[i] = i;
    // Create the inverse permutation vector
    std::sort(permVecInv.begin(),permVecInv.end(),cmpArrValue(partVec));
    
    // Convert the inverse permutation vector into permutation vector  
    for (int i=0; i < arrSize; i++)
      permVec[permVecInv[i]] = i;
    return permVec;
  }


  void recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix,
				std::vector<int> globalInvPerm, 
				int globalStartIndex, 
				int threshold, 
				user_IndexTree::node* root, 
				std::string LR_Method)
  {
    int matrixSize = inputMatrix.rows();


    // Convert sparse matrix into graph
    SCOTCH_Graph adjGraph;
    convertSparseMatrixIntoGraph(inputMatrix,&adjGraph);

    // Initialize partitioning strategy
    SCOTCH_Strat* partStratPtr = SCOTCH_stratAlloc() ;
    assert_msg(SCOTCH_stratInit(partStratPtr) == 0, 
	      "Could not initialize partitioning strategy.");

    // Partition graph
    SCOTCH_Num* parttab = (SCOTCH_Num*)calloc(matrixSize,sizeof(SCOTCH_Num));
    assert_msg(SCOTCH_graphPart(&adjGraph,2,partStratPtr,parttab) == 0, 
	      "Partitioning Failed.");

    // Order the global inverse permutation array
    std::vector<SCOTCH_Num> partVec(parttab,parttab + matrixSize);
    std::sort(globalInvPerm.begin() + globalStartIndex ,globalInvPerm.begin() + globalStartIndex + matrixSize, cmpArrValue(partVec));
  
    // Create index tree node
    std::sort(partVec.begin(),partVec.end());
    std::vector<SCOTCH_Num>::iterator iter = find(partVec.begin(),partVec.end(),1);
    int locSplitIndex = iter - partVec.begin();
    int globSplitIndex = globalStartIndex + locSplitIndex;
    root->splitIndex = globSplitIndex;
    root->topOffDiag_minRank = -1;
    root->bottOffDiag_minRank = -1;
    root->LR_Method = LR_Method;
    if (matrixSize <= threshold){
      root->left = NULL;
      root->right = NULL;
      return;
    }
    
    // Convert partition array into permuatation array
    std::vector<int> permVector = convertBinPartArrayIntoPermVector(parttab,matrixSize);
    
    // Permute the original matrix
    Eigen::SparseMatrix<double> permutedMatrix = permuteRowsCols(inputMatrix, permVector);

    // Extract the top and bottom diagonal matrices
    int nTop = locSplitIndex;
    int nBott = matrixSize - nTop;
    Eigen::SparseMatrix<double> topDiagonal = permutedMatrix.topLeftCorner(nTop,nTop);
    Eigen::SparseMatrix<double> bottomDiagonal = permutedMatrix.bottomRightCorner(nBott,nBott);
    
    // Partition the top and bottom diagonal matrices
    user_IndexTree::node* leftNode = new user_IndexTree::node;  
    user_IndexTree::node* rightNode = new user_IndexTree::node;
    root->left = leftNode;
    root->right = rightNode;
    recPartitionSparseMatrix(topDiagonal,globalInvPerm,globalStartIndex,threshold,leftNode,LR_Method);
    recPartitionSparseMatrix(bottomDiagonal,globalInvPerm,nTop + globalStartIndex,threshold,rightNode,LR_Method);

    SCOTCH_graphExit(&adjGraph);
    free(parttab);
    return;
  }
  

  std::vector<int> recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix, 
					    int threshold, 
					    user_IndexTree & usrTree,
					    std::string LR_Method)
  {
    assert(inputMatrix.rows() == inputMatrix.cols());
    int matrixSize = inputMatrix.rows();
    std::vector<int> globalInvPerm(matrixSize);
    std::vector<int> permVector(matrixSize);

    // Initialize gloablInvPerm
    for (int i = 0; i < matrixSize ; i++)
      globalInvPerm[i] = i;

    // Initialize user tree
    usrTree.rootNode = new user_IndexTree::node;

    // Recursively partition matrix
    recPartitionSparseMatrix(inputMatrix,globalInvPerm,0,threshold,usrTree.rootNode,LR_Method);
    
    // Convert gloablInvPerm to a permutation vector
    for (int i = 0; i < matrixSize ; i++)
      permVector[globalInvPerm[i]] = i;

    return permVector;
  }


  Eigen::MatrixXd createOneLevelSchurCmpl(const Eigen::SparseMatrix<double> &inputSpMatrix,
					  user_IndexTree &usrTree,
					  int treeSizeThresh, 
					  std::string LR_Method, 
					  std::string inputFileName)
  {
    SCOTCH_Graph adjGraph;
    convertSparseMatrixIntoGraph(inputSpMatrix,&adjGraph);
    
    // Find graph statistics                                                              
    SCOTCH_Num numVertices,numEdges;
    SCOTCH_graphSize(&adjGraph,&numVertices,&numEdges);

    // Order graph                                                                        

    // Initialize ordering strategy                                           
    SCOTCH_Strat* orderingStratPtr = SCOTCH_stratAlloc() ;
    assert_msg(SCOTCH_stratInit(orderingStratPtr) == 0, 
	      "Could not initialize ordering strategy.");
    
    SCOTCH_Num stratFlag = SCOTCH_STRATLEVELMIN | SCOTCH_STRATLEVELMAX ;
    assert_msg(SCOTCH_stratGraphOrderBuild(orderingStratPtr,stratFlag,1,0.01) == 0,
	      "Could not initialize ordering strategy string.");

    // Initialize variables                                                    
    SCOTCH_Num* permtab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
    SCOTCH_Num* peritab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
    SCOTCH_Num* treetab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
    SCOTCH_Num* rangtab = (SCOTCH_Num*)calloc((numVertices+1),sizeof(SCOTCH_Num));
    SCOTCH_Num* cblknbr = (SCOTCH_Num*)calloc(1,sizeof(SCOTCH_Num));
    
    // Reorder graph                                                                      
    assert_msg(SCOTCH_graphOrder(&adjGraph, orderingStratPtr, permtab, peritab, cblknbr, rangtab, treetab) == 0,
	      "Graph ordering failed.");

    // Find the range of columns corresponding to root supernode                          
    int rootSeparatorIndex = *cblknbr;
    int rootNodeStartIndex = *(rangtab + rootSeparatorIndex-1);
    int rootNodeEndIndex = *(rangtab + rootSeparatorIndex);
    int rootSize = rootNodeEndIndex - rootNodeStartIndex;

    // Permute rows and columns of the original sparse matrix                          
    std::vector<int> permVector(permtab, permtab + numVertices);
    Eigen::SparseMatrix<double> permutedMatrix = permuteRowsCols(inputSpMatrix, permVector);

    // Extract root separator matrix                                                      
    Eigen::SparseMatrix<double> rootSeparatorMatrix = permutedMatrix.bottomRightCorner(rootSize,rootSize);

    // Obtain SchurComplement                                                           
    Eigen::MatrixXd result;
    if (inputFileName == "default"){
      int leafSize = numVertices - rootSize;
      Eigen::SparseMatrix<double> leafMatrix = permutedMatrix.topLeftCorner(leafSize,leafSize);      
      Eigen::SparseMatrix<double> leafToRoot = permutedMatrix.bottomLeftCorner(rootSize,leafSize);
      Eigen::SparseMatrix<double> leafToRoot_T = leafToRoot.transpose();
      
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > LDLTSolver;
      LDLTSolver.compute(leafMatrix);
      assert_msg(LDLTSolver.info() == Eigen::Success,
		"Factorization failed.");
      Eigen::SparseMatrix<double> solverSoln = LDLTSolver.solve(leafToRoot_T);
      assert_msg(LDLTSolver.info() == Eigen::Success,
		"Solving failed.");
      Eigen::SparseMatrix<double> sparseSchur = rootSeparatorMatrix - leafToRoot * solverSoln;
      
      // Recursively Partition Schur Complement
      std::vector<int> sepPermRecVec = recPartitionSparseMatrix(rootSeparatorMatrix,treeSizeThresh,usrTree,LR_Method);
  
      // Permute Schur Complement      
      Eigen::SparseMatrix<double> permutedSchur = permuteRowsCols(sparseSchur, sepPermRecVec);
      Eigen::MatrixXd denseSchur(permutedSchur);
      result = denseSchur;
    }else{
      recPartitionSparseMatrix(rootSeparatorMatrix,treeSizeThresh,usrTree,LR_Method);
      result = readBinaryIntoMatrixXd(inputFileName);
      assert(rootSeparatorMatrix.rows() == result.rows());
    }

    // Free space
    free(permtab); 
    free(peritab); 
    free(treetab); 
    free(rangtab); 
    free(cblknbr); 
    SCOTCH_graphExit(&adjGraph);

    return result;
  }

} // end namespace smf
