#include "helperFunctions.hpp"


/* Function : readMtxIntoSparseMatrix
 *-------------------------------------
 * This function reads a sparse matrix market format (*.mmx) file and returns an Eigen sparse matrix object.
 * Currently it only supports matrix object type with coordinate format. Only real or double data types are acceptable at this time.
 * The symmetricity can only be general or symmetric.
 * inputFileName : The path of the input matrix market file.
 */
Eigen::SparseMatrix<double> readMtxIntoSparseMatrix(const std::string inputFileName){
  //open the file
  std::ifstream inputFile;
  inputFile.open(inputFileName.c_str());
  if (!inputFile.fail()){
    std::string currLine;
    int numRows,numCols,nnz;
    double value;
    int currRow,currCol;
    int error;
    bool isSymmetric;
    //check header
    char str1[20],str2[20],str3[20],str4[20],str5[20];
    getline(inputFile,currLine);
    error = sscanf(currLine.c_str(),"%s %s %s %s %s",str1,str2,str3,str4,str5);
    if ((error != 5) || (strcmp(str1,"%%MatrixMarket") != 0)){
      std::cout<<"Error! Incorrect file header."<<std::endl;
      exit(EXIT_FAILURE);
    }
    if (strcmp(str2,"matrix") != 0){
      std::cout<<"Error! Only matrix object type is acceptable at this time."<<std::endl;
      exit(EXIT_FAILURE);
    }
    if (strcmp(str3,"coordinate") != 0){
      std::cout<<"Error! Only coordinate format is acceptable at this time."<<std::endl;
      exit(EXIT_FAILURE);
    }
    if ((strcmp(str4,"real") != 0) && (strcmp(str4,"double") != 0)){
      std::cout<<"Error! Only real or double data types are acceptable at this time."<<std::endl;
      exit(EXIT_FAILURE);
    }
    if ((strcmp(str5,"general") == 0))
      isSymmetric = false;
    else if ((strcmp(str5,"symmetric") == 0))
      isSymmetric = true;
    else{
      std::cout<<"Error! Only general or symmetric symmetry types are acceptable at this time."<<std::endl;
      exit(EXIT_FAILURE);
    } 
      
    //start filling the matrix
    while (inputFile.peek() == '%')
      inputFile.ignore(2048,'\n');
    getline(inputFile,currLine);
    error = sscanf(currLine.c_str(),"%u %u %u",&numRows,&numCols,&nnz);
    //check format correctness
    if (error != 3){
      std::cout<<"Error! Bad format."<<std::endl;
      exit(EXIT_FAILURE);
    }
    Eigen::SparseMatrix<double> result(numRows,numCols);
    std::vector<Eigen::Triplet<double,int> > tripletVector;
    int numEntries = 0;
    while ((!inputFile.eof()) && (numEntries < nnz)){
      getline(inputFile,currLine);
      error = sscanf(currLine.c_str(),"%u %u %lf",&currRow,&currCol,&value);
      //check format correctness
      if (error != 3){
	std::cout<<"Error! Bad format."<<std::endl;
	exit(EXIT_FAILURE);
      }
      Eigen::Triplet<double,int> currTriplet(currRow-1,currCol-1,value);
      tripletVector.push_back(currTriplet);
      // push back adjoint value into the matrix
      if (isSymmetric){
	Eigen::Triplet<double,int> adjTriplet(currCol-1,currRow-1,value);
	tripletVector.push_back(adjTriplet);
      }
      numEntries++;
    }
    inputFile.close();
    result.setFromTriplets(tripletVector.begin(),tripletVector.end());
    return result;
  }else{
    std::cout<<"Error! File "<<inputFileName<<" could not be opened."<<std::endl;
    exit(EXIT_FAILURE);
  }
}

void saveSparseMatrixIntoMtx(const Eigen::SparseMatrix<double> &inputMatrix,const std::string outputFileName){
  int numRows = inputMatrix.rows();
  int numCols = inputMatrix.cols();
  int numNnz = inputMatrix.nonZeros();
  std::ofstream outputFile;
  outputFile.open(outputFileName.c_str());
  if (!outputFile.is_open()){
    std::cout<<"Error! Unable to open file for saving."<<std::endl;
    exit(EXIT_FAILURE);
  }
  outputFile<<"%%MatrixMarket matrix coordinate real"<<std::endl;
  outputFile<<numRows<<" "<<numCols<<" "<<numNnz<<std::endl;
  for (int k = 0; k < inputMatrix.outerSize(); k++)
    for (Eigen::SparseMatrix<double>::InnerIterator it (inputMatrix,k); it; ++it)
      outputFile<<it.row()+1<<" "<<it.col()+1<<" "<<it.value()<<std::endl;
  outputFile.close();
}

/* Function: saveMatrixXdToBinary                                                       
 * ------------------------------                                                       
 * This function saves a dense matrix (Eigen's MatrixXd) as a binary file (SVD_F_DB) file format.                                                                               
 * inputMatrix : The dense matrix being saved.                                          
 * outputFileName : Path of the output file.                       
 */
void saveMatrixXdToBinary(const Eigen::MatrixXd& inputMatrix, const std::string outputFileName){
  std::ofstream outputFile;
  int nRows = inputMatrix.rows();
  int nCols = inputMatrix.cols();
  outputFile.open(outputFileName.c_str(),std::ios::binary);
  if (outputFile.is_open()){
    outputFile.write((char*)&nRows,sizeof(int));
    outputFile.write((char*)&nCols,sizeof(int));
    for (int i = 0; i < nRows ;i++)
      for (int j = 0; j< nCols ;j++){
        double currValue = inputMatrix(i,j);
        outputFile.write((char*)&currValue,sizeof(double));
      }
  }
  outputFile.close();
}

/* Function: readBinaryIntoMatrixXd                                              
 * -------------------------------                                                      
 * This function reads a dense matrix binary file (SVD_F_DB) and outputs on return, a dense matrix (Eigen's MatrixXd).                                                          
 * inputFileName : Path of the input file.                                              
 */
Eigen::MatrixXd readBinaryIntoMatrixXd(const std::string inputFileName){
  std::ifstream inputFile;
  inputFile.open(inputFileName.c_str());
  if (inputFile.is_open()){
    int nRows,nCols;
    inputFile.read((char*)&nRows,sizeof(int));
    inputFile.read((char*)&nCols,sizeof(int));
    Eigen::MatrixXd result(nRows,nCols);
    for (int i = 0; i < nRows ;i++)
      for (int j = 0; j< nCols ;j++){
        double currValue;
        inputFile.read((char*)&currValue,sizeof(double));
        result(i,j) = currValue;
      }
    inputFile.close();
    return result;
  }else{
    std::cout<<"Error! File "<<inputFileName<<" could not be opened"<<std::endl;
    exit(EXIT_FAILURE);
  }
}


Eigen::SparseMatrix<double> permuteRows(const Eigen::SparseMatrix<double> &originalMatrix, const std::vector<int> &permVector,bool transpose = false){
  int numVertices = originalMatrix.rows();
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix(numVertices);
  for (int i=0; i<numVertices; i++)
    permMatrix.indices()[i] = permVector[i];
  if (transpose == false)
    return (permMatrix * originalMatrix);
  else 
    return permMatrix.transpose() * originalMatrix;
}

/* Function : permuteRowsCols
 * --------------------------
 * This function permutes the rows and columns of a sparse matrix in a symmetric manner according to a given permutation vector.
 * It returns the permuted matrix as an Eigen sparse matrix object. 
 * originalMatrix : Eigen sparse matrix object to be permuted.
 * permVector : Permutation vector. If k = permVector[i], then row i of the original matrix is now row k of the reordered matrix.
 */
Eigen::SparseMatrix<double> permuteRowsCols(const Eigen::SparseMatrix<double> &originalMatrix, const std::vector<int> &permVector){
  int numVertices = originalMatrix.rows();
  assert(originalMatrix.rows() == originalMatrix.cols());
 
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix(numVertices);
  for (int i=0; i<numVertices; i++)
    permMatrix.indices()[i] = permVector[i];
 
  return (permMatrix * (permMatrix * originalMatrix).transpose()).transpose();
  

  /*Old version code
  Eigen::SparseMatrix<double> permMatrix(numVertices,numVertices);
  std::vector<Eigen::Triplet<double,int> > tripletVector;
  for (int i = 0; i < numVertices; i++){
    Eigen::Triplet<double,int> currTriplet(permVector[i],i,1);
    tripletVector.push_back(currTriplet);
  }
  permMatrix.setFromTriplets(tripletVector.begin(),tripletVector.end());
  Eigen::SparseMatrix<double> result(numVertices,numVertices);
  result = permMatrix * originalMatrix;
  result = result * (permMatrix.transpose());
  return result;
  */
}



Eigen::MatrixXd permuteRows(const Eigen::MatrixXd &originalMatrix, const std::vector<int> &permVector,bool transpose = false){
  int numVertices = originalMatrix.rows();
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix(numVertices);
  for (int i=0; i<numVertices; i++)
    permMatrix.indices()[i] = permVector[i];
  if (transpose == false)
    return (permMatrix * originalMatrix);
  else 
    return permMatrix.transpose() * originalMatrix;
}

Eigen::MatrixXd permuteRowsCols(const Eigen::MatrixXd &originalMatrix, const std::vector<int> &permVector){
  int numVertices = originalMatrix.rows();
  assert(originalMatrix.rows() == originalMatrix.cols());
 
  Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> permMatrix(numVertices);
  for (int i=0; i<numVertices; i++)
    permMatrix.indices()[i] = permVector[i];
  return (permMatrix * (permMatrix * originalMatrix).transpose()).transpose();
}


void  convertSparseMatrixIntoGraph(const Eigen::SparseMatrix<double> &inputMatrix,SCOTCH_Graph* graphPtr,const std::string fileName){
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
  if (system(graphGenCommand.c_str()) != 0){
    std::cout<<"Error! Could not generate graph from sparse matrix."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  // Initialize graph
  if (SCOTCH_graphInit(graphPtr) != 0){
    std::cout<<"Error! Could not initialize graph."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  // Load graph from file
  if ((filePtr = fopen(graphFileName.c_str(),"r")) == NULL){
    std::cout<<"Error! Could not open file."<<std::endl;
    exit(EXIT_FAILURE);
  }
  if (SCOTCH_graphLoad(graphPtr, filePtr,0,0) != 0){
    std::cout<<"Error! Could not load graph."<<std::endl;
    exit(EXIT_FAILURE);
  }
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

std::vector<int> convertBinPartArrayIntoPermVector(SCOTCH_Num* parttab,int arrSize){
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


void recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix,std::vector<int> globalInvPerm, const int globalStartIndex, const int threshold, user_IndexTree::node* root, const std::string LR_Method){

  int matrixSize = inputMatrix.rows();


  // Convert sparse matrix into graph
  SCOTCH_Graph adjGraph;
  convertSparseMatrixIntoGraph(inputMatrix,&adjGraph);

  // Initialize partitioning strategy
  SCOTCH_Strat* partStratPtr = SCOTCH_stratAlloc() ;
  if(SCOTCH_stratInit(partStratPtr) != 0){
    std::cout<<"Error! Could not initialize partitioning strategy."<<std::endl;
    exit(EXIT_FAILURE);
  }

  // Partition graph
  SCOTCH_Num* parttab = (SCOTCH_Num*)calloc(matrixSize,sizeof(SCOTCH_Num));
  if(SCOTCH_graphPart(&adjGraph,2,partStratPtr,parttab) !=0 ){
    std::cout<<"Error! Partitioning Failed."<<std::endl;
    exit(EXIT_FAILURE);
  }

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

std::vector<int> recPartitionSparseMatrix(const Eigen::SparseMatrix<double> &inputMatrix, const int threshold, user_IndexTree & usrTree,const std::string LR_Method){

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


Eigen::MatrixXd createOneLevelSchurCmpl(const Eigen::SparseMatrix<double> &inputSpMatrix,user_IndexTree &usrTree,const int treeSizeThresh, const std::string LR_Method, std::string inputFileName){

  SCOTCH_Graph adjGraph;
  convertSparseMatrixIntoGraph(inputSpMatrix,&adjGraph);
  
  // Find graph statistics                                                              
  SCOTCH_Num numVertices,numEdges;
  SCOTCH_graphSize(&adjGraph,&numVertices,&numEdges);

  // Order graph                                                                        

  // Initialize ordering strategy                                           
  SCOTCH_Strat* orderingStratPtr = SCOTCH_stratAlloc() ;
  if(SCOTCH_stratInit(orderingStratPtr) != 0){
    std::cout<<"Error! Could not initialize ordering strategy."<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  SCOTCH_Num stratFlag = SCOTCH_STRATLEVELMIN | SCOTCH_STRATLEVELMAX ;
  if(SCOTCH_stratGraphOrderBuild(orderingStratPtr,stratFlag,1,0.01) != 0){
    std::cout<<"Error! Could not initialize ordering strategy string."<<std::endl;
    exit(EXIT_FAILURE);
  }

  // Initialize variables                                                    
  SCOTCH_Num* permtab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
  SCOTCH_Num* peritab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
  SCOTCH_Num* treetab = (SCOTCH_Num*)calloc(numVertices,sizeof(SCOTCH_Num));
  SCOTCH_Num* rangtab = (SCOTCH_Num*)calloc((numVertices+1),sizeof(SCOTCH_Num));
  SCOTCH_Num* cblknbr = (SCOTCH_Num*)calloc(1,sizeof(SCOTCH_Num));
  
  // Reorder graph                                                                      
  if (SCOTCH_graphOrder(&adjGraph, orderingStratPtr, permtab, peritab, cblknbr, rangtab, treetab) != 0){
    std::cout<<"Error! Graph ordering failed."<<std::endl;
    exit(EXIT_FAILURE);
  }

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
    if(LDLTSolver.info() != Eigen::Success){
      std::cout<<"Error! Factorization failed."<<std::endl;
      exit(EXIT_FAILURE);
    }
    Eigen::SparseMatrix<double> solverSoln = LDLTSolver.solve(leafToRoot_T);
    if(LDLTSolver.info() != Eigen::Success){
      std::cout<<"Error! Solving failed."<<std::endl;
      exit(EXIT_FAILURE);
    }
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
};

