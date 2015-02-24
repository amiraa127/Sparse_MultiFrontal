#include "matrixIO.hpp"

/* Function: readTxtIntoMatrix
 * ---------------------------
 * This function reads a text file and outputs a dense matrix (Eigen's MatrixXd) on return.
 * The text file format is as follows:
 * Fisrt line : nRows nCols
 * All other lines: i j value(i,j)
 * This function is very slow for large matrices. Use the binary format instead.
 * inputFileName : Path of the input text file.
 */
Eigen::MatrixXd readTxtIntoMatrix(const std::string inputFileName){
  std::ifstream inputFile;
  inputFile.open(inputFileName.c_str());
  if (!inputFile.fail()){
    std::string currLine;
    int numRows,numCols;
    double value;
    int currRow = 0;
    int currCol = 0;
    int error;
    getline(inputFile,currLine);
    error = sscanf(currLine.c_str(),"%u %u",&numRows,&numCols);
    //check format
    if (error != 2){
      std::cout<<"Error! Bad format."<<std::endl;
      exit(EXIT_FAILURE);
    }
    int numEntries = 0;
    int maxEntries = numRows * numCols;
    Eigen::MatrixXd result(numRows,numCols);
    while ((!inputFile.eof()) && (numEntries < maxEntries)){
      getline(inputFile,currLine);
      error = sscanf(currLine.c_str(),"%u %u %lf",&currRow,&currCol,&value);
      if ((currRow == (numRows - 2)) && (currCol == (numCols - 2)))
	  std::cout<<value<<std::endl;
      //check format
      if (error != 3){
	std::cout<<"Error! Bad format."<<std::endl;
	exit(EXIT_FAILURE);
      }
      result(currRow,currCol) = value;
      numEntries ++;
    }
    inputFile.close();
    return result;
  }else{
    std::cout<<"Error! File "<<inputFileName<<" could not be opened"<<std::endl;
    exit(EXIT_FAILURE); 
  }
}


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
      if (isSymmetric && currCol != currRow){
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


void saveVectorAsText(const std::string outputFileName, const std::vector<double> & inputVector){
  std::ofstream outputFile;
  outputFile.open(outputFileName.c_str());
  if (!outputFile.is_open()){
    std::cout<<"Error! Unable to open file for saving."<<std::endl;
    exit(EXIT_FAILURE);
  }
  for (unsigned int i = 0; i < inputVector.size(); i++)
      outputFile<<i<<" "<<inputVector[i]<<" "<<std::endl;
  outputFile.close();
}
