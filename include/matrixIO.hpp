#ifndef  MATRIXIO_HPP
#define  MATRIXIO_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense> 
#include <vector>

/* Function: readTxtIntoMatrix
 * ---------------------------
 * This function reads a text file and outputs a dense matrix (Eigen's MatrixXd) on return.
 * The text file format is as follows:
 * Fisrt line : nRows nCols
 * All other lines: i j value(i,j)
 * This function is very slow for large matrices. Use the binary format instead.
 * inputFileName : Path of the input text file.
 */
Eigen::MatrixXd readTxtIntoMatrix(const std::string inputFileName);



/* Function : readMtxIntoSparseMatrix
 *-------------------------------------
 * This function reads a sparse matrix market format (*.mmx) file and returns an Eigen sparse matrix object.
 * Currently it only supports matrix object type with coordinate format. Only real or double data types are acceptable at this time.
 * The symmetricity can only be general or symmetric.
 * inputFileName : The path of the input matrix market file.
 */
Eigen::SparseMatrix<double> readMtxIntoSparseMatrix(const std::string inputFileName);



void saveSparseMatrixIntoMtx(const Eigen::SparseMatrix<double> &inputMatrix,const std::string outputFileName);


/* Function: saveMatrixXdToBinary
 * ------------------------------                                                       
 * This function saves a dense matrix (Eigen's MatrixXd) as a binary file (SVD_F_DB) file format.                                                                               
 * inputMatrix : The dense matrix being saved.                                          
 * outputFileName : Path of the output file.                       
 */
void saveMatrixXdToBinary(const Eigen::MatrixXd& inputMatrix, const std::string outputFileName);



/* Function: readBinaryIntoMatrixXd                                              
 * -------------------------------                                                      
 * This function reads a dense matrix binary file (SVD_F_DB) and outputs on return, a dense matrix (Eigen's MatrixXd).                                                          
 * inputFileName : Path of the input file.                                              
 */
Eigen::MatrixXd readBinaryIntoMatrixXd(const std::string inputFileName);


void saveVectorAsText(const std::string outputFileName, const std::vector<double> & inputVector);

#endif

