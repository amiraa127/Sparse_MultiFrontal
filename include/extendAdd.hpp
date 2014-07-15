#ifndef EXTEND_ADD_HPP
#define EXTEND_ADD_HPP

#include <Eigen/Dense>
#include "HODLR_Matrix.hpp"
#include <string>
#include <iostream>


// done
Eigen::MatrixXd extend(std::vector<int> & extendIdxVec,int parentSize,Eigen::MatrixXd & child,int min_i,int min_j,int numRows,int numCols,std::string mode);
// done
HODLR_Matrix extend(std::vector<int> & extendIdxVec,int parentSize,HODLR_Matrix & childHODLR);

// done but needs some work
void extendAddUpdate(HODLR_Matrix & parentHODLR, Eigen::MatrixXd & D,std::vector<int> & updateIdxVec,double tol,std::string mode);

// done but needs work
void extendAddUpdate(HODLR_Matrix & parentHODLR, HODLR_Matrix & D_HODLR,std::vector<int> & updateIdxVec,double tol,std::string mode);

// done 
void extendAddUpdate(HODLR_Matrix & parentHODLR, Eigen::MatrixXd & updateU,Eigen::MatrixXd & updateV,std::vector<int> & updateIdxVec,double tol,std::string mode);

int add_LR(Eigen::MatrixXd & result_U,Eigen::MatrixXd & result_K,Eigen::MatrixXd & result_V,const Eigen::MatrixXd & U1, const Eigen::MatrixXd & V1, const Eigen::MatrixXd & U2, const Eigen::MatrixXd & V2,double tol,std::string mode);




















#endif
