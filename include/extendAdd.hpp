#ifndef EXTEND_ADD_HPP
#define EXTEND_ADD_HPP

//Standard C++   
#include <iostream>
#include <string>

//External Dependencies   
#include <Eigen/Dense>

//Custom Dependencies
#include "HODLR_Matrix.hpp"



Eigen::MatrixXd extend(std::vector<int> & extendIdxVec,int parentSize,Eigen::MatrixXd & child,int min_i,int min_j,int numRows,int numCols,std::string mode);
HODLR_Matrix extend(std::vector<int> & extendIdxVec,int parentSize,HODLR_Matrix & childHODLR);
void extendAddUpdate(HODLR_Matrix & parentHODLR, Eigen::MatrixXd & D,std::vector<int> & updateIdxVec,double tol,std::string mode);
void extendAddUpdate(HODLR_Matrix & parentHODLR, HODLR_Matrix & D_HODLR,std::vector<int> & updateIdxVec,double tol,std::string mode);
void extendAddUpdate(HODLR_Matrix & parentHODLR, Eigen::MatrixXd & U,Eigen::MatrixXd & V,std::vector<int> & updateIdxVec,double tol,std::string mode);
void extendAddUpdate(HODLR_Matrix & parentHODLR, std::vector<Eigen::MatrixXd*> D_Array,std::vector<HODLR_Matrix*> D_HODLR_Array,std::vector<Eigen::MatrixXd*> U_Array,std::vector<Eigen::MatrixXd*> V_Array,std::vector<std::vector<int> > & updateIdxVec_Array_D,std::vector<std::vector<int> > & updateIdxVec_Array_D_HODLR,double tol,std::string mode,int maxRank = -1);


template <typename T>
void extractFromMatrixBlk(T & parentMatrix,const int min_i,const int min_j,const int numRows,const int numCols,std::vector<int> & parentRowColIdxVec,const std::string mode,Eigen::MatrixXd & parentExtract);
template <typename T>
void extractFromChild(const int parentSize,const int min_i,const int min_j,const int numRows,const int numCols,T & childMatrix,std::vector<int> & parentRowColIdxVec,std::vector<int> & updateIdxVec,const std::string mode,Eigen::MatrixXd & childExtract);
Eigen::MatrixXd extractFromLR(Eigen::MatrixXd & extendU,Eigen::MatrixXd & extendV,const int min_i,const int min_j, const int numRows,const int numCols,std::vector<int> & rowColIdxVec,const std::string mode,const int numPoints);
  
#endif
