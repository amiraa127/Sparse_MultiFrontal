#ifndef DIAG_IML_PRECOND_Sparse
#define DIAG_IML_PRECOND_Sparse


#include "Eigen_IML_Vector.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <math.h>

class diag_IML_Precond{
  
public:
  // constructors
  Eigen::VectorXd diagElms;
  int n;
  diag_IML_Precond(Eigen::SparseMatrix<double> inputMat){diagElms = inputMat.diagonal(); n = diagElms.rows(); }
  
  Eigen_IML_Vector solve(const Eigen_IML_Vector & other){
    Eigen::VectorXd soln(n);
    for (int i = 0; i < n ; i++){
      if (fabs(diagElms(i)) > 1e-16)
	soln(i) = other(i)/diagElms(i);
      else
	soln(i) = 0;
    }
    return soln;
  }
};
#endif
