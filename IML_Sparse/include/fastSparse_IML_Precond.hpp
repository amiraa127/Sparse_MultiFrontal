#ifndef FASTSPARSE_IML_PRECOND
#define FASTSPARSE_IML_PRECOND

#include "Eigen_IML_Vector.hpp"
#include "sparseMF.hpp"

class fastSparse_IML_Precond: public sparseMF{

  public:
  
  fastSparse_IML_Precond(Eigen::SparseMatrix<double> & inputMatrix):sparseMF(inputMatrix){}
  
  Eigen_IML_Vector solve(const Eigen_IML_Vector & other){
    return fast_Solve(other);
  }
};











#endif
