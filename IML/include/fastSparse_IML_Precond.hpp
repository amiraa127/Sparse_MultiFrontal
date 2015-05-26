#ifndef SMF_FASTSPARSE_IML_PRECOND
#define SMF_FASTSPARSE_IML_PRECOND

#include "Eigen_IML_Vector.hpp"
#include "sparseMF.hpp"

class fastSparse_IML_Precond: public smf::sparseMF{

  public:
  
  fastSparse_IML_Precond(Eigen::SparseMatrix<double> & inputMatrix)
    : smf::sparseMF(inputMatrix){}
  
  Eigen_IML_Vector solve(const Eigen_IML_Vector & other){
    return fast_Solve(other);
  }
};

#endif // SMF_FASTSPARSE_IML_PRECOND
