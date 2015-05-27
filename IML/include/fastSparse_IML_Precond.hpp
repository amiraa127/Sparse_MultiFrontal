#ifndef SMF_FASTSPARSE_IML_PRECOND
#define SMF_FASTSPARSE_IML_PRECOND

#include "Eigen_IML_Vector.hpp"
#include "sparseMF.hpp"

namespace iml
{
  /** \addtogroup solver
  *  @{
  */

  /// Interface to \ref smf::sparseMF solver.
  /**
   * This class calls \ref smf::sparseMF::fast_Solve method
   * on \ref Eigen_IML_Vector data structures to solve a linear system.
   * 
   * Can be used as a preconditioner for other \ref iml iterative methods.
   **/
  class fastSparse_IML_Precond : public smf::sparseMF // TODO: make inheritance private
  {
  public:
    /// constructor
    fastSparse_IML_Precond(Eigen::SparseMatrix<double>& inputMatrix)
      : smf::sparseMF(inputMatrix) {}
    
    /// solve a linear system with rhs given by \p rhs and returns the solution
    Eigen_IML_Vector solve(const Eigen_IML_Vector& rhs)
    {
      return fast_Solve(rhs);
    }
  };
  
  /** @}*/

} // end namespace iml
  
#endif // SMF_FASTSPARSE_IML_PRECOND
