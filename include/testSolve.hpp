#ifndef SMF_TEST_SOLVE_HPP
#define SMF_TEST_SOLVE_HPP

//Standard C++   
#include <iostream>

// Eigen library
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "sparseMF.hpp"
#include "output.hpp"

namespace smf
{

  void testSolveSp(Eigen::SparseMatrix<double> & inputSpMatrix,std::string mode)
  {
    msg("MatrixSize = ", inputSpMatrix.rows());
    Eigen::VectorXd exactSoln_Sp = Eigen::VectorXd::LinSpaced(Eigen::Sequential,inputSpMatrix.rows(),-2,2);
    Eigen::VectorXd RHS_Sp = inputSpMatrix * exactSoln_Sp;
    sparseMF solver(inputSpMatrix);
    solver.setPrintResultInfo(true);
    Eigen::MatrixXd soln_Sp;
    if (mode == "fast_Implicit")
      soln_Sp = solver.fast_Solve(RHS_Sp);                                                                                                                                                
    else if (mode == "fast_Iterative")
      soln_Sp = solver.iterative_Solve(RHS_Sp,100,1e-10,1e-1);
    else if (mode == "LU")
      soln_Sp = solver.LU_Solve(RHS_Sp);
    else if (mode == "implicit")
      soln_Sp = solver.implicit_Solve(RHS_Sp);
    else{
      error_exit("Unknown operation mode");
    }  
    double error = (exactSoln_Sp - soln_Sp).norm()/exactSoln_Sp.norm();
    msg("Relative Error = ", error);
  }

} // end namespace smf

#endif // SMF_TEST_SOLVE_HPP
