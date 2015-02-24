#ifndef EIGEN_IML_Matrix
#define EIGEN_IML_Matrix

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "Eigen_IML_Vector.hpp"

class Eigen_IML_Matrix : public  Eigen::SparseMatrix<double>
{
public:
  // constructors
  Eigen_IML_Matrix(void):Eigen::SparseMatrix<double>() {}
 
  /*
  typedef Eigen::SparseMatrix<double> Base;
  // This constructor allows you to construct MyVectorType from Eigen expressions
  template<typename OtherDerived>
  Eigen_IML_Matrix(const Eigen::MatrixBase<OtherDerived>& other): Eigen::SparseMatrix<double>(other) { }

  // This method allows you to assign Eigen expressions to MyMatrixType
  template<typename OtherDerived>
  Eigen_IML_Matrix & operator= (const Eigen::MatrixBase <OtherDerived>& other)
  {
    this->Base::operator=(other);
    return *this;
  }
  */
 
  Eigen_IML_Matrix & operator= (const Eigen::SparseMatrix<double>& other)
  {
    this->Base::operator=(other);
    return *this;
  }

  Eigen_IML_Vector trans_mult(Eigen_IML_Vector & other){
    
    return ((this->transpose())*other);

  }
};

#endif
