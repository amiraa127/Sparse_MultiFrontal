#ifndef EIGEN_IML_VECTOR
#define EIGEN_IML_VECTOR

#include <Eigen/Dense>


class Eigen_IML_Vector : public  Eigen::VectorXd
{
public:
  // constructors
  Eigen_IML_Vector(void):Eigen::VectorXd() {}
  Eigen_IML_Vector(unsigned int n):Eigen::VectorXd(n){}

  
  typedef Eigen::VectorXd Base;
  // This constructor allows you to construct MyVectorType from Eigen expressions
  template<typename OtherDerived>
  Eigen_IML_Vector(const Eigen::MatrixBase<OtherDerived>& other): Eigen::VectorXd(other) { }

  // This method allows you to assign Eigen expressions to MyVectorType
  template<typename OtherDerived>
  Eigen_IML_Vector & operator= (const Eigen::MatrixBase <OtherDerived>& other)
  {
    this->Base::operator=(other);
    return *this;
  }

  Eigen_IML_Vector & operator= (const double other)
  {
    this->Base::setConstant(other);
    return *this;
  }

};

inline double dot (Eigen_IML_Vector &left,Eigen_IML_Vector &right){
  return left.dot(right);
}

inline double norm (const Eigen_IML_Vector &myVec){
  return myVec.norm();
}
#endif
