#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <iostream>

typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 9, 1> ColVector9d;
class SL3 {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		SL3() :_mat(Eigen::Matrix3d::Identity(3,3)){
	}
	SL3 operator * (const SL3& tr2) const
	{
		SL3 result(*this);
		result._mat *= tr2._mat;
		result.regularize();
		return result;
	}

	//Convert to SL3 from sl3
	void fromVector(const Vector8d& v){
		Eigen::Matrix3d tmp = Eigen::Matrix3d::Zero();
		/*
		for (int i = 0;i<base.size();i++)
		{
			ColVector9d debugB(base[i].data());
			std::cout<<debugB<<"\n";
		}
		 */
		for (int i = 0; i < 8; i++)
		{
			tmp += v[i] * base[i];
		}
		this->_mat = tmp.exp();
		regularize();
	}

	void inline regularize()
	{
		this->_mat = this->_mat / pow(this->_mat.determinant(), 1.0/3);
	}

	
	Vector8d toVector() const{
		return ret;
	}
		
	Eigen::Matrix3d _mat;
	
};