#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 9, 1> ColVector9d;
class SL3 {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		SL3() :_mat(Eigen::Matrix3d::Identity(3,3)){
		base.resize(8);
		base[0] << 0, 0, 1, 0, 0, 0, 0, 0, 0;
		base[1] << 0, 0, 0, 0, 0, 1, 0, 0, 0;
		base[2] << 0, 1, 0, 0, 0, 0, 0, 0, 0;
		base[3] << 0, 0, 0, 1, 0, 0, 0, 0, 0;
		base[4] << 1, 0, 0, 0, -1, 0, 0, 0, 0;
		base[5] << 0, 0, 0, 0, -1, 0, 0, 0, 1;
		base[6] << 0, 0, 0, 0, 0, 0, 1, 0, 0;
		base[7] << 0, 0, 0, 0, 0, 0, 0, 1, 0;
		for (int i = 0; i < 8; i++)
		{
			memcpy(fullBase.data() + i * 9, base[i].data(), sizeof(double) * 9);
		}
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
		Vector8d ret;
		Eigen::Matrix3d tmp;
		tmp = this->_mat.log();
		ColVector9d tmpb(tmp.data()); //as is the col-major order, works fine.
		ret = fullBase.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(tmpb);
		return ret;
	}
		
	Eigen::Matrix3d _mat;
	std::vector<Eigen::Matrix3d> base;
	Eigen::Matrix<double, 9, 8> fullBase;
	
};