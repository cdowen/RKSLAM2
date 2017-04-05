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
	}
	SL3 operator * (const SL3& tr2) const
	{
		SL3 result(*this);
		result._mat *= tr2._mat;
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
	}

	//TODO: test if Eigen map can point its memory to correct place
	Vector8d toVector() const{
		Vector8d ret;
		Eigen::Matrix3d tmp;
		tmp = this->_mat.log();
		std::vector<Eigen::Map<ColVector9d> > Avec;
		for (int i = 0; i < 8; i++)
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmpM(base[i]);
			Eigen::Map<ColVector9d> tmpCol(tmpM.data(), tmpM.size());
			Avec.push_back(tmpCol);
		}
		Eigen::Matrix<double, 9, 8> tmpA;
		tmpA << Avec[0], Avec[1], Avec[2], Avec[3], Avec[4], Avec[5], Avec[6], Avec[7];
		Eigen::Map<ColVector9d> tmpb(tmp.data(), tmp.size());
		ret = tmpA.jacobiSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(tmpb);
		return ret;
	}
		
	Eigen::Matrix3d _mat;
	std::vector<Eigen::Matrix3d> base;
	
	
};