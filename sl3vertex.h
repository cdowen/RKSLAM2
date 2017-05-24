#ifndef SL3VERTEX_H
#define SL3VERTEX_H
#include <Eigen/Dense>
#include <g2o/core/base_vertex.h>
class VertexSL3 :public g2o::BaseVertex<9, Eigen::Matrix3d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef Eigen::Matrix<double, 9, 1> Vector9d;
		VertexSL3(){_estimate <<1,0,0,0,1,0,0,0,1;mu=0;}
	double mu;
	inline virtual void setToOriginImpl()
	{
		_estimate << 1,0,0,0,1,0,0,0,1;
		mu = 0;
	}

	virtual void oplusImpl(const double* update);

	virtual bool read(std::istream& is);
	virtual bool write(std::ostream& os) const;
};
#endif