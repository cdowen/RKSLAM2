#ifndef SL3VERTEX_H
#define SL3VERTEX_H
#include <Eigen/Dense>
#include <g2o/core/base_vertex.h>
class VertexSL3 :public g2o::BaseVertex<8, Eigen::Matrix3d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef Eigen::Matrix<double, 8, 1> Vector8d;
		VertexSL3(){_numOplusCalls=0; _estimate <<1,0,0,0,1,0,0,0,1;}
	static const int orthogonalizeAfter = 1000; 
	inline virtual void setToOriginImpl()
	{
		_estimate << 1,0,0,0,1,0,0,0,1;
	}

	virtual void oplusImpl(const double* update);

	virtual bool read(std::istream& is);
	Vector8d read();
	virtual bool write(std::ostream& os) const;
protected:
	int _numOplusCalls;
};
#endif