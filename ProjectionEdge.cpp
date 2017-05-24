#include "ProjectionEdge.h"

void EdgeProjection::computeError()
{
	Eigen::Vector3d hLoc;
	hLoc << loc[0], loc[1], 1;
	const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
	hLoc = v1->estimate()*hLoc;
	hLoc = hLoc / hLoc[2];
	Eigen::Vector2d res;
	res <<hLoc[0], hLoc[1];
	_error = res - _measurement;
}

void EdgeProjection::linearizeOplus()
{
	_jacobianOplusXi(0,0) = loc[0];
	_jacobianOplusXi(0,1) = loc[1];
	_jacobianOplusXi(0,2) = 1;
	_jacobianOplusXi(0,3) = 0;
	_jacobianOplusXi(0,4) = 0;
	_jacobianOplusXi(0,5) = 0;
	_jacobianOplusXi(0,6) = 0;
	_jacobianOplusXi(0,7) = 0;
	_jacobianOplusXi(1,0) = 0;
	_jacobianOplusXi(1,1) = 0;
	_jacobianOplusXi(1,2) = 0;
	_jacobianOplusXi(1,3) = loc[0];
	_jacobianOplusXi(1,4) = loc[1];
	_jacobianOplusXi(1,5) = 1;
	_jacobianOplusXi(1,6) = 0;
	_jacobianOplusXi(1,7) = 0;
}

bool EdgeProjection::write(std::ostream &os) const
{
	return os.good();
}

bool::EdgeProjection::read(std::istream &)
{
	return true;
}