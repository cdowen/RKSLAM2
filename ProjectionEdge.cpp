#include <cfloat>
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
	const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
	const Eigen::Matrix3d h = v1->estimate();
	double ww = h(2, 0)*loc[0]+h(2,1)*loc[1]+1;
	ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
	double xi = (h(0,0)*loc[0]+h(0,1)*loc[1]+h(0,2))*ww;
	double yi = (h(1,0)*loc[0]+h(1,1)*loc[1]+h(1,2))*ww;

	_jacobianOplusXi(0,0) = loc[0]*ww;
	_jacobianOplusXi(0,1) = loc[1]*ww;
	_jacobianOplusXi(0,2) = ww;
	_jacobianOplusXi(0,3) = 0;
	_jacobianOplusXi(0,4) = 0;
	_jacobianOplusXi(0,5) = 0;
	_jacobianOplusXi(0,6) = -loc[0]*ww*xi;
	_jacobianOplusXi(0,7) = -loc[1]*ww*xi;
	_jacobianOplusXi(1,0) = 0;
	_jacobianOplusXi(1,1) = 0;
	_jacobianOplusXi(1,2) = 0;
	_jacobianOplusXi(1,3) = loc[0]*ww;
	_jacobianOplusXi(1,4) = loc[1]*ww;
	_jacobianOplusXi(1,5) = ww;
	_jacobianOplusXi(1,6) = -loc[0]*ww*yi;
	_jacobianOplusXi(1,7) = -loc[1]*ww*yi;
}

bool EdgeProjection::write(std::ostream &os) const
{
	return os.good();
}

bool::EdgeProjection::read(std::istream &)
{
	return true;
}