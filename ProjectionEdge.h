#ifndef PROJECTIONEDGE_H
#define PROJECTIONEDGE_H
#include <g2o/core/base_unary_edge.h>
#include <Eigen/Core>
#include "sl3vertex.h"

class EdgeProjection :public g2o::BaseUnaryEdge < 2, Eigen::Vector2d, VertexSL3 >
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Eigen::Vector2i loc;
	EdgeProjection(){};
	void computeError();
	void linearizeOplus();
	virtual bool read(std::istream&);
	virtual bool write(std::ostream&) const;
};

#endif