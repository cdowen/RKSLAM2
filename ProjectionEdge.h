#ifndef PROJECTIONEDGE_H
#define PROJECTIONEDGE_H
#include <g2o/core/base_unary_edge.h>
#include <Eigen/Core>
#include "sl3vertex.h"

class EdgeProjection :public g2o::BaseUnaryEdge < 2, Eigen::Vector2d, SL3 >
{

};

#endif