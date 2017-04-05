#include "sl3vertex.h"
#include <g2o/core/base_unary_edge.h>
#include <opencv2/opencv.hpp>

class VertexSL3;
class EdgeSL3 :public g2o::BaseUnaryEdge < 1, unsigned char, VertexSL3 >
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		EdgeSL3(){};
	cv::Mat* image, *xgradient, *ygradient;
	Eigen::Vector2i loc;
	//TODO: 处理投影到图像外的点
	Eigen::Vector3d homo_project();
	void computeError();
	void linearOplus();
	virtual bool read(std::istream&);
	virtual bool write(std::ostream&) const;
};