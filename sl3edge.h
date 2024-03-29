#include "sl3vertex.h"
#include <g2o/core/base_unary_edge.h>
#include <opencv2/opencv.hpp>


class EdgeSL3 :public g2o::BaseUnaryEdge < 1, double, VertexSL3 >
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		EdgeSL3(){};
	cv::Mat _image, xgradient, ygradient;
	Eigen::Vector2d loc;

	Eigen::Vector2d homo_project();
	void computeError();
	void linearizeOplus();
	virtual bool read(std::istream&);
	virtual bool write(std::ostream&) const;
	bool isValid = true;
};