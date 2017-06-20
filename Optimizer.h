#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Core>
class Frame;
class KeyFrame;
class Optimizer
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef Eigen::Matrix<double, 7,1, Eigen::ColMajor> Vector7d;
	static Eigen::Matrix3d ComputeHGlobalSBI(Frame *fr1, Frame *fr2);
	static Eigen::Matrix3d ComputeHGlobalKF(KeyFrame *kf, Frame *fr2);
	static Vector7d PoseEstimation(Frame* fr);

	static cv::Mat mK;
};

#endif
