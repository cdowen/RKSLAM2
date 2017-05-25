#ifndef OPTIMIZER_H
#define OPTIMIZER_H
class Frame;
class KeyFrame;
class Optimizer
{
public:
	static cv::Mat ComputeHGlobalSBI(Frame* fr1, Frame* fr2);
	static cv::Mat ComputeHGlobalKF(KeyFrame* kf, Frame* fr2);
	static cv::Mat PoseEstimation(Frame* fr);

	static cv::Mat mK;
};

#endif
