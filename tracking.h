#ifndef TRACKING_H
#define TRACKING_H
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "KeyFrame.h"

class Tracking
{
public:
	cv::Mat ComputeHGlobalSBI(Frame* fr1, Frame* fr2);
	cv::Mat ComputeHGlobalKF();
	std::vector<KeyFrame*> SearchTopOverlapping();

	Frame* currFrame, *lastFrame;
};
#endif
