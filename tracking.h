#ifndef TRACKING_H
#define TRACKING_H
#include <opencv2/opencv.hpp>
#include "Frame.h"
class tracking
{
public:
	cv::Mat ComputeHGlobalSBI(Frame* fr1, Frame* fr2);
};
#endif
