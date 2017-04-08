#ifndef FRAME_H
#define FRAME_H
#include <opencv2/opencv.hpp>
class Frame
{
public:
	cv::Mat image;
	std::vector<cv::KeyPoint>keypoints;
	int Fast_threshold=40;

};
#endif