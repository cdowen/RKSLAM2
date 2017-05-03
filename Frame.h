#ifndef FRAME_H
#define FRAME_H
#include <opencv2/opencv.hpp>
#include "MapPoint.h"
#include <unordered_map>

class KeyFrame;
class Frame
{
public:
	cv::Mat image;
	cv::Mat sbiImg;

	unsigned long id;
	double timestamp;
	std::vector<cv::KeyPoint> keypoints;
	// store mappoints, NULL means no points;
	std::vector<MapPoint*> mappoints;
	const int Fast_threshold=40;	
	//map a keypoint here with corresponding feature points in keyframe;
	std::unordered_multimap<KeyFrame*, std::pair<cv::KeyPoint*, cv::KeyPoint*>> matchedGroup;
	std::map<KeyFrame*, cv::Mat> keyFrameSet;
};
#endif