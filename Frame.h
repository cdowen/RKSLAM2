#ifndef FRAME_H
#define FRAME_H
#include <opencv2/opencv.hpp>
#include "MapPoint.h"
#include <unordered_map>

class Frame
{
public:
	cv::Mat image;
	unsigned long id;
	double timestamp;
	std::vector<cv::KeyPoint> keypoints;
	std::vector<MapPoint*> mappoints;
	const int Fast_threshold=40;	
	//map a keypoint here with corresponding feature points in keyframe;
	std::unordered_multimap<cv::KeyPoint*, std::pair<Frame*, cv::KeyPoint*>> matchedGroup;
};
#endif