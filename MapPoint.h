#ifndef MAPPOINT_H
#define MAPPOINT_H
#include <opencv2/opencv.hpp>
class KeyFrame;
class MapPoint
{
public:
	MapPoint(){Tw = cv::Mat::zeros(3,1, CV_64FC1);};
	// 4x4 matrix representing coordinates in world reference
	cv::Mat_<double> Tw;
	std::map<KeyFrame*, cv::KeyPoint*> allObservation;
};
#endif