#ifndef MAPPOINT_H
#define MAPPOINT_H
#include <opencv2/opencv.hpp>
class Frame;
class MapPoint
{
public:
	MapPoint():Tw(cv::Mat::zeros(3,1,CV_64F)),WellConstrained(false){};
	// 4x4 matrix representing coordinates in world reference
	cv::Mat_<double> Tw;
	std::map<Frame*, cv::KeyPoint*> allObservation;
	bool WellConstrained;
};
#endif