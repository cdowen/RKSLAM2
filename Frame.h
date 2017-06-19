#ifndef FRAME_H
#define FRAME_H
#include <opencv2/opencv.hpp>
#include "MapPoint.h"
#include <unordered_map>
#include <Eigen/Core>

class KeyFrame;
class Frame
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	cv::Mat image;
	cv::Mat sbiImg;
	std::vector<cv::Mat> ImgPyrForInitial;

	unsigned long id;
	double timestamp;
	std::vector<cv::KeyPoint> keypoints;
	// store mappoints, NULL means no points;
	std::vector<MapPoint*> mappoints;
	const int Fast_threshold=40;
	//map a keypoint here with corresponding feature points in keyframe;
	typedef std::unordered_map<KeyFrame*, std::map<int,int>> MultiMatch;
	//std::unordered_multimap<KeyFrame*, std::pair<cv::KeyPoint*, cv::KeyPoint*>> matchedGroup;
	MultiMatch matchedGroup;
	std::map<KeyFrame*, Eigen::Matrix3d> keyFrameSet;
	//camera pose
	cv::Mat mTcw=cv::Mat::eye(4,4,CV_64F);
};
#endif