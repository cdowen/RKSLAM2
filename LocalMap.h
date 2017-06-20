//
// Created by user on 17-5-9.
//

#ifndef RKSLAM2_LOCALMAP_H
#define RKSLAM2_LOCALMAP_H

#include "Frame.h"
#include "KeyFrame.h"
#include "tracking.h"
#include <vector>


class LocalMap
{
public:
	std::vector<Frame*> AllFrameInWindow;

	LocalMap(Tracking* tracking);
	void AddFrameInWindow(Frame* fr1);
	double ComputeParallax(cv::KeyPoint kp1,cv::KeyPoint kp2,KeyFrame* kf1,Frame* fr2);
	MapPoint* Triangulation(cv::KeyPoint kp1,cv::KeyPoint kp2,KeyFrame *kf1, Frame *fr2);
	MapPoint* GetPositionByOptimization(cv::KeyPoint kp1,cv::KeyPoint kp2, KeyFrame* kf1, Frame* fr2);
	void LocalOptimize(Frame* fr1);

private:
	const int FrameSizeInWindow=20;
	cv::Mat mK;
	Tracking* tr;

};




#endif //RKSLAM2_LOCALMAP_H
