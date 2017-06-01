#ifndef TRACKING_H
#define TRACKING_H
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "KeyFrame.h"
#include "Initialization.h"
class Optimizer;
class Tracking
{
public:
	enum eTrackingState{
		SYSTEM_NOT_READY=-1,
		NO_IMAGES_YET=0,
		NOT_INITIALIZED=1,
		OK=2,
		LOST=3
	};
	eTrackingState mState=eTrackingState::NOT_INITIALIZED;
	eTrackingState mLastProcessedState;

	Tracking();
	void Run(std::string);
	std::vector<KeyFrame*> SearchTopOverlapping();

	Frame* currFrame, *lastFrame;
	cv::Mat mK;

private:
	void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
					std::vector<double> &vTimestamps);
	//For Initialization
	Initialization* Initializer;
	Frame* FirstFrame;
	Frame* SecondFrame;
	int nDesiredPoint=1000;
};
#endif
