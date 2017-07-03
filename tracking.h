#ifndef TRACKING_H
#define TRACKING_H
#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "KeyFrame.h"
#include "Initialization.h"
class LocalMap;
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
	bool DecideKeyFrame(const Frame* fr1, int count);
	LocalMap* localMap;

private:
	void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
					std::vector<double> &vTimestamps);
	int minimalKeyFrameInterval=5;
	//For Initialization
	Initialization* Initializer;
	Frame* FirstFrame;
	Frame* SecondFrame;
	int PyramidLevel=5;
	const int cell_size=30;
	int ShiTScore_Threshold=20;
	//Initialize with VINS-mono.
	std::vector<Frame*>VINS_FramesInWindow;

	const int minimalKeyFrameInterval = 5;
};
#endif
