#ifndef MATCHER_H
#define MATCHER_H

#include "Frame.h"
#include "KeyFrame.h"

class Matcher{
public:
	Matcher(){};
	int SearchForInitialization( Frame* fr1, Frame* fr2, int radius=15);
	std::map<int,int> MatchedPoints;
	int SearchMatchByGlobal(Frame* fr1, std::map<KeyFrame*, cv::Mat> globalH);
	int SearchMatchByLocal(Frame* currFrame, std::vector<KeyFrame*> kfs);
	std::map<cv::KeyPoint*, cv::KeyPoint*> matchByH(Frame* fr1, Frame* fr2, cv::Mat H);
	int MatchByLocalH(Frame *currFrame, KeyFrame *kfs);
  
private:
  	int SSDcompute(Frame* fr1, Frame*fr2, cv::KeyPoint kp1, cv::KeyPoint kp2);
  	int SSD_error_th=300;
  	const double globalSearchHalfLength = 5;
  	const int patchHalfSize = 4;


};

#endif//MATCHER_H