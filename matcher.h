#ifndef MATCHER_H
#define MATCHER_H

#include "Frame.h"
#include "KeyFrame.h"

class Matcher{
public:
  Matcher(){};
  int SearchForInitialization( Frame* fr1, Frame* fr2, int radius=4);
  std::map<int,int> MatchedPoints;
  int SearchMatchByGlobal(Frame* fr1, std::map<KeyFrame*, cv::Mat> globalH);
  
private:
  int SSDcompute(Frame* fr1, Frame*fr2, cv::KeyPoint kp1, cv::KeyPoint kp2);
  std::unordered_multimap<cv::KeyPoint*, std::pair<Frame*, cv::KeyPoint*>> matchByH(Frame* fr1, Frame* fr2, cv::Mat H);
  cv::Mat warpPatch(Frame* fr1, cv::KeyPoint kp1, cv::Mat H);
  int SSD_error_th=200;
  const double globalSearchHalfLength = 5;
  const int patchHalfSize = 4;
};

#endif//MATCHER_H