#ifndef MATCHER_H
#define MATCHER_H

#include "Frame.h"

class matcher{
public:
  matcher(){};
  int SearchForInitialization( Frame* fr1, Frame* fr2, int radius=4);
  std::map<int,int> MatchedPoints;
  
private:
  int SSDcompute(Frame* fr1, Frame*fr2, cv::KeyPoint kp1, cv::KeyPoint kp2);
  int SSD_error_th=200;
};

#endif//MATCHER_H