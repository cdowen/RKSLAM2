#ifndef MATCHER_H
#define MATCHER_H

#include "Frame.h"
#include "KeyFrame.h"
#include <Eigen/Core>

class Matcher{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Matcher(){};
	std::map<int ,int > SearchForInitialization( Frame* fr1, Frame* fr2);
	int SearchMatchByGlobal(Frame *fr1, std::map<KeyFrame *, Eigen::Matrix3d> globalH);
	int SearchMatchByLocal(Frame* currFrame, std::vector<KeyFrame*> kfs);
	std::map<int,int> matchByH(Frame* fr1, Frame* fr2, Eigen::Matrix3d& H);
	int MatchByLocalH(Frame *currFrame, KeyFrame *kfs);

private:
  	int SSDcompute(Frame* fr1, Frame*fr2, cv::KeyPoint kp1, cv::KeyPoint kp2);
  	int SSD_error_th=10000;
	int SSD_error_avg = 900;
  	const double globalSearchHalfLength = 5;
	const double InitSearchHalfLength=15;
  	const int patchHalfSize = 4;
	const double reprojError = 8;
	const double minSiftError=340000;

};

#endif//MATCHER_H