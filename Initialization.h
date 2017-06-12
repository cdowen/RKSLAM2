#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <opencv2/opencv.hpp>
#include "matcher.h"
#include "Frame.h"

class Tracking;
class Initialization
{
public:
  Initialization(Tracking* tracking, Frame* ReferenceFrame);
  //bool Initialize(const Frame &CurrentFrame, std::map<int,int> MatchedPoints,cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
  bool Initialize(const Frame& CurrentFrame, std::map<int,int>& MatchedPoints, cv::Mat& R21, cv::Mat& t21,std::vector<cv::Point3d> &vP3D, std::vector<bool> &vbTriangulated);
	cv::Mat DeltaH=cv::Mat::eye(3,3,CV_64FC1);
private:
  //for test.
    Frame* _ReferenceFrame;

  std::vector<cv::KeyPoint> mvKeys1; std::vector<cv::KeyPoint> mvKeys2;
  std::vector<cv::Point2f> mMatchedKeys1; std::vector<cv::Point2f> mMatchedKeys2;
  cv::Mat InlierH; cv::Mat InlierE;
  cv::Mat mK;
  cv::Mat R; cv::Mat t;
  int MaxIterations;
  

  //Compute Homography between source and destination planes. mvKeys1=H21*mvKeys2.
  void FindHomography(float& score, cv::Mat& H21);
  //Compute Essential Mat from the corresponding points in two images. mvKeys2*E21*mvKeys1=0.
  void FindEssentialMat(float& score, cv::Mat& E21);
  
  //Recover pose(R21 t21)  and structure(vP3D vbTriangulated) from Homography.
  // International Journal of Pattern Recognition and Artificial Intelligence, 1988
  bool RecoverPoseH(cv::Mat Homography, cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3d> &vP3D, std::vector<bool> &vbTriangulated,double minParallax, int minTriangulated);
  //Recover pose from EssentialMat.
  bool RecoverPoseE(cv::Mat EssentialMat, cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3d> &vP3D, std::vector<bool> &vbTriangulated,double minParallax, int minTriangulated);
  
  // Check R&t by computing triangulated points and its parallax. Return the number of visible 3Dpoints.
  // Maximum allowed reprojection error to treat a point pair as an inlier: 2 pixel.
  int CheckRT(const cv::Mat R, const cv::Mat t, cv::Mat vbMatchesInliers, std::vector<cv::Point3d> &vP3D, double th2, std::vector<bool> &vbGood, double &parallax);
  //Triangulate 3Dpoints
  void Triangulate(const cv::Point2f pt1, const cv::Point2f pt2, const cv::Mat P1, const cv::Mat P2, cv::Mat &x3D);
  //decompose EssentialMat for R and t.
  void decomposeEssentialMat(cv::InputArray _E, cv::OutputArray _R1, cv::OutputArray _R2, cv::OutputArray _t);
	void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
};

#endif//INITIALIZATION_H