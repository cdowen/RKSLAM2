#include "tracking.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include <stdio.h>
#include"Initialization.h"

bool testSL3();

int main()
{ 
	cv::Mat im1, im2;
	Frame* fr1 = new Frame();
	Frame* fr2 = new Frame();
	im1 = cv::imread("1.png", cv::IMREAD_GRAYSCALE);
	im2 = cv::imread("2.png", cv::IMREAD_GRAYSCALE);
	fr1->image = im1;
	fr2->image = im2;
	Tracking tr; tr.mK=(cv::Mat_<double>(3,3) <<517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);

	//TODO
	//auto a = tr.ComputeHGlobalSBI(fr1, fr2);

	cv::FAST(fr1->image, fr1->keypoints, fr1->Fast_threshold);
	cv::FAST(fr2->image, fr2->keypoints, fr2->Fast_threshold);
	Matcher Match;
	std::vector<cv::KeyPoint> kp1, kp2;
	std::vector<cv::DMatch> dm;
	int match_num = Match.SearchForInitialization(fr1, fr2, 15);
	int tmpd = 0;
	for (std::map<int, int>::iterator it = Match.MatchedPoints.begin(); it != Match.MatchedPoints.end(); ++it)
	{
		cv::DMatch tmp;
		tmp.imgIdx = 0;
		tmp.trainIdx = tmp.queryIdx = tmpd;
		dm.push_back(tmp);
		kp1.push_back(fr1->keypoints[it->first]);
		kp2.push_back(fr2->keypoints[it->second]);
		tmpd++;
	}
	cv::Mat out;
	cv::drawMatches(im1, kp1, im2, kp2, dm, out);
	//cv::imshow("matches", out);
	std::cout <<"Total number of matched points is: "<< match_num<<std::endl;
	
	Initialization* Initializer;
	Initializer= new Initialization(&tr,*fr1,2000);
	cv::Mat R21;cv::Mat t21;
	std::vector<cv::Point3f> vP3D; std::vector <bool> vbTriangulated;
	if(!Initializer->Initialize(*fr2, Match.MatchedPoints,R21,t21,vP3D,vbTriangulated))
	{
	  std::cout<<"Failed to Initialize";
	}else std::cout<<"System Initialized !";
	
	
	cv::waitKey(0);


	//getchar();


}