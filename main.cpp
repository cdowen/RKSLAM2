#include "tracking.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include <stdio.h>
#include"Initialization.h"

bool testSL3();
void testMatchByH(Frame* fr1, Frame* fr2, cv::Mat H);

int main()
{ 
	cv::Mat im1, im2;
	Frame* fr1 = new Frame();
	Frame* fr2 = new Frame();
	im1 = cv::imread("1.png", cv::IMREAD_GRAYSCALE);
	im2 = cv::imread("2.png", cv::IMREAD_GRAYSCALE);

	cv::resize(im1, fr1->sbiImg, cv::Size(40, 30));
	cv::GaussianBlur(fr1->sbiImg, fr1->sbiImg, cv::Size(0, 0), 0.75);
	cv::resize(im2, fr2->sbiImg, cv::Size(40, 30));
	cv::GaussianBlur(fr2->sbiImg, fr2->sbiImg, cv::Size(0, 0), 0.75);
	fr1->image = im1;
	fr2->image = im2;
	Tracking tr; tr.mK=(cv::Mat_<double>(3,3) <<517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);

	//TODO
	//auto a = tr.ComputeHGlobalSBI(fr1, fr2);

	cv::FAST(fr1->image, fr1->keypoints, fr1->Fast_threshold);
	cv::FAST(fr2->image, fr2->keypoints, fr2->Fast_threshold);
	Matcher Match;
	std::map<KeyFrame*, cv::Mat> vec;
	vec.insert(std::make_pair(static_cast<KeyFrame*>(fr1), a));
	int match_num_1 = Match.SearchMatchByGlobal(fr2, vec);
	auto b = tr.ComputeHGlobalKF(static_cast<KeyFrame*>(fr1), fr2);


	std::vector<cv::KeyPoint> kp1, kp2;
	std::vector<cv::DMatch> dm;
	int match_num = Match.SearchForInitialization(fr1, fr2, 15);
	int tmpd = 0;
	for (auto it = fr2->matchedGroup.begin(); it != fr2->matchedGroup.end(); ++it)
	{
		cv::DMatch tmp;
		tmp.imgIdx = 0;
		tmp.trainIdx = tmp.queryIdx = tmpd;
		dm.push_back(tmp);
		kp1.push_back(*it->second.second);
		kp2.push_back(*it->first);
		tmpd++;
	}
	cv::Mat out;
	cv::drawMatches(im1, kp1, im2, kp2, dm, out);
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


	imshow("matches", out);
	std::cout << "match by direct alignment:"<<match_num<<"\n";
	cv::waitKey(0);


}