#include "tracking.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include <stdio.h>

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
	cv::FAST(fr1->image, fr1->keypoints, fr1->Fast_threshold);
	cv::FAST(fr2->image, fr2->keypoints, fr2->Fast_threshold);


	Tracking tr;

	auto a = tr.ComputeHGlobalSBI(fr1, fr2);
	// Just for test
	Matcher Match;
	std::map<KeyFrame*, cv::Mat> vec;
	vec.insert(std::make_pair(static_cast<KeyFrame*>(fr1), a));
	int match_num_1 = Match.SearchMatchByGlobal(fr2, vec);
	auto b = tr.ComputeHGlobalKF(static_cast<KeyFrame*>(fr1), fr2);


	std::vector<cv::KeyPoint> kp1, kp2;
	std::vector<cv::DMatch> dm;
	int match_num = Match.SearchForInitialization(fr1, fr2, 4);
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
	imshow("matches", out);
	std::cout << "match by direct alignment:"<<match_num<<"\n";
	cv::waitKey(0);


}