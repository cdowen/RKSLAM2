#include "tracking.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include <stdio.h>

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
	Tracking tr;

	auto a = tr.ComputeHGlobalSBI(fr1, fr2);

	cv::FAST(fr1->image, fr1->keypoints, fr1->Fast_threshold);
	cv::FAST(fr2->image, fr2->keypoints, fr2->Fast_threshold);
	Matcher Match;
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
	std::cout << match_num;
	cv::waitKey(0);


	getchar();


}