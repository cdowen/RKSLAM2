#include "sl3.h"
#include <iostream>
#include "Frame.h"
#include "KeyFrame.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include "tracking.h"

bool testSL3()
{
	Vector8d d;
	d << 1, 2, 3, 4, 5, 6, 7, 8;
	SL3 data;
	data.fromVector(d);
	std::cout << data._mat << "\n";
	for (int i = 0; i < 100; i++)
	{
		data.regularize();
	}
	Vector8d e;
	e = data.toVector();
	std::cout << e << "\n";
	return true;
}

// fr1 as the reference frame; project points in fr2 to fr1;
void testMatchByH(Frame* fr1, Frame* fr2, cv::Mat H)
{
	Matcher Match;
	std::map<KeyFrame*, cv::Mat> vec;
	vec.insert(std::make_pair(static_cast<KeyFrame*>(fr2), H));
	int match_num = Match.SearchMatchByGlobal(fr1, vec);
	std::cout<<"matched points by H:"<<match_num<<"\n";
}

void testProjection(Frame* lastFrame, Frame* currFrame)
{
	Tracking tr;
	cv::Mat_<double> input, res;
	input = cv::Mat(3, 1, CV_64FC1);
	res = cv::Mat(3,1,CV_64FC1);
	cv::Mat a = tr.ComputeHGlobalSBI(lastFrame, currFrame);
	cv::Mat reM = cv::Mat(currFrame->image.size(), CV_8UC1, cv::Scalar(0));
	for (int i = 0;i<currFrame->image.rows;i++)
	{
		for (int j = 0;j<currFrame->image.cols;j++)
		{
			input(0) = j;input(1) = i;input(2) = 1;
			res = a*input;
			res = res/res(2);
			if (res(0)>=0&&res(0)<currFrame->image.size().width&&res(1)>=0&&res(1)<currFrame->image.size().height)
			{
				reM.at<uint8_t>(res(1),res(0)) = lastFrame->image.at<uint8_t>(i,j);
			}
		}
	}
	cv::imshow("result", reM);
	cv::imshow("second frame", currFrame->image);
	cv::waitKey(0);
}