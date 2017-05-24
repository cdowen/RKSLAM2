#include "sl3vertex.h"
#include <iostream>
#include "Frame.h"
#include "KeyFrame.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include "tracking.h"

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
	cv::Mat &currImage = currFrame->image;
	cv::Mat reM = cv::Mat(currImage.size(), CV_8UC1, cv::Scalar(0));
	for (int i = 0; i < currImage.rows; i++)
	{
		for (int j = 0; j < currImage.cols; j++)
		{
			input(0) = j;input(1) = i;input(2) = 1;
			res = a*input;
			if (res(0)>=0 && res(0) < currImage.size().width && res(1) >= 0 && res(1) < currImage.size().height)
			{
				reM.at<uint8_t>(res(1),res(0)) = lastFrame->image.at<uint8_t>(i,j);
			}
		}
	}
	cv::imshow("result", reM);
	cv::imshow("second frame", currImage);
	cv::Mat resultImg;
	cv::addWeighted(reM, 0.5, currImage, 0.5, 0.5, resultImg);
	cv::imshow("blended", resultImg);
	cv::Mat differ;
	cv::absdiff(reM, currImage, differ);
	differ = differ&(reM!=0);
	double error = differ.dot(differ);
	std::cout<<error/1200<<"\n";
	cv::waitKey(0);
}