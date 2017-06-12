#include "sl3vertex.h"
#include <iostream>
#include "Frame.h"
#include "KeyFrame.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include "tracking.h"
#include "Optimizer.h"
#include <opencv2/xfeatures2d.hpp>

// fr1 as the reference frame; project points in fr2 to fr1;
void testMatchByH(Frame* fr1, Frame* fr2, cv::Mat H)
{
	Matcher Match;
	std::map<KeyFrame*, cv::Mat> vec;
	vec.insert(std::make_pair(static_cast<KeyFrame*>(fr2), H));
	int match_num = Match.SearchMatchByGlobal(fr1, vec);
	std::cout<<"matched points by H:"<<match_num<<"\n";
}

void testProjection(Frame* lastFrame, Frame* currFrame, cv::Mat a = cv::Mat())
{
	if (lastFrame == NULL||currFrame==NULL)
	{
		lastFrame = new Frame();
		lastFrame->image = cv::imread("1.png", cv::IMREAD_GRAYSCALE);
		currFrame = new Frame();
		currFrame->image = cv::imread("2.png", cv::IMREAD_GRAYSCALE);
		cv::resize(lastFrame->image, lastFrame->sbiImg, cv::Size(40, 30));
		cv::GaussianBlur(lastFrame->sbiImg, lastFrame->sbiImg, cv::Size(0, 0), 0.75);
		cv::resize(currFrame->image, currFrame->sbiImg, cv::Size(40, 30));
		cv::GaussianBlur(currFrame->sbiImg, currFrame->sbiImg, cv::Size(0,0), 0.75);
	}
	cv::Mat_<double> input, res;
	input = cv::Mat(3, 1, CV_64FC1);
	res = cv::Mat(3, 1, CV_64FC1);
	if (a.empty())
	{
		a = Optimizer::ComputeHGlobalSBI(lastFrame, currFrame);
		// real answer
		//a = (cv::Mat_<double>(3,3)<<
		//						  1.006182e+000, 2.459331e-003, 1.633217e-003,
		//6.525023e-004, 1.013484e+000, -3.232950e-003,
		//			  -6.010459e-004, -2.420502e-002, 9.999996e-001);
		// for Shen Chenlong
		//a = (cv::Mat_<double>(3,3)<<
		//						  1.160721381656755, -0.008292699626746215, -40.02850611710956,
		//0.03435974520788249, 1.024597816043882, -13.14882445942777,
		//6.140997224529456e-05, 2.661706740529356e-06, 1);

	}
	cv::Mat &currImage = currFrame->image;
	cv::Mat reM = cv::Mat(currImage.size(), CV_8UC1, cv::Scalar(0));
	for (int i = 0; i < currImage.rows; i++)
	{
		for (int j = 0; j < currImage.cols; j++)
		{
			input(0) = j;input(1) = i;input(2) = 1;
			//std::cout<<"No:"<<i*currImage.cols+j<<std::endl;
			//res = a*input;
			res = a.inv()*input;
			//std::cout<<"before regulation:"<<res<<std::endl;
			res = res/res(2);
			//std::cout<<"after regulation:"<<res<<std::endl;
			if (res(0)>=0 && res(0) < currImage.size().width && res(1) >= 0 && res(1) < currImage.size().height)
			{
				//reM.at<uint8_t>(res(1),res(0)) = lastFrame->image.at<uint8_t>(i,j);
				reM.at<uint8_t>(i,j) =lastFrame->image.at<uint8_t>(res(1), res(0));
			}
		}
	}
	cv::imshow("original frame", lastFrame->image);
	cv::imshow("result", reM);
	cv::imshow("second frame", currImage);
	cv::Mat resultImg;
	cv::addWeighted(reM, 0.5, currImage, 0.5, 0.5, resultImg);
	cv::imshow("blended", resultImg);
	cv::Mat differ;
	cv::absdiff(reM, currImage, differ);
	differ = differ&(reM!=0);
	double error = differ.dot(differ);
	std::cout<<error/(640*480)<<"\n";
	cv::waitKey(0);
}

void drawMatch(Frame* lastFrame, Frame* currFrame, std::map<int,int> matches)
{
	int tmpd;
	std::vector<cv::DMatch> dm;
	std::vector<cv::KeyPoint> kp1, kp2;
	for (auto it = matches.begin(); it != matches.end(); ++it)
	{
		cv::DMatch tmp;
		tmp.imgIdx = 0;
		tmp.trainIdx = tmp.queryIdx = tmpd;
		dm.push_back(tmp);
		kp1.push_back(lastFrame->keypoints[it->second]);
		kp2.push_back(currFrame->keypoints[it->first]);
		tmpd++;
	}
	cv::Mat out;
	cv::drawMatches(lastFrame->image, kp1, currFrame->image, kp2, dm, out);
	imshow("matches", out);
	cv::waitKey(0);
}

void drawProjection(Frame* lastFrame, Frame* currFrame, std::map<int,int> matches)
{
	std::vector<cv::Point2f> kp1, kp2;
	for (auto it = matches.begin();it!=matches.end();it++)
	{
		kp1.push_back(lastFrame->keypoints[it->second].pt);
		kp2.push_back(currFrame->keypoints[it->first].pt);
	}
	cv::Mat mask;
	cv::Mat H = cv::findHomography(kp1, kp2, cv::RANSAC, 2, mask);
	testProjection(lastFrame, currFrame, H);
}

void findCorrespondenceByKp(Frame* lastFrame, Frame* currFrame, std::map<int,int>& matches)
{
	cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
	cv::Mat desc1, desc2;
	sift->detectAndCompute(lastFrame->image, cv::noArray(), lastFrame->keypoints, desc1);
	sift->detectAndCompute(currFrame->image, cv::noArray(), currFrame->keypoints, desc2);
	std::vector<cv::KeyPoint>& kp1 = lastFrame->keypoints;
	std::vector<cv::KeyPoint>& kp2 = currFrame->keypoints;
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> dMatches;
	matcher.match(desc1, desc2, dMatches, cv::noArray());
	double max_dist = 0; double min_dist = 100;
	for( int i = 0; i < desc1.rows; i++ )
	{ double dist = dMatches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	std::map<int,int> good_matches;
	std::vector<cv::DMatch> good_matches2;

	for( int i = 0; i < desc1.rows; i++ )
	{
		if( dMatches[i].distance <= std::max(3*min_dist, 0.02) )
		{
			good_matches2.push_back(dMatches[i]);
			good_matches.insert(std::make_pair(dMatches[i].trainIdx, dMatches[i].queryIdx));
		}
	}

	std::vector<cv::Point2f> kpp1, kpp2;
	for (auto it = good_matches.begin();it!=good_matches.end();it++)
	{
		kpp1.push_back(lastFrame->keypoints[it->second].pt);
		kpp2.push_back(currFrame->keypoints[it->first].pt);
	}
	cv::Mat mask;
	cv::Mat H = cv::findHomography(kpp1, kpp2, cv::RANSAC, 1, mask);
	auto iter = good_matches.begin();
	for (int i = 0;i<mask.rows*mask.cols;i++)
	{
		if (mask.at<uint8_t>(i)==1)
		{
			matches.insert(std::make_pair(iter->first, iter->second));
		}
		iter++;
	}
	std::cout<<"before filter:"<<good_matches.size()<<" after filter:"<<matches.size()<<"\n";
}
void drawMatchInitial(Frame* lastFrame, Frame* currFrame, std::vector<cv::Point2f>matches)
{
	int tmpd=0;
	std::vector<cv::DMatch> dm;
	std::vector<cv::KeyPoint> kp1, kp2;
	kp1=lastFrame->keypoints;
	cv::KeyPoint::convert(matches,kp2);
	for (int it=0;it<matches.size();++it)
	{
		cv::DMatch tmp;
		tmp.imgIdx = 0;
		tmp.trainIdx = tmp.queryIdx = tmpd;
		dm.push_back(tmp);
		tmpd++;
	}
	cv::Mat out;
	cv::drawMatches(lastFrame->image, kp1, currFrame->image, kp2, dm, out);
	imshow("matches", out);
	cv::waitKey(0);
}
