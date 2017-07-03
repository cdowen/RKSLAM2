#include "matcher.h"
#include <stdint.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "KeyFrame.h"
std::map<int,int> Matcher::SearchForInitialization(Frame* fr1, Frame* fr2)
{
	std::map<int,int> MatchedPoints={};
	for (int i = 0; i < fr1->keypoints.size(); i++)
	{
		cv::KeyPoint kp1 = fr1->keypoints.at(i);
		int x = kp1.pt.x;
		int y = kp1.pt.y;
		int min_SSD_error = SSD_error_th;  int Matched_id = -1;
		for (int j = 0; j < fr2->keypoints.size(); j++)
		{
			cv::KeyPoint kp2 = fr2->keypoints.at(j);
			if (abs(kp2.pt.x - x)<InitSearchHalfLength&&abs(kp2.pt.y-y)<InitSearchHalfLength)
			{
				int SSD_error = SSDcompute(fr1, fr2, kp1, kp2);
				//std::cout<<SSD_error<<std::endl;
				if (SSD_error < min_SSD_error)
				{
					min_SSD_error = SSD_error;
					Matched_id = j;
				}
			}
		}
		if (Matched_id>-1)
		{
			// also find backwards in a married-matches check.
			x=fr2->keypoints.at(Matched_id).pt.x;
			y=fr2->keypoints.at(Matched_id).pt.y;
			min_SSD_error=SSD_error_th; int Back_Matched_id=-1;
			for (int j = 0; j <fr1->keypoints.size() ; ++j)
			{
				cv::KeyPoint back_kp1=fr1->keypoints.at(j);
				if (abs(back_kp1.pt.x-x)<InitSearchHalfLength&&abs(back_kp1.pt.y-y)<InitSearchHalfLength)
				{
					int Back_SSD_error=SSDcompute(fr2,fr1,fr2->keypoints.at(Matched_id),back_kp1);
					if(Back_SSD_error<min_SSD_error)
					{
						min_SSD_error=Back_SSD_error;
						Back_Matched_id=j;
					}
				}
			}
			if (pow(fr1->keypoints[i].pt.x-fr1->keypoints[Back_Matched_id].pt.x,2)+pow(fr1->keypoints[i].pt.y-fr1->keypoints[Back_Matched_id].pt.y,2)<=4)
			{
				MatchedPoints[i] = Matched_id;
			}
		}
	}

	return MatchedPoints;
}

//compute the sum of squared difference (SSD) between patches(9x9).
int Matcher::SSDcompute(Frame* fr1, Frame* fr2, cv::KeyPoint kp1, cv::KeyPoint kp2)
{
	int x1 = kp1.pt.x; int y1 = kp1.pt.y;
	int x2 = kp2.pt.x; int y2 = kp2.pt.y;
	int width = fr1->image.size().width;
	int height = fr1->image.size().height;
	if (x1-patchHalfSize<0||x1+patchHalfSize+1>width||x2-patchHalfSize<0||x2+patchHalfSize+1>width||y1-patchHalfSize<0||y1+patchHalfSize+1>height||y2-patchHalfSize<0||y2+patchHalfSize+1>height)
	{
		return SSD_error_th;
	} else
	{
		cv::Mat fr1_Range = fr1->image.colRange(x1 - patchHalfSize, x1 + patchHalfSize+1).rowRange(y1 - patchHalfSize, y1 + patchHalfSize+1);
		cv::Mat fr2_Range = fr1->image.colRange(x2 - patchHalfSize, x2 + patchHalfSize+1).rowRange(y2 - patchHalfSize, y2 + patchHalfSize+1);

		cv::Mat differ = fr1_Range - fr2_Range;
		return (int)differ.dot(differ);
	}
}

int Matcher::SearchMatchByGlobal(Frame *fr1, std::map<KeyFrame *, Eigen::Matrix3d> globalH)
{
	int count = 0;
	for (auto i = globalH.begin();i!=globalH.end(); i++)
	{
		auto tmpd = matchByH(i->first,fr1, i->second);
		fr1->matchedGroup.insert(std::make_pair(i->first, tmpd));
		count+=tmpd.size();
		std::cout<<"matched with kp size: "+std::to_string(i->first->timestamp)+"->"+std::to_string(fr1->timestamp)+":"<<tmpd.size()<<"\n";
	}
	return count;
}

// Find corresponding feature points in two frames with homography
// fr1 represent keyframe to project.
// TODO: different search range for well and ill conditioned points
std::map<int,int> Matcher::matchByH(Frame* fr1, Frame* fr2, Eigen::Matrix3d& h)
{
#ifdef USE_SIFT
	//match with SIFT
	cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> sift = cv::xfeatures2d::SiftDescriptorExtractor::create();
	cv::Mat desc1, desc2;
	sift->compute(fr1->image, fr1->keypoints, desc1);
	sift->compute(fr2->image, fr2->keypoints, desc2);
#endif
	std::map<int,int> MatchedPoints={};
	// kpl:points in fr1.
	Eigen::Vector2d kpl;
	// ppl:points projected to fr2.
	Eigen::Vector2d ppl;
	const Eigen::Matrix3d hinv = h.inverse();
	int width = fr1->image.size().width;
	int height = fr1->image.size().height;
	for (int i = 0; i < fr1->keypoints.size(); i++)
	{
		cv::KeyPoint kp = fr1->keypoints[i];
		kpl(0) = kp.pt.x; kpl(1) = kp.pt.y;
		ppl = (h*kpl.homogeneous()).hnormalized();
#ifndef USE_SIFT
		cv::Mat_<uint8_t> warped(patchHalfSize * 2, patchHalfSize * 2, uint8_t(0));
		int warpedSize = 0;
		//calculate warped patch.
		for (int ii = -patchHalfSize; ii < patchHalfSize; ii++)
		{
			for (int jj = -patchHalfSize; jj < patchHalfSize; jj++)
			{
				// equation 7 in rkslam.
				Eigen::Vector3d tmp;
				tmp = h*kpl.homogeneous();
				tmp = tmp/tmp(2);
				tmp(0)+=ii;tmp(1)+=jj;
				tmp = (hinv*tmp)/tmp(2);
				if (tmp(0)>0 && tmp(0) < width&&tmp(1) > 0 && tmp(1) < height)
				{
					warped(jj + patchHalfSize, ii + patchHalfSize) = fr1->image.at<uint8_t>(tmp(1), tmp(0));
				}
			}
		}
		// image warp failed;
		warpedSize = cv::countNonZero(warped);
		if (warpedSize==0)
		{
			continue;
		}
#endif
		int min_SSD_error = SSD_error_avg;
		int min_SIFT_error = minSiftError;
		// iterate to find the smallest error in fr2.
		int Matched_id=-1;
		cv::Mat differ;
		for (int j = 0; j < fr2->keypoints.size(); j++)
		{
			cv::Point2f pt2 = fr2->keypoints[j].pt;
			//if (((hinv*tmp.homogeneous()).hnormalized()-kpl).squaredNorm()>reprojError)
			//{
			//	continue;
			//}
			if (abs(ppl(0) - pt2.x)<globalSearchHalfLength&&abs(ppl(1)-pt2.y)<globalSearchHalfLength)
			{
#ifdef USE_SIFT
				cv::Mat result;
				cv::absdiff(desc1.row(i), desc2.row(j), result);
				double thres = result.dot(result);
				//std::cout<<result<<"\n";
				if (thres<min_SIFT_error)
				{
					min_SIFT_error = thres;
					Matched_id = j;
				}
#else
				int u = pt2.y-patchHalfSize;int d = pt2.y+patchHalfSize;int l = pt2.x-patchHalfSize;int r = pt2.x+patchHalfSize;
				if (u<0||d>=height||l<0||r>=width)
				{
					continue;
				}
				cv::Mat patch = fr2->image.rowRange(pt2.y - patchHalfSize, pt2.y + patchHalfSize).colRange(pt2.x - patchHalfSize, pt2.x + patchHalfSize);
				cv::absdiff(warped-cv::mean(warped), patch-cv::mean(patch), differ);
				differ = differ&(warped!=0);
				int ssdError = differ.dot(differ);
				ssdError = ssdError/warpedSize;
				if (ssdError < min_SSD_error)
				{
					min_SSD_error = ssdError;
					Matched_id = j;
				}
#endif
			}
		}
		if (Matched_id>-1)
		{
			MatchedPoints.insert(std::make_pair(Matched_id,i));
		}
	}
	return MatchedPoints;
}

inline cv::Mat warpPoint(cv::Mat_<double> point, cv::Mat H)
{
	return H*point / point(2);
}


int Matcher::MatchByLocalH(Frame* currFrame, KeyFrame* kfs)
{
	const int MAX_LOCAL_HOMO = 5;
	const int RANSAC_TH = 3;
	int matchNum = 0;
	for (auto iter = currFrame->matchedGroup.begin();iter!=currFrame->matchedGroup.end(); iter++)
	{
		KeyFrame* KFs=iter->first;
		std::vector<cv::Point2f> fkp, kfkp;
		for (auto iter2:iter->second)
		{
			fkp.push_back(currFrame->keypoints[iter2.first].pt);
			kfkp.push_back(KFs->keypoints[iter2.second].pt);
		}
		for (int i = 0;i<MAX_LOCAL_HOMO;i++)
		{
			cv::Mat mask;
			// minimum points count for cv::findHomography
			if (kfkp.size()<4)
			{
				break;
			}
			cv::Mat homo = cv::findHomography(kfkp, fkp, CV_RANSAC, RANSAC_TH, mask);
			if (homo.empty())
			{
				std::cout<<"cannot calculate homography"<<std::endl;
				continue;
			}
			// remove inliners
			for (int j = 0;j<fkp.size();j++)
			{
				if (mask.at<uint8_t>(j)!=0)
				{
					fkp.erase(fkp.begin()+j);
					kfkp.erase(kfkp.begin()+j);
				}
			}
			Eigen::Matrix3d ehomo;
			cv::cv2eigen(homo, ehomo);
			auto ret = matchByH(kfs,currFrame,ehomo);
			iter->second.insert(ret.begin(), ret.end());
			matchNum+=ret.size();
		}
	}
	return matchNum;
}


int Matcher::SearchMatchByLocal(Frame* currFrame, std::vector<KeyFrame*> kfs)
{
	int totalMatchNum = 0;
	for (int i = 0;i<kfs.size();i++)
	{
		totalMatchNum+=MatchByLocalH(currFrame, kfs[i]);
	}
	return totalMatchNum;
}


