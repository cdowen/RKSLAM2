#include <opencv2/opencv.hpp>
#include "tracking.h"
#include <Eigen/Sparse>
#include "Map.h"
#include <opencv2/core/eigen.hpp>
#include "LocalMap.h"
#include <math.h>
#include <chrono>
#include "Optimizer.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vikit/vision.h>

Tracking::Tracking():Initializer(static_cast<Initialization*>(NULL)){};

enum InitializeMethod
{
	Delta_H=0,
	LK=1,
	CoarseToFine=2
};


void Tracking::Run(std::string pathtoData)
{
	void drawMatch(Frame* lastFrame, Frame* currFrame, std::map<cv::KeyPoint*, cv::KeyPoint*> matches);
	void testProjection(Frame* lastFrame, Frame* currFrame, cv::Mat a = cv::Mat());
	void findCorrespondenceByKp(Frame* lastFrame, Frame* currFrame, std::map<int,int>& matches);
	//Load Images.
	std::vector<std::string> vstrImageFilenames;
	std::vector<double> vTimestamps;
	std::string strFile=pathtoData+"/rgb.txt";
	LoadImages(strFile,vstrImageFilenames,vTimestamps);

	int nImages=vstrImageFilenames.size();

	for(int ni=0;ni<nImages;ni++)
	{
		InitializeMethod Method=LK;
		cv::Mat im=cv::imread(pathtoData+"/"+vstrImageFilenames[ni],cv::IMREAD_GRAYSCALE);
		double tframe=vTimestamps[ni];
		Frame* fr = new Frame();
		cv::resize(im, fr->sbiImg, cv::Size(40, 30));
		cv::GaussianBlur(fr->sbiImg, fr->sbiImg, cv::Size(0, 0), 0.75);
		fr->image = im;
		fr->id = ni;
		if(Method==LK)
		{
			//Build Image Pyramid.
			fr->ImgPyrForInitial.resize(PyramidLevel);
			fr->ImgPyrForInitial[0]=fr->image;
			for (int i = 1; i < PyramidLevel; ++i)
			{
				fr->ImgPyrForInitial[i]=cv::Mat(fr->ImgPyrForInitial[i-1].rows/2,fr->ImgPyrForInitial[i-1].cols/2,CV_8U);
				vk::halfSample(fr->ImgPyrForInitial[i-1],fr->ImgPyrForInitial[i]);
			}
		}
		fr->timestamp = tframe;
		lastFrame = currFrame;
		currFrame = fr;

		//Initialize.
		if(mState==NOT_INITIALIZED)
		{
			if(Method==LK)
			{
				if (!Initializer)
				{
					//Detect Fast Feature with Shi-Tomasi-Score and grid.
					int grid_n_cols_=std::ceil(static_cast<double>(fr->image.cols)/cell_size);
					int grid_n_rows_=std::ceil(static_cast<double>(fr->image.rows)/cell_size);
					std::vector<float > score_at_grid(grid_n_cols_*grid_n_rows_,0);
					std::vector<cv::KeyPoint>best_KeyPoint_at_grid(grid_n_cols_*grid_n_rows_);
					for (int i = 0; i <PyramidLevel ; ++i)
					{
						const int scale=(1<<i);
						std::vector<cv::KeyPoint> CandidateKeyPoint;
						cv::FAST(fr->ImgPyrForInitial[i],CandidateKeyPoint,20,true);
						for (auto it=CandidateKeyPoint.begin();it!=CandidateKeyPoint.end();++it)
						{
							float score=vk::shiTomasiScore(fr->ImgPyrForInitial[i],it->pt.x,it->pt.y);
							//Divide the image in cells of fixed size(30*30)
							int k= static_cast<int>((it->pt.y*scale)/cell_size)*grid_n_cols_+ static_cast<int>((it->pt.x*scale)/cell_size);
							if (score>score_at_grid[k])
							{
								cv::Point2f pt;pt.x=it->pt.x*scale;pt.y=it->pt.y*scale;
								best_KeyPoint_at_grid[k]=cv::KeyPoint(pt,1,-1,score);
							}
						}
					}
					//Leave the points above the threshold.
					for (std::vector<cv::KeyPoint>::iterator kp=best_KeyPoint_at_grid.begin();kp!=best_KeyPoint_at_grid.end();++kp)
					{
						if (kp->response>ShiTScore_Threshold)fr->keypoints.push_back(*kp);
					}
					std::cout<<fr->keypoints.size()<<" is detected by Fast.\n";

					if (fr->keypoints.size() >= 100)
					{
						Initializer = new Initialization(this, fr);
						FirstFrame = fr;
						std::cout << ni << "th(" << std::setprecision(16) << tframe
								  << ") image is selected as FirstFrame!\n";
					}
				}
				else
				{
					std::cout << ni << "th(" << std::setprecision(16) << tframe
							  << ") image is selected as SecondFrame!\n";
					//Match by LK.
					std::vector<cv::Point2f>first_frame_point,matched_point_LK;
					for (int i = 0; i <FirstFrame->keypoints.size() ; ++i)
					{
						first_frame_point.push_back(FirstFrame->keypoints[i].pt);
						matched_point_LK.push_back(FirstFrame->keypoints[i].pt);
					}
					std::vector<uchar >status;
					std::vector<float >error;
					const int klt_win_size=30;
					const int klt_max_iter=30;
					const double klt_eps=0.001;
					cv::TermCriteria termcrit (cv::TermCriteria::COUNT+cv::TermCriteria::EPS,klt_max_iter,klt_eps);
					cv::calcOpticalFlowPyrLK(FirstFrame->ImgPyrForInitial[0],fr->ImgPyrForInitial[0],first_frame_point,matched_point_LK,status,error,cv::Size2i(klt_win_size,klt_win_size),4,termcrit);
					std::vector<cv::KeyPoint>::iterator First_keypoint_it=FirstFrame->keypoints.begin();
					std::vector<cv::Point2f>::iterator matched_point_LK_it=matched_point_LK.begin();
					for (size_t i = 0; First_keypoint_it!=FirstFrame->keypoints.end(); ++i)
					{
						if(!status[i])
						{
							First_keypoint_it=FirstFrame->keypoints.erase(First_keypoint_it);
							matched_point_LK_it=matched_point_LK.erase(matched_point_LK_it);
							continue;
						}
						++First_keypoint_it;
						++matched_point_LK_it;
					}
					std::cout << "Match " << matched_point_LK.size() << " points. \n";

					if (matched_point_LK.size() < 150)
					{
						delete Initializer;
						Initializer = static_cast<Initialization*>(NULL);
						continue;
					}

					//for test: draw matches when initializing with LK.
					//void drawMatchInitial(Frame* lastFrame, Frame* currFrame, std::vector<cv::Point2f>matches);
					//drawMatchInitial(FirstFrame,fr,matched_point_LK);
					SecondFrame = fr;
					cv::KeyPoint::convert(matched_point_LK,SecondFrame->keypoints);

					cv::Mat R21;
					cv::Mat t21;
					std::vector<cv::Point3d> vP3D;
					std::vector<bool> vbTriangulated;
					std::map<int,int>MatchedPoints;
					for (int j = 0; j <matched_point_LK.size() ; ++j)
					{
						MatchedPoints.insert(std::make_pair(j,j));
					}
					if (!Initializer->Initialize(*SecondFrame, MatchedPoints, R21, t21, vP3D,
													  vbTriangulated))
					{
						std::cout << "Failed to Initialize.\n\n";

					} else
					{
						std::cout << "System Initialized !\n\n";
						//for test: compare the precision of the H between using sbi and using opencv::computeH.
						//void testProjection(Frame *lastFrame, Frame *currFrame, cv::Mat a = cv::Mat());
						//testProjection(FirstFrame, SecondFrame);
						//testProjection(FirstFrame,SecondFrame,Initializer->DeltaH);

						// store mTcw of keyframes
						//TODO
						std::cout<<"R = "<<R21<<std::endl;
						std::cout<<"t = "<<t21<<std::endl;
						SecondFrame->mTcw.colRange(0,3).rowRange(0,3)=R21;
						SecondFrame->mTcw.colRange(0,3).row(3)=t21;

						// store points in keyframe
						FirstFrame->mappoints.reserve(FirstFrame->keypoints.size());
						std::fill(FirstFrame->mappoints.begin(), FirstFrame->mappoints.end(), nullptr);
						SecondFrame->mappoints.reserve(SecondFrame->keypoints.size());
						std::fill(SecondFrame->mappoints.begin(), SecondFrame->mappoints.end(), nullptr);
						Map *map = Map::getInstance();
						std::map<int,int> mapData;
						for (std::map<int,int>::iterator MatchedPair=MatchedPoints.begin();MatchedPair!=MatchedPoints.end();++MatchedPair)
						{
							int vbPointNum = 0;

							mapData.insert(std::make_pair(MatchedPair->second, MatchedPair->first));
							if (vbTriangulated[vbPointNum])
							{
								MapPoint *mp = new MapPoint;
								mp->Tw(0) = vP3D[vbPointNum].x;
								mp->Tw(1) = vP3D[vbPointNum].y;
								mp->Tw(2) = vP3D[vbPointNum].z;
								mp->allObservation.insert(
										std::make_pair(static_cast<KeyFrame *>(FirstFrame),
													   &FirstFrame->keypoints[MatchedPair->first]));
								mp->allObservation.insert(std::make_pair(static_cast<KeyFrame *>(SecondFrame),
																		 &SecondFrame->keypoints[MatchedPair->second]));
								FirstFrame->mappoints[vbPointNum] = mp;
								SecondFrame->mappoints[MatchedPair->second] = mp;
								map->allMapPoint.push_back(mp);
							}
							++vbPointNum;
						}

						SecondFrame->matchedGroup.insert(std::make_pair(static_cast<KeyFrame *>(FirstFrame), mapData));
						map->allKeyFrame.push_back(static_cast<KeyFrame *>(FirstFrame));
						map->allKeyFrame.push_back(static_cast<KeyFrame *>(SecondFrame));
						lastFrame = SecondFrame;
						mState = OK;

						continue;
					}
				}
			}
		}
			if(mState==OK)
			{
				std::cout<<"Tracking..."<<std::endl;
				Map* map = Map::getInstance();
				auto kf = map->allKeyFrame.back();
				auto a = Optimizer::ComputeHGlobalSBI(lastFrame, currFrame);
				testProjection(lastFrame,currFrame, a);
				std::map<int,int>matches;
				findCorrespondenceByKp(lastFrame, currFrame,matches);
				std::vector<KeyFrame*> kfs = SearchTopOverlapping();
				std::map<KeyFrame*, cv::Mat> khs;

				for (int i = 0;i<kfs.size();i++)
				{
					cv::Mat b = Optimizer::ComputeHGlobalKF(kfs[i], lastFrame);
					//testProjection(kfs[i], lastFrame, b);
					//findCorrespondenceByKp(kfs[i], lastFrame);
					void drawMatch(Frame* lastFrame, Frame* currFrame, std::map<cv::KeyPoint*, cv::KeyPoint*> matches);
					//drawMatch(kfs[i], lastFrame, lastFrame->matchedGroup[kfs[i]]);
					cv::Mat c = b*a;
					khs.insert(std::make_pair(kfs[i], c));
				}
				Matcher match;
				std::cout<<"Matched points with keyframe:"<<match.SearchMatchByGlobal(currFrame, khs)<<"\n";
				std::cout<<"Matched points with local homo:"<<match.SearchMatchByLocal(currFrame, kfs)<<"\n";

				//void testMatchHomo(Frame* lastFrame, Frame* currFrame, std::map<cv::KeyPoint*, cv::KeyPoint*> matches);
				//void findCorespondenceByKp(Frame* lastFrame, Frame* currFrame);
				//testMatchHomo(lastFrame, currFrame, currFrame->matchedGroup[static_cast<KeyFrame*>(lastFrame)]);
				//std::cout<<"Matched points with keyframe:"<<match.SearchMatchByGlobal(currFrame, khs)<<"\n";
				//testProjection(khs.begin()->first, currFrame);
				//findCorespondenceByKp(khs.begin()->first, currFrame);
				//std::cout<<"Matched points with local homo:"<<match.SearchMatchByLocal(currFrame, kfs)<<"\n";

				/*
				// match with direct alignment.
				int datanum = 0;
				for (int i = 0;i<kfs.size(); i++)
				{
					datanum += match.SearchForInitialization(kfs[i], currFrame, 15);
					std::unordered_multimap<KeyFrame*, std::pair<cv::KeyPoint*, cv::KeyPoint*>> data;
					for (auto j = match.MatchedPoints.begin();j!=match.MatchedPoints.end();j++)
					{
						data.insert(std::make_pair(kfs[i], std::make_pair(&currFrame->keypoints[j->first], &kfs[i]->keypoints[j->second])));
					}
					currFrame->matchedGroup = data;
				}
				std::cout<<"Matched points with keyframe:"<<datanum<<"\n";
				*/


				//std::map<KeyFrame*, cv::Mat> vec;
				//vec.insert(std::make_pair(static_cast<KeyFrame*>(fr1), a));
				//int match_num_1 = Match.SearchMatchByGlobal(fr2, vec);
				//auto b = tr.ComputeHGlobalKF(static_cast<KeyFrame*>(fr1), fr2);
			}
		}
	//Local Mapping.




	}

// search for x most overlapping keyframe with last frame.
std::vector<KeyFrame*> Tracking::SearchTopOverlapping()
{
	const int MAX_KEYFRAME_COUNT = 5;
	std::map<KeyFrame*, int> kfv;
	for (auto mit = lastFrame->matchedGroup.begin();mit!=lastFrame->matchedGroup.end();mit++)
	{
		kfv.insert(std::make_pair(mit->first, mit->second.size()));
	}
	std::vector<KeyFrame*> kfs;
	for (int i = 0;i<MAX_KEYFRAME_COUNT;i++)
	{
		int max_v = 0;
		KeyFrame* max_kf = nullptr;
		for (std::map<KeyFrame*, int>::iterator it = kfv.begin();it!=kfv.end();it++)
		{
			if (it->second>max_v)
			{
				max_v = it->second;
				max_kf = it->first;
			}
		}
		if (max_kf != nullptr)
		{
			kfs.push_back(max_kf);
			kfv.erase(max_kf);
		}
	}
	return kfs;
}

void Tracking::LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames, std::vector<double> &vTimestamps)
{
	std::ifstream f;
	f.open(strFile.c_str());

	// skip first three lines
	std::string s0;
	std::getline(f,s0);
	std::getline(f,s0);
	std::getline(f,s0);

	while(!f.eof())
	{
		std::string s;
		getline(f,s);
		if(!s.empty())
		{
			std::stringstream ss;
			ss << s;
			double t;
			std::string sRGB;
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrImageFilenames.push_back(sRGB);
		}
	}
}
