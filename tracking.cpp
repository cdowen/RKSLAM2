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
Tracking::Tracking():Initializer(static_cast<Initialization*>(NULL)){};

void Tracking::Run(std::string pathtoData)
{

	//for test.
	std::ofstream file;
	file.open("file.txt");
	void drawMatch(Frame* lastFrame, Frame* currFrame, std::map<cv::KeyPoint*, cv::KeyPoint*> matches);
	void testProjection(Frame* lastFrame, Frame* currFrame, cv::Mat a = cv::Mat());
	void findCorrespondenceByKp(Frame* lastFrame, Frame* currFrame);
	//Load Images.
	std::vector<std::string> vstrImageFilenames;
	std::vector<double> vTimestamps;
	std::string strFile=pathtoData+"/rgb.txt";
	LoadImages(strFile,vstrImageFilenames,vTimestamps);

	int nImages=vstrImageFilenames.size();

	for(int ni=50;ni<nImages;ni++)
	{
		cv::Mat im=cv::imread(pathtoData+"/"+vstrImageFilenames[ni],cv::IMREAD_GRAYSCALE);
		double tframe=vTimestamps[ni];
		Frame* fr = new Frame();
		cv::resize(im, fr->sbiImg, cv::Size(40, 30));
		cv::GaussianBlur(fr->sbiImg, fr->sbiImg, cv::Size(0, 0), 0.75);
		fr->image = im;
		fr->id = ni;
		cv::FAST(fr->image, fr->keypoints, fr->Fast_threshold, true);
		if(fr->keypoints.size()>nDesiredPoint)
		{
			cv::KeyPointsFilter::retainBest(fr->keypoints,nDesiredPoint);
			fr->keypoints.resize(nDesiredPoint);
		}
		std::cout<<"The number of FAST: "<<fr->keypoints.size()<<std::endl;
		fr->mappoints.reserve(fr->keypoints.size());
		std::fill(fr->mappoints.begin(), fr->mappoints.end(), nullptr);
		fr->timestamp = tframe;
		lastFrame = currFrame;
		currFrame = fr;
		//Initialize.
		if(mState==NOT_INITIALIZED)
		{
			if (!Initializer)
			{
				if (fr->keypoints.size() >= 100)
				{
					Initializer = new Initialization(this, *fr, 2000);
					FirstFrame = fr;
					std::cout << ni << "th(" <<std::setprecision(16) << tframe
							  << ") image is selected as FirstFrame!\n";
				}
			} else
			{
				std::cout << ni << "th(" << std::setprecision(16) << tframe << ") image is selected as SecondFrame!\n";
				if (fr->keypoints.size() < 100)
				{
					delete Initializer;
					Initializer = nullptr;
					continue;
				}

				Matcher Match;

				std::vector<cv::KeyPoint> kp1, kp2;

				//for test.
				std::vector<cv::DMatch>match1to2;
				std::vector<cv::KeyPoint>keypoints1={};
				std::vector<cv::KeyPoint>keypoints2={};

				std::cout<<"Match "<<Match.SearchForInitialization(FirstFrame,fr)<<" points.\n";


				if (Match.SearchForInitialization(FirstFrame, fr) < 10)
				{
					delete Initializer;
					Initializer = nullptr;
					continue;
				}
				SecondFrame = fr;

				//for test.
				/*int num=0;
				for (std::map<int,int>::iterator MatchedPair=Match.MatchedPoints.begin();MatchedPair!=Match.MatchedPoints.end();++MatchedPair)
				{
					cv::DMatch dm;
					dm.imgIdx=0;
					dm.queryIdx=dm.trainIdx=num++;
					match1to2.push_back(dm);
					keypoints1.push_back(FirstFrame->keypoints[MatchedPair->first]);
					keypoints2.push_back(SecondFrame->keypoints[MatchedPair->second]);
				}
				cv::Mat out;
				cv::drawMatches(FirstFrame->image, keypoints1, SecondFrame->image, keypoints2, match1to2, out);
				imshow("matches", out);
				cv::waitKey(0);*/

				//KeyFrame *debug_kf = static_cast<KeyFrame *>(currFrame);
				//assert(debug_kf->sbiImg.size().height == 30 && debug_kf->sbiImg.size().width == 40);
				cv::Mat R21;
				cv::Mat t21;
				std::vector<cv::Point3d> vP3D;
				std::vector<bool> vbTriangulated;
				if (!Initializer->Initialize(*SecondFrame, Match.MatchedPoints, R21, t21, vP3D, vbTriangulated))
				{
					std::cout << "Failed to Initialize.\n\n";
				}
				else
				{
					std::cout << "System Initialized !\n\n";
					// store mTcw of keyframes
					std::cout<<"R = "<<R21<<std::endl;
					std::cout<<"t = "<<t21<<std::endl;
					SecondFrame->mTcw.colRange(0,3).rowRange(0,3)=R21;
					SecondFrame->mTcw.colRange(0,3).row(3)=t21;

					// store points in keyframe
					Map *map = Map::getInstance();
					std::map<cv::KeyPoint*, cv::KeyPoint*> mapData;
					for (std::map<int,int>::iterator MatchedPair=Match.MatchedPoints.begin();MatchedPair!=Match.MatchedPoints.end();++MatchedPair)
					{
						int vbPointNum=0;

						mapData.insert(std::make_pair(&SecondFrame->keypoints[MatchedPair->second], &FirstFrame->keypoints[MatchedPair->first]));
						if (vbTriangulated[vbPointNum])
						{
							MapPoint *mp = new MapPoint;
							mp->Tw(0) = vP3D[vbPointNum].x;
							mp->Tw(1) = vP3D[vbPointNum].y;
							mp->Tw(2) = vP3D[vbPointNum].z;
							mp->allObservation.insert(
									std::make_pair(static_cast<KeyFrame *>(FirstFrame), &FirstFrame->keypoints[MatchedPair->first]));
							mp->allObservation.insert(std::make_pair(static_cast<KeyFrame *>(SecondFrame),
																	 &SecondFrame->keypoints[MatchedPair->second]));
							FirstFrame->mappoints[vbPointNum] = mp;
							SecondFrame->mappoints[MatchedPair->second] = mp;
							map->allMapPoint.push_back(mp);

							//for test
							file<<std::setprecision(9)<<vP3D[vbPointNum].x<<" "<<vP3D[vbPointNum].y<<" "<<vP3D[vbPointNum].z<<std::endl;

						}
						++vbPointNum;
					}
					SecondFrame->matchedGroup.insert(std::make_pair(static_cast<KeyFrame*>(FirstFrame), mapData));
					map->allKeyFrame.push_back(static_cast<KeyFrame *>(FirstFrame));
					map->allKeyFrame.push_back(static_cast<KeyFrame *>(SecondFrame));
					lastFrame = SecondFrame;
					mState = OK;
					//drawMatch(FirstFrame, SecondFrame, mapData);
					//void findCorrespondenceByKp(Frame* lastFrame, Frame* currFrame);
					//findCorrespondenceByKp(lastFrame, currFrame);
					//for test
					file.close();
					/*cv::Mat out;
					cv::drawMatches(FirstFrame->image, keypoints1, SecondFrame->image, keypoints2, match1to2, out);
					imshow("matches", out);
					cv::waitKey(0);*/

					continue;
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
				findCorrespondenceByKp(lastFrame, currFrame);
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




				/*int tmpd = 0;
				 std::vector<cv::DMatch> dm;
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
				imshow("matches", out);
				cv::waitKey(0);*/

				//getchar();
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
