#include <opencv2/opencv.hpp>
#include "tracking.h"
#include <Eigen/Dense>
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
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/surface_matching.hpp>
#include <opencv2/surface_matching/ppf_helpers.hpp>

Tracking::Tracking():Initializer(static_cast<Initialization*>(NULL)){logfile.open("logfile.txt");};
enum InitializeMethod
{
	SVO=1,
	VINS_Mono=2,
	DEPTH=3,
};


// depth1:scene depth;depth2:model depth, compute projection from model to scene;
cv::Mat getPosByICP(cv::Mat_<double> depth1, cv::Mat_<double> depth2, cv::Mat_<double> mK)
;

void Tracking::Run(std::string pathtoData)
{
	typedef Eigen::Matrix<double, 7, 1, Eigen::ColMajor> Vector7d;
	void drawMatch(Frame* lastFrame, Frame* currFrame, std::map<int, int> matches);
	void testProjection(Frame* lastFrame, Frame* currFrame, Eigen::Matrix3d h = Eigen::Matrix3d::Zero());
	void findCorrespondenceByKp(Frame* lastFrame, Frame* currFrame, std::map<int,int>& matches);
	void drawProjection(Frame* lastFrame, Frame* currFrame, std::map<int,int> matches);
	cv::Mat generateImage(cv::Mat image);

	//Load Images.
	localMap = new LocalMap(this);
	std::vector<std::string> vstrImageFilenames, vstrImageFilenamesD;
	std::vector<double> vTimestamps, vTimestampsD;
	std::string strFile=pathtoData+"/rgb.txt";
	std::string depthFile = pathtoData+"/depth.txt";
	LoadImages(strFile,vstrImageFilenames,vTimestamps);
	LoadImages(depthFile, vstrImageFilenamesD, vTimestampsD);

	int nImages=vstrImageFilenames.size();
	Optimizer::mK = this->mK;
	for(int ni=0;ni<nImages;ni++)
	{
		InitializeMethod Method = DEPTH;
		cv::Mat im=cv::imread(pathtoData+"/"+vstrImageFilenames[ni],cv::IMREAD_GRAYSCALE);
		cv::Mat depthIm = cv::imread(pathtoData+"/"+vstrImageFilenamesD[ni], cv::IMREAD_ANYDEPTH);
		double tframe=vTimestamps[ni];
		Frame* fr = new Frame();
		cv::resize(im, fr->sbiImg, cv::Size(40, 30));
		cv::GaussianBlur(fr->sbiImg, fr->sbiImg, cv::Size(0, 0), 0.75);
		fr->image = im;
		fr->depthImg = depthIm*1./5000;
		fr->id = ni;
		fr->timestamp = tframe;
		lastFrame = currFrame;
		currFrame = fr;

		//Initialize.
		if(mState==NOT_INITIALIZED)
		{
			if (Method == DEPTH)
			{
				if (fabs(fr->timestamp-1305031107.37541)<0.001)
				{
					cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(500);
					sift->detect(currFrame->image, currFrame->keypoints);
					fr->mTcw = cv::Mat::ones(4,4,CV_64FC1);
					FirstFrame = fr;
				}
				if (fabs(fr->timestamp-1305031107.91154)<0.001)
				{
					cv::Mat_<double> mmK(mK);
					SecondFrame = fr;
					cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(500);
					sift->detect(currFrame->image, currFrame->keypoints);

					cv::Mat desc1, desc2;
					sift->compute(FirstFrame->image, FirstFrame->keypoints, desc1);
					sift->compute(SecondFrame->image, SecondFrame->keypoints, desc2);

					fr->mTcw = getPosByICP(FirstFrame->depthImg, SecondFrame->depthImg, mmK);

					cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
					std::vector<cv::DMatch> matches;
					matcher->match(desc1, desc2, matches);
					std::map<int, int> tmpm;
					for (int i = 0;i<matches.size();i++)
					{
						tmpm.insert(std::make_pair(matches[i].trainIdx, matches[i].queryIdx));
					}
					SecondFrame->matchedGroup.insert(std::make_pair(static_cast<KeyFrame*>(FirstFrame), tmpm));


					cv::Mat_<double> depth = FirstFrame->depthImg;
					cv::Mat Twc = fr->mTcw.inv();
					FirstFrame->mappoints.resize(FirstFrame->keypoints.size());
					std::fill(FirstFrame->mappoints.begin(), FirstFrame->mappoints.end(), nullptr);
					SecondFrame->mappoints.resize(SecondFrame->keypoints.size());
					std::fill(SecondFrame->mappoints.begin(), SecondFrame->mappoints.end(), nullptr);
					for (int i = 0;i<matches.size();i++)
					{
						MapPoint* mp = new MapPoint;
						cv::Mat_<double> loc(3,1);
						cv::Point2f pt = FirstFrame->keypoints[matches[i].queryIdx].pt;
						loc(0) = (pt.x-mmK(0,2))*depth(pt.y, pt.x)/mmK(0,0);
						loc(1) = (pt.y-mmK(1,2))*depth(pt.y, pt.x)/mmK(1,1);
						loc(2) = depth(pt.y, pt.x);
						if (loc(2)==0.0)
						{
							delete mp;
							continue;
						}
						mp->Tw = Twc.colRange(0,3).rowRange(0,3)*loc + Twc.col(3).rowRange(0,3);
						mp->allObservation.insert(std::make_pair(FirstFrame, &FirstFrame->keypoints[matches[i].queryIdx]));
						mp->allObservation.insert(std::make_pair(SecondFrame, &SecondFrame->keypoints[matches[i].trainIdx]));

						FirstFrame->mappoints[matches[i].queryIdx] = mp;
						SecondFrame->mappoints[matches[i].trainIdx] = mp;
					}
					Map* map = Map::getInstance();
					map->addKeyFrame(static_cast<KeyFrame*>(FirstFrame));
					map->addKeyFrame(static_cast<KeyFrame*>(SecondFrame));
					mState = OK;
				}
				continue;
			}
			if(Method==SVO||VINS_Mono)
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

			int grid_n_cols_=std::ceil(static_cast<double>(fr->image.cols)/cell_size);
			int grid_n_rows_=std::ceil(static_cast<double>(fr->image.rows)/cell_size);
			std::vector<float > score_at_grid(grid_n_cols_*grid_n_rows_,0);
			std::vector<cv::KeyPoint>best_KeyPoint_at_grid(grid_n_cols_*grid_n_rows_);

			if (!Initializer)
			{
				//Detect Fast Feature with Shi-Tomasi-Score and grid.
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
				if(Method==VINS_Mono)
				{
					VINS_FramesInWindow.clear();
					VINS_FramesInWindow.reserve(10);
					VINS_FramesInWindow.push_back(FirstFrame);
				}
			}
			else
			{
				std::cout << ni << "th(" << std::setprecision(16) << tframe
						  << ") image is selected as SecondFrame!\n";
				if(Method==SVO)
				{
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
					std::vector<double >disparities={};
					for (size_t i = 0; First_keypoint_it!=FirstFrame->keypoints.end(); ++i)
					{
						if(!status[i])
						{
							First_keypoint_it=FirstFrame->keypoints.erase(First_keypoint_it);
							matched_point_LK_it=matched_point_LK.erase(matched_point_LK_it);
							continue;
						}
						disparities.push_back(Eigen::Vector2d(First_keypoint_it->pt.x-matched_point_LK_it->x,First_keypoint_it->pt.y-matched_point_LK_it->y).norm());
						++First_keypoint_it;
						++matched_point_LK_it;
					}
					std::cout << "Match " << matched_point_LK.size() << " points. \n";

					if (matched_point_LK.size() < 150)
					{
						delete Initializer;
						Initializer = static_cast<Initialization*>(NULL);
						std::cout<<"Failed to Initialize: few matched points.";
						continue;
					}
					std::sort(disparities.begin(),disparities.end());
					if (disparities[(disparities.size()-1)/2]<50.0)
					{
						continue;
					}
					//for test: draw matches when initializing with LK.
					void drawMatchInitial(Frame* lastFrame, Frame* currFrame, std::vector<cv::Point2f>matches);
					drawMatchInitial(FirstFrame,fr,matched_point_LK);

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
						void testProjection(Frame *lastFrame, Frame *currFrame, Eigen::Matrix3d h = Eigen::Matrix3d::Zero());
						testProjection(FirstFrame, SecondFrame);
						Eigen::Matrix3d deltaH;
						cv::cv2eigen(Initializer->DeltaH,deltaH);
						testProjection(FirstFrame,SecondFrame,deltaH);

						// store mTcw of keyframes
						//TODO
						std::cout<<"R = "<<R21<<std::endl;
						std::cout<<"t = "<<t21<<std::endl;
						SecondFrame->mTcw.colRange(0,3).rowRange(0,3)=R21;
						SecondFrame->mTcw.colRange(0,3).row(3)=t21;

						// store points in keyframe
						FirstFrame->mappoints.resize(FirstFrame->keypoints.size());
						std::fill(FirstFrame->mappoints.begin(), FirstFrame->mappoints.end(), nullptr);
						SecondFrame->mappoints.resize(SecondFrame->keypoints.size());
						std::fill(SecondFrame->mappoints.begin(), SecondFrame->mappoints.end(), nullptr);
						Map *map = Map::getInstance();
						std::map<int,int> mapData;
						int vbPointNum = 0;
						for (std::map<int,int>::iterator MatchedPair=MatchedPoints.begin();MatchedPair!=MatchedPoints.end();++MatchedPair)
						{
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
				else if(Method==VINS_Mono)
				{
					bool SuccessFlag= false;
					bool SuccessMatchingFlag= false;
					for (int frs = 0; frs < VINS_FramesInWindow.size(); ++frs)
					{
						//Match by LK.
						std::vector<cv::Point2f> first_frame_point, matched_point_LK;
						for (int i = 0; i < VINS_FramesInWindow[frs]->keypoints.size(); ++i)
						{
							first_frame_point.push_back(VINS_FramesInWindow[frs]->keypoints[i].pt);
							matched_point_LK.push_back(VINS_FramesInWindow[frs]->keypoints[i].pt);
						}
						std::vector<uchar> status;
						std::vector<float> error;
						const int klt_win_size = 30;
						const int klt_max_iter = 30;
						const double klt_eps = 0.001;
						cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, klt_max_iter,
												  klt_eps);
						cv::calcOpticalFlowPyrLK(VINS_FramesInWindow[frs]->ImgPyrForInitial[0], fr->ImgPyrForInitial[0],
												 first_frame_point, matched_point_LK, status, error,
												 cv::Size2i(klt_win_size, klt_win_size), 4, termcrit);
						std::vector<cv::KeyPoint>::iterator First_keypoint_it = VINS_FramesInWindow[frs]->keypoints.begin();
						std::vector<cv::Point2f>::iterator matched_point_LK_it = matched_point_LK.begin();
						std::vector<double> disparities = {};
						for (size_t i = 0; First_keypoint_it != VINS_FramesInWindow[frs]->keypoints.end(); ++i)
						{
							if (!status[i])
							{
								First_keypoint_it = VINS_FramesInWindow[frs]->keypoints.erase(First_keypoint_it);
								matched_point_LK_it = matched_point_LK.erase(matched_point_LK_it);
								continue;
							}
							disparities.push_back(Eigen::Vector2d(First_keypoint_it->pt.x - matched_point_LK_it->x,
																  First_keypoint_it->pt.y -
																  matched_point_LK_it->y).norm());
							++First_keypoint_it;
							++matched_point_LK_it;
						}
						//for test: draw matches when initializing with LK.
						void drawMatchInitial(Frame* lastFrame, Frame* currFrame, std::vector<cv::Point2f>matches);
						drawMatchInitial(VINS_FramesInWindow[frs],fr,matched_point_LK);

						std::cout << "Match " << matched_point_LK.size() << " points. \n";

						if (matched_point_LK.size() < 150)
						{
							std::cout << "Failed to Initialize: few matched points.\n";
							continue;
						} else SuccessMatchingFlag = true;
						cv::KeyPoint::convert(matched_point_LK, fr->keypoints);

						// detect extra features to retain points' size.
						if (matched_point_LK.size() < grid_n_cols_ * grid_n_rows_)
						{
							std::vector<cv::KeyPoint> ExtraKeyPoint;
							cv::FAST(fr->image, ExtraKeyPoint, 20, true);
							cv::KeyPointsFilter::retainBest(ExtraKeyPoint,
															grid_n_cols_ * grid_n_rows_ - matched_point_LK.size());
							fr->keypoints.insert(fr->keypoints.end(), ExtraKeyPoint.begin(), ExtraKeyPoint.end());
						}

						std::sort(disparities.begin(), disparities.end());
						if (disparities[(disparities.size() - 1) / 2] < 30.0)
						{
							std::cout<<"Failed to Initialize: disparity points.\n";
							continue;
						}
						SecondFrame = fr;

						cv::Mat R21;
						cv::Mat t21;
						std::vector<cv::Point3d> vP3D;
						std::vector<bool> vbTriangulated;
						std::map<int, int> MatchedPoints;
						for (int j = 0; j < matched_point_LK.size(); ++j)
						{
							MatchedPoints.insert(std::make_pair(j, j));
						}
						if (!Initializer->Initialize(*SecondFrame, MatchedPoints, R21, t21, vP3D,
													 vbTriangulated))
						{
							std::cout << "Failed to Initialize.\n\n";
						} else
						{
							SuccessFlag=true;
							std::cout << "System Initialized !\n\n";
							FirstFrame=VINS_FramesInWindow[frs];
							//for test: compare the precision of the H between using sbi and using opencv::computeH.
							void testProjection(Frame *lastFrame, Frame *currFrame, Eigen::Matrix3d h = Eigen::Matrix3d::Zero());
							testProjection(FirstFrame, SecondFrame);
							Eigen::Matrix3d deltaH;
							cv::cv2eigen(Initializer->DeltaH,deltaH);
							testProjection(FirstFrame,SecondFrame,deltaH);

							// store mTcw of keyframes
							//TODO
							std::cout << "R = " << R21 << std::endl;
							std::cout << "t = " << t21 << std::endl;
							SecondFrame->mTcw.colRange(0, 3).rowRange(0, 3) = R21;
							SecondFrame->mTcw.colRange(0, 3).row(3) = t21;

							// store points in keyframe
							FirstFrame->mappoints.reserve(FirstFrame->keypoints.size());
							std::fill(FirstFrame->mappoints.begin(), FirstFrame->mappoints.end(), nullptr);
							SecondFrame->mappoints.reserve(SecondFrame->keypoints.size());
							std::fill(SecondFrame->mappoints.begin(), SecondFrame->mappoints.end(), nullptr);
							Map *map = Map::getInstance();
							std::map<int, int> mapData;
							for (std::map<int, int>::iterator MatchedPair = MatchedPoints.begin();
								 MatchedPair != MatchedPoints.end(); ++MatchedPair)
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

							SecondFrame->matchedGroup.insert(
									std::make_pair(static_cast<KeyFrame *>(FirstFrame), mapData));
							map->allKeyFrame.push_back(static_cast<KeyFrame *>(FirstFrame));
							map->allKeyFrame.push_back(static_cast<KeyFrame *>(SecondFrame));
							lastFrame = SecondFrame;
							mState = OK;

							break;
						}
					}
					if (SuccessMatchingFlag== false)
					{
						delete Initializer;
						Initializer = static_cast<Initialization*>(NULL);
					}
					if (SuccessFlag== false)
					{
						if (VINS_FramesInWindow.size()==10)
							VINS_FramesInWindow.erase(VINS_FramesInWindow.begin());
						VINS_FramesInWindow.push_back(fr);
					}
				}
			}
		}
			if(mState==OK)
			{
				std::cout<<"Tracking..."<<std::endl;
				//cv::FAST(currFrame->image, currFrame->keypoints, 20);
				cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift = cv::xfeatures2d::SiftFeatureDetector::create(500);
				sift->detect(currFrame->image, currFrame->keypoints);
				Map* map = Map::getInstance();
				auto a = Optimizer::ComputeHGlobalSBI(lastFrame, currFrame);
				std::map<int,int>matches;
				std::vector<KeyFrame*> kfs = SearchTopOverlapping();
				std::map<KeyFrame*, Eigen::Matrix3d> khs;
				for (int i = 0;i<kfs.size();i++)
				{
					Eigen::Matrix3d b = Optimizer::ComputeHGlobalKF(kfs[i], lastFrame);
					Eigen::Matrix3d c = a*b;
					khs.insert(std::make_pair(kfs[i], c));
  					//testProjection(kfs[i], currFrame, c);
				}
				if (std::find(map->allKeyFrame.begin(), map->allKeyFrame.end(), lastFrame)!=map->allKeyFrame.end())
				{
					khs.insert(std::make_pair(static_cast<KeyFrame*>(lastFrame), a));
				}
				currFrame->keyFrameSet = khs;
				Matcher match;
				int matchKfNum = match.SearchMatchByGlobal(currFrame, khs);
				std::cout<<"Matched points with keyframe:"<<matchKfNum<<"\n";
				if (matchKfNum<100)
				{
					//std::cout << "Matched points with local homo:" << match.SearchMatchByLocal(currFrame, kfs) << "\n";
				}
				for (auto iter = currFrame->matchedGroup.begin();iter!=currFrame->matchedGroup.end();iter++)
				{
					//if (std::abs(currFrame->timestamp-1305031108.111378)<0.01)
					{
						//drawMatch(iter->first, currFrame, iter->second);
						//testProjection(iter->first, currFrame, currFrame->keyFrameSet[iter->first]);
						//cv::waitKey(0);
					}
				}

				currFrame->mappoints.resize(currFrame->keypoints.size());
				std::fill(currFrame->mappoints.begin(),currFrame->mappoints.end(),nullptr);
				for (auto it = currFrame->matchedGroup.begin();it!=currFrame->matchedGroup.end();it++)
				{
					Frame* kf = it->first;
					for (auto it2 = it->second.begin();it2!=it->second.end();it2++)
					{
						currFrame->mappoints[it2->first] = kf->mappoints[it2->second];
						if (kf->mappoints[it2->second]!= nullptr)
						{
							kf->mappoints[it2->second]->allObservation.insert(std::make_pair(currFrame, &currFrame->keypoints[it2->first]));
						}
					}
				}
				// Pose estimation
				//auto Tcw = Optimizer::PoseEstimation(currFrame);
				std::vector<cv::Point3f> p3d;
				std::vector<cv::Point2f> p2d;
				for (int i = 0;i<currFrame->mappoints.size();i++)
				{
					MapPoint* mp = currFrame->mappoints[i];
					if (mp!= nullptr)
					{
						p3d.push_back(cv::Point3f(mp->Tw(0), mp->Tw(1), mp->Tw(2)));
						p2d.push_back(currFrame->keypoints[i].pt);
					}
				}
				if (p3d.size()<=8)
				{
					logfile.close();
					exit(0);
				}
				cv::Mat rvec, tvec;
				cv::solvePnP(p3d, p2d, mK, cv::noArray(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
				cv::Mat rot;
				cv::Rodrigues(rvec, rot);
				rot.copyTo(currFrame->mTcw.rowRange(0,3).colRange(0,3));
				tvec.copyTo(currFrame->mTcw.col(3).rowRange(0,3));
				currFrame->mTcw.row(3).colRange(0,3) = cv::Mat::zeros(1, 3, CV_64FC1);
				currFrame->mTcw.at<double>(3,3) = 1;
				currFrame->mTcw = currFrame->mTcw.inv();
				cv::Mat_<double> Tcw = currFrame->mTcw;
				logfile<<std::setprecision(16)<<currFrame->timestamp<<" "<<Tcw(0)<<" "<<Tcw(1)<<" "<<Tcw(2)<<"\n";
				// Mappoint creation
				//g2o::SE3Quat t;
				//t.fromVector(Tcw);
				//logfile<<std::setprecision(16)<<currFrame->timestamp<<" "<<Tcw[0]<<" "<<Tcw[1]<<" "<<Tcw[2]<<"\n";
				//cv::eigen2cv(t.to_homogeneous_matrix(), currFrame->mTcw);
				for (auto it = currFrame->matchedGroup.begin();it!=currFrame->matchedGroup.end();it++)
				{
					Frame* kf = it->first;
					for (auto it2 = it->second.begin();it2!=it->second.end();it2++)
					{
						currFrame->mappoints[it2->first] = kf->mappoints[it2->second];
						if (kf->mappoints[it2->second]== nullptr)
						{
							double par = localMap->ComputeParallax(kf->keypoints[it2->second], currFrame->keypoints[it2->first],
																   static_cast<KeyFrame*>(kf), currFrame);
							if (par>1)
							{
								kf->mappoints[it2->second] = localMap->Triangulation(kf->keypoints[it2->second], currFrame->keypoints[it2->first],
																					 static_cast<KeyFrame*>(kf), currFrame);

							}
							else
							{
								kf->mappoints[it2->second] = localMap->GetPositionByOptimization(kf->keypoints[it2->second], currFrame->keypoints[it2->first],
																								 static_cast<KeyFrame*>(kf), currFrame);
							}
						}
						kf->mappoints[it2->second]->allObservation.insert(std::make_pair(currFrame, &currFrame->keypoints[it2->first]));
					}
				}
				if (DecideKeyFrame(currFrame, matchKfNum))
				{
					std::cout<<"new keyframe selected"<<"\n";
					map->addKeyFrame(static_cast<KeyFrame*>(currFrame));
				}
			}
		}
	//Local Mapping.




	}

// search for x most overlapping keyframe with last frame.
std::vector<KeyFrame*> Tracking::SearchTopOverlapping()
{
	const int MAX_KEYFRAME_COUNT = 5;
	const int MIN_MATCH_COUNT = 20;
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
		if (max_v<MIN_MATCH_COUNT)
		{
			break;
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

bool Tracking::DecideKeyFrame(const Frame* currFrame, int count)
{
	if (this->mState!=Tracking::OK)
	{
		return false;
	}
	Map* map = Map::getInstance();
	const unsigned long lastFrameId = map->allKeyFrame.back()->id;
	if (currFrame->id-lastFrameId<=minimalKeyFrameInterval)
	{
		return false;
	}
	Eigen::Matrix4d t1, t2;
	cv::Mat a = map->allKeyFrame.back()->mTcw;
	cv::cv2eigen(a, t1);
	//TODO: suitable threshold for comparison
	cv::cv2eigen(currFrame->mTcw, t2);
	return (t1.col(3).hnormalized()-t2.col(3).hnormalized()).squaredNorm()>0.1||count<50;
}

// depth1:scene depth;depth2:model depth, compute projection from model to scene;
cv::Mat getPosByICP(cv::Mat_<double> depth1, cv::Mat_<double> depth2, cv::Mat_<double> mK)
{
	int pnumScene = cv::countNonZero(depth1);
	int pnumModel = cv::countNonZero(depth2);
	cv::Mat_<float> scene = cv::Mat(pnumScene, 3, CV_32FC1);
	cv::Mat_<float> model = cv::Mat(pnumModel, 3, CV_32FC1);
	int count = 0;
	for (int i = 0; i < depth1.rows; i++)
	{
		for (int j = 0; j < depth1.cols; j++)
		{
			if (depth1(i, j) == 0)
			{
				continue;
			}
			scene(count, 0) = (j - mK(0, 2)) * depth1(i, j) / mK(0, 0);
			scene(count, 1) = (i - mK(1, 2)) * depth1(i, j) / mK(1, 1);
			scene(count, 2) = depth1(i, j);
			count++;
		}
	}

	count = 0;
	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			if (depth2(i, j) == 0)
			{
				scene(count, 0) = (j - mK(0, 2)) * depth2(i, j) / mK(0, 0);
				scene(count, 1) = (i - mK(1, 2)) * depth2(i, j) / mK(1, 1);
				scene(count, 2) = depth2(i, j);
				count++;
			}
		}
	}

	cv::Mat_<float> sceneNor(pnumScene, 6, CV_32FC1);
	cv::Mat_<float> modelNor(pnumModel, 6, CV_32FC1);
	cv::ppf_match_3d::computeNormalsPC3d(scene, sceneNor, 8, false, nullptr);
	cv::ppf_match_3d::computeNormalsPC3d(model, modelNor, 8, false, nullptr);
	int min = pnumModel > pnumScene ? pnumScene : pnumModel;
	sceneNor = sceneNor.rowRange(0, min);
	modelNor = modelNor.rowRange(0, min);
	std::cout << modelNor << "\n";
	std::cout << sceneNor << "\n";
	cv::ppf_match_3d::ICP icpAlg;
	double error;
	double pose[16];
	for (int i = 0; i < 16; i++)
	{
		pose[i] = 0;
	}
	pose[0] = pose[5] = pose[10] = pose[15] = 1;
	icpAlg.registerModelToScene(modelNor, sceneNor, error, pose);
	std::cout<<"ICP error in initialization:"<<error<<"\n";
	cv::Mat result(4, 4, CV_64FC1, pose);
	return result;
}
