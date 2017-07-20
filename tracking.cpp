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
#include <sophus/se3.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>

Tracking::Tracking():Initializer(static_cast<Initialization*>(NULL)){logfile.open("logfile.txt");csvlog.open("log.csv");};
enum InitializeMethod
{
	SVO=1,
	VINS_Mono=2,
	DEPTH=3,
};


// depth1:scene depth;depth2:model depth, compute projection from model to scene;
cv::Mat getPosByICP(cv::Mat_<double> depth1, cv::Mat_<double> depth2, cv::Mat_<double> mK)
;

MapPoint* getMpByDepth(cv::Mat_<double> depthImg, cv::Point2f pt, cv::Mat_<double> mK);

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

	unsigned long nImages=vstrImageFilenames.size();
	Optimizer::mK = this->mK;
	for(unsigned long ni=0;ni<nImages;ni++)
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
				//if (fabs(fr->timestamp-1305031102.1753)<0.001)
				if (FirstFrame == nullptr)
				{
					//cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
					fr->mTcw = cv::Mat::ones(4,4,CV_64FC1);
					FirstFrame = fr;
					Sophus::SE3 tcw = Sophus::SE3();
					Eigen::Vector4d rvec = tcw.inverse().so3().unit_quaternion().coeffs();
					Eigen::Vector3d tvec = tcw.inverse().translation();
					logfile<<std::setprecision(16)<<currFrame->timestamp<<" "<<tvec[0]<<" "<<tvec[1]<<" "<<tvec[2]<<" "<<
						   rvec[0]<<" "<<rvec[1]<<" "<<rvec[2]<<" "<<rvec[3]<<"\n";
					continue;
				}
				//if (fabs(fr->timestamp-1305031102.2753)<0.001)
				if (FirstFrame!= nullptr&&SecondFrame== nullptr)
				{
					cv::Mat_<double> mmK(mK);
					SecondFrame = fr;
					Matcher match;
					match.orb_->detect(FirstFrame->image, FirstFrame->keypoints);
					match.orb_->detect(SecondFrame->image, SecondFrame->keypoints);
					std::ofstream f;
					f.open("keypoints.txt");
					for (int i = 0;i<SecondFrame->keypoints.size();i++)
					{
						f<<SecondFrame->keypoints[i].pt.x<<","<<SecondFrame->keypoints[i].pt.y<<","<<SecondFrame->keypoints[i].angle<<"\n";
					}
					f.flush();
					f.close();
					std::vector<KeyFrame*> kfs;
					kfs.push_back(static_cast<KeyFrame*>(FirstFrame));
					match.SearchMatchBySIFT(SecondFrame, kfs);

					//fr->mTcw = getPosByICP(FirstFrame->depthImg, SecondFrame->depthImg, mmK);
					//fr->mTcw.convertTo(fr->mTcw, CV_64FC1);


					cv::Mat_<double> depth = FirstFrame->depthImg;
					cv::Mat Twc = fr->mTcw.inv();
					FirstFrame->mappoints.resize(FirstFrame->keypoints.size());
					std::fill(FirstFrame->mappoints.begin(), FirstFrame->mappoints.end(), nullptr);
					SecondFrame->mappoints.resize(SecondFrame->keypoints.size());
					std::fill(SecondFrame->mappoints.begin(), SecondFrame->mappoints.end(), nullptr);
					auto matches = SecondFrame->matchedGroup[static_cast<KeyFrame*>(FirstFrame)];
					for (auto a = matches.begin();a!=matches.end();a++)
					{
						MapPoint* mp = getMpByDepth(depth, FirstFrame->keypoints[a->second].pt, mmK);
						if (mp== nullptr)
						{
							continue;
						}
						mp->allObservation.insert(std::make_pair(FirstFrame, &FirstFrame->keypoints[a->second]));
						mp->allObservation.insert(std::make_pair(SecondFrame, &SecondFrame->keypoints[a->first]));

						FirstFrame->mappoints[a->second] = mp;
						SecondFrame->mappoints[a->first] = mp;
					}

					Sophus::SE3	tcw = Optimizer::PoseEstimationPnP(currFrame, csvlog);
					Eigen::Vector4d rvec = tcw.inverse().so3().unit_quaternion().coeffs();
					Eigen::Vector3d tvec = tcw.inverse().translation();
					logfile<<std::setprecision(16)<<currFrame->timestamp<<" "<<tvec[0]<<" "<<tvec[1]<<" "<<tvec[2]<<" "<<
						   rvec[0]<<" "<<rvec[1]<<" "<<rvec[2]<<" "<<rvec[3]<<"\n";

					cv::eigen2cv(tcw.matrix(), currFrame->mTcw);

					Map* map = Map::getInstance();
					map->addKeyFrame(static_cast<KeyFrame*>(FirstFrame));
					map->addKeyFrame(static_cast<KeyFrame*>(SecondFrame));

					mState = OK;
				}
				continue;
			}

		}
			if(mState==OK)
			{
				std::cout<<"Tracking..."<<std::endl;
				//cv::FAST(currFrame->image, currFrame->keypoints, 20);
				//cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> sift = cv::xfeatures2d::SiftFeatureDetector::create();
				Matcher match;
				match.orb_->detect(currFrame->image, currFrame->keypoints);
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
				}
				if (std::find(map->allKeyFrame.begin(), map->allKeyFrame.end(), lastFrame)!=map->allKeyFrame.end())
				{
					kfs.push_back(static_cast<KeyFrame*>(lastFrame));
					khs.insert(std::make_pair(static_cast<KeyFrame*>(lastFrame), a));
				}
				currFrame->keyFrameSet = khs;
				int matchKfNum = match.SearchMatchByGlobal(currFrame, khs);
				//int matchKfNum = match.SearchMatchBySIFT(currFrame, kfs);
				csvlog<<std::setprecision(16)<<currFrame->timestamp<<","<<0<<","<<matchKfNum<<",";
				std::cout<<"Matched points with keyframe:"<<matchKfNum<<"\n";
				if (matchKfNum<100)
				{
					//std::cout << "Matched points with local homo:" << match.SearchMatchByLocal(currFrame, kfs) << "\n";
				}
				for (auto iter = currFrame->matchedGroup.begin();iter!=currFrame->matchedGroup.end();iter++)
				{
					//if (std::abs(currFrame->timestamp-1305031108.111378)<0.01)
					if (ni>=10)
					{
						//drawMatch(iter->first, currFrame, iter->second);
						//testProjection(iter->first, currFrame, currFrame->keyFrameSet[iter->first]);
						//std::cout<<"the "<<ni<<" th frame"<<"\n";
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
						if (kf->mappoints[it2->second]!= nullptr)
						{
							kf->mappoints[it2->second]->allObservation.insert(std::make_pair(currFrame, &currFrame->keypoints[it2->first]));
							currFrame->mappoints[it2->first] = kf->mappoints[it2->second];
						}
					}
				}

				// Pose estimation
				auto Tcwd = Optimizer::PoseEstimation(currFrame, csvlog);
				Eigen::Quaterniond quat;
				quat.coeffs() = Tcwd.tail<4>();
				Sophus::SE3 tcw;
				tcw.setQuaternion(quat);
				tcw.translation() = Tcwd.head<3>();
				//Sophus::SE3 tcw = Optimizer::PoseEstimationPnP(currFrame, csvlog);
				if (tcw.translation() == Eigen::Vector3d(0, 0, 0))
				{
					logfile.close();
					csvlog.close();
					std::cout<<"no log data"<<"\n";
					exit(0);
				}
				Eigen::Vector4d rvec = tcw.inverse().so3().unit_quaternion().coeffs();
				Eigen::Vector3d tvec = tcw.inverse().translation();
				logfile<<std::setprecision(16)<<currFrame->timestamp<<" "<<tvec[0]<<" "<<tvec[1]<<" "<<tvec[2]<<" "<<
					   rvec[0]<<" "<<rvec[1]<<" "<<rvec[2]<<" "<<rvec[3]<<"\n";
				// Mappoint creation
				//g2o::SE3Quat t;
				//t.fromVector(Tcw);
				//logfile<<std::setprecision(16)<<currFrame->timestamp<<" "<<Tcw[0]<<" "<<Tcw[1]<<" "<<Tcw[2]<<"\n";
				//cv::eigen2cv(t.to_homogeneous_matrix(), currFrame->mTcw);
				/*
				for (auto it = currFrame->matchedGroup.begin();it!=currFrame->matchedGroup.end();it++)
				{
					Frame* kf = it->first;
					for (auto it2 = it->second.begin();it2!=it->second.end();it2++)
					{
						if (kf->mappoints[it2->second]== nullptr)
						{
							kf->mappoints[it2->second] = getMpByDepth(kf->depthImg, kf->keypoints[it2->second].pt, mK, kf->mTcw);
							MapPoint* mp = getMpByDepth(currFrame->depthImg, currFrame->keypoints[it2->first].pt, mK, currFrame->mTcw);
							if (mp!= nullptr)
							{
								kf->mappoints[it2->second] = mp;
							}
						}
						currFrame->mappoints[it2->first] = kf->mappoints[it2->second];
						if (kf->mappoints[it2->second]!= nullptr)
						{
							kf->mappoints[it2->second]->allObservation.insert(std::make_pair(currFrame, &currFrame->keypoints[it2->first]));
						}
					}
				}
				if (DecideKeyFrame(currFrame, matchKfNum))
				{
					std::cout<<"new keyframe selected"<<"\n";
					map->addKeyFrame(static_cast<KeyFrame*>(currFrame));
				}
				 */
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
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudScene(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudModel(new pcl::PointCloud<pcl::PointXYZ>);

	cloudScene->is_dense = false;
	cloudScene->width = depth1.cols;cloudScene->height = depth1.rows;
	cloudScene->points.resize(cloudScene->width*cloudScene->height);
	int count = 0;
	for (int i = 0; i < depth1.rows; i++)
	{
		for (int j = 0; j < depth1.cols; j++)
		{
			cloudScene->points[count].x = (j - mK(0, 2)) * depth1(i, j) / mK(0, 0);
			cloudScene->points[count].y = (i - mK(1, 2)) * depth1(i, j) / mK(1, 1);
			cloudScene->points[count].z = depth1(i, j);
			if (depth1(i,j)==0)
			{
				cloudScene->points[count].x = NAN;
				cloudScene->points[count].y = NAN;
				cloudScene->points[count].z = NAN;
			}
			count++;
		}
	}

	cloudModel->is_dense = false;
	cloudModel->width = depth2.cols;cloudModel->height = depth2.rows;
	cloudModel->points.resize(cloudModel->width*cloudModel->height);
	count = 0;
	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			cloudModel->points[count].x = (j - mK(0, 2)) * depth2(i, j) / mK(0, 0);
			cloudModel->points[count].y = (i - mK(1, 2)) * depth2(i, j) / mK(1, 1);
			cloudModel->points[count].z = depth2(i, j);
			if (depth2(i,j)==0)
			{
				cloudModel->points[count].x = NAN;
				cloudModel->points[count].y = NAN;
				cloudModel->points[count].z = NAN;
			}
			count++;
		}
	}
	std::vector<int> indexes;
	pcl::removeNaNFromPointCloud(*cloudModel, *cloudModel, indexes);
	pcl::removeNaNFromPointCloud(*cloudScene, *cloudScene, indexes);

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setEuclideanFitnessEpsilon(1e-12);
	icp.setMaximumIterations(100);
	icp.setInputSource(cloudModel);
	icp.setInputTarget(cloudScene);
	pcl::PointCloud<pcl::PointXYZ> Final;
	icp.align(Final);
	std::cout<<"has converged"<<icp.hasConverged()<<"score:"<<icp.getFitnessScore()<<"\n";
	Eigen::Matrix4f t = icp.getFinalTransformation();
	std::cout<<t<<"\n";
	cv::Mat res;
	cv::eigen2cv(t, res);
	return res;
}

MapPoint* getMpByDepth(cv::Mat_<double> depthImg, cv::Point2f pt, cv::Mat_<double> mK)
{
	cv::Mat_<double> loc(3, 1);
	double d = depthImg(pt.y, pt.x);
	if (d==0.0)
	{
		int dx[4] = {-1,0,1,0};
		int dy[4] = {0,-1,0,1};
		for ( int i=0; i<4; i++ )
		{
			d = depthImg(pt.y + dy[i], pt.x + dx[i]);
			if (d != 0)
			{
				break;
			}
		}
	}
	if (d==0.0)
	{
		return nullptr;
	}
	loc(0) = (pt.x-mK(0,2))*d/mK(0,0);
	loc(1) = (pt.y-mK(1,2))*d/mK(1,1);
	loc(2) = d;
	MapPoint* mp = new MapPoint;
	mp->Tw = loc;
	return mp;
}


