#include <opencv2/opencv.hpp>
#include "tracking.h"
#include <Eigen/Sparse>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "sl3vertex.h"
#include "sl3edge.h"
#include "Map.h"
#include "ProjectionEdge.h"
#include <opencv2/core/eigen.hpp>
#include <math.h>

#include <chrono>
#include <g2o/types/slam3d/se3quat.h>

Tracking::Tracking():Initializer(static_cast<Initialization*>(NULL)){};

void Tracking::Run(std::string pathtoData)
{

	//Load Images.
	std::vector<std::string> vstrImageFilenames;
	std::vector<double> vTimestamps;
	std::string strFile=pathtoData+"/rgb.txt";
	LoadImages(strFile,vstrImageFilenames,vTimestamps);

	int nImages=vstrImageFilenames.size();

	for(int ni=500;ni<nImages;ni++)
	{
		cv::Mat im=cv::imread(pathtoData+"/"+vstrImageFilenames[ni],cv::IMREAD_GRAYSCALE);
		double tframe=vTimestamps[ni];
		Frame* fr = new Frame();
		cv::resize(im, fr->sbiImg, cv::Size(40, 30));
		cv::GaussianBlur(fr->sbiImg, fr->sbiImg, cv::Size(0, 0), 0.75);
		fr->image = im;
		cv::FAST(fr->image, fr->keypoints, fr->Fast_threshold);
		fr->mappoints.reserve(fr->keypoints.size());
		std::fill(fr->mappoints.begin(), fr->mappoints.end(), nullptr);
		fr->id = ni;
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
					std::cout << ni << "th(" << std::setprecision(16) << tframe
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

				if (Match.SearchForInitialization(FirstFrame, fr, 15) < 100)
				{
					delete Initializer;
					Initializer = nullptr;
					continue;
				}
				SecondFrame = fr;
				KeyFrame *debug_kf = static_cast<KeyFrame *>(currFrame);
				assert(debug_kf->sbiImg.size().height == 30 && debug_kf->sbiImg.size().width == 40);
				cv::Mat R21;
				cv::Mat t21;
				std::vector<cv::Point3f> vP3D;
				std::vector<bool> vbTriangulated;
				if (!Initializer->Initialize(*SecondFrame, Match.MatchedPoints, R21, t21, vP3D, vbTriangulated))
				{
					std::cout << "Failed to Initialize.\n\n";
				}
				else
				{
					std::cout << "System Initialized !\n\n";
					// store mTcw of keyframes
					SecondFrame->mTcw.colRange(0,3).rowRange(0,3)=R21;
					SecondFrame->mTcw.colRange(0,3).row(3)=t21;

					// store points in keyframe
					Map *map = Map::getInstance();
					std::map<cv::KeyPoint*, cv::KeyPoint*> mapData;
					for (int i = 0; i < vP3D.size(); i++)
					{
						mapData.insert(std::make_pair(&SecondFrame->keypoints[Match.MatchedPoints[i]], &FirstFrame->keypoints[i]));
						if (vbTriangulated[i])
						{
							MapPoint *mp = new MapPoint;
							mp->Tw(0) = vP3D[i].x;
							mp->Tw(1) = vP3D[i].y;
							mp->Tw(2) = vP3D[i].z;
							mp->allObservation.insert(
									std::make_pair(static_cast<KeyFrame *>(FirstFrame), &FirstFrame->keypoints[i]));
							mp->allObservation.insert(std::make_pair(static_cast<KeyFrame *>(SecondFrame),
																	 &SecondFrame->keypoints[Match.MatchedPoints[i]]));
							FirstFrame->mappoints[i] = mp;
							SecondFrame->mappoints[Match.MatchedPoints[i]] = mp;
							map->allMapPoint.push_back(mp);
						}
						SecondFrame->matchedGroup.insert(std::make_pair(static_cast<KeyFrame*>(FirstFrame), mapData));
					}
					map->allKeyFrame.push_back(static_cast<KeyFrame *>(FirstFrame));
					map->allKeyFrame.push_back(static_cast<KeyFrame *>(SecondFrame));
					lastFrame = SecondFrame;
					mState = OK;
					continue;
				}
			}
		}
			if(mState==OK)
			{
				std::cout<<"Tracking..."<<std::endl;
				Map* map = Map::getInstance();
				auto kf = map->allKeyFrame.back();
				auto a = ComputeHGlobalSBI(lastFrame, currFrame);
				std::vector<KeyFrame*> kfs = SearchTopOverlapping();
				std::map<KeyFrame*, cv::Mat> khs;
				for (int i = 0;i<kfs.size();i++)
				{
					cv::Mat b = ComputeHGlobalKF(kfs[i], lastFrame);
					cv::Mat c = b*a;
					khs.insert(std::make_pair(kfs[i], c));
				}
				Matcher match;
				std::cout<<"Matched points with keyframe:"<<match.SearchMatchByGlobal(currFrame, khs)<<"\n";
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
	}

typedef g2o::BlockSolver<g2o::BlockSolverTraits<8, 1>> BlockSolver_8_1;
constexpr float thHuber2D = sqrt(5.99);// from ORBSLAM
const float thHuberDeltaI = 0.1;
const float thHuberDeltaX = 10;
const int numIterations = 100;
// compute homography that transform a point in fr1 to corresponding location in fr2.
//  xfr2 = H*xfr1, consider fr1 as keyframe
// TODO:calculate image gradients with interpolation
cv::Mat Tracking::ComputeHGlobalSBI(Frame* fr1, Frame* fr2)
{
	cv::Mat im1, im2, xgradient, ygradient;
	im1 = fr1->sbiImg; im2 = fr2->sbiImg;

	Sobel(im2, xgradient, CV_32FC1, 1, 0);
	Sobel(im2, ygradient, CV_32FC1, 0, 1);
	xgradient = xgradient / 4.0;
	ygradient = ygradient / 4.0;


	g2o::SparseOptimizer optimizer;
	BlockSolver_8_1::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverDense<BlockSolver_8_1::PoseMatrixType>();

	BlockSolver_8_1 * solver_ptr = new BlockSolver_8_1(linearSolver);

	g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	g2o::SparseOptimizerTerminateAction* action;
	action = new g2o::SparseOptimizerTerminateAction();
	action->setGainThreshold(0.00001);
	action->setMaxIterations(10);
	optimizer.addPostIterationAction(action);

	VertexSL3* vSL3 = new VertexSL3();
	vSL3->updateCache();
	vSL3->setId(0);
	optimizer.addVertex(vSL3);

	for (int i = 0; i < im1.size().height*im1.size().width; i++)
	{
		EdgeSL3* e = new EdgeSL3();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		e->setMeasurement(*(im1.data + i));
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		e->setRobustKernel(rk);
		rk->setDelta(thHuberDeltaI);

		e->loc[0] = i%im1.size().width;
		e->loc[1] = i/im1.size().width;
		e->xgradient = &xgradient;
		e->ygradient = &ygradient;
		e->image = &im2;
		e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
		optimizer.addEdge(e);
	}
	optimizer.setVerbose(true);
	solver->setWriteDebug(true);
	optimizer.initializeOptimization();
	cv::Mat result;
	double data[8];
	//for (int i = 0; i < 200; i++)
	//{
	int validCount = 0;
		optimizer.optimize(50);
		SL3 est;
		VertexSL3* sl3d = static_cast<VertexSL3*>(optimizer.vertex(0));
		est = sl3d->estimate();
		cv::eigen2cv(est._mat, result);
		std::cout << result << "\n";
		std::vector<g2o::OptimizableGraph::Edge*> edges = optimizer.activeEdges();
		for (int i = 0; i < edges.size(); i++)
		{
			EdgeSL3* e = static_cast<EdgeSL3*>(edges[i]);
			if (e->isValid)
			{
				validCount++;
			}
		}
		std::cout << "Valid edge count:" << validCount << "\n";
	//}
	return result;
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

cv::Mat Tracking::ComputeHGlobalKF(KeyFrame* kf, Frame* fr2)
{
	cv::Mat im1 = kf->sbiImg;
	cv::Mat im2 = fr2->sbiImg;
	cv::Mat xgradient, ygradient;
	Sobel(im2, xgradient, CV_32FC1, 1, 0);
	Sobel(im2, ygradient, CV_32FC1, 0, 1);
	xgradient = xgradient / 4.0;
	ygradient = ygradient / 4.0;

	g2o::SparseOptimizer optimizer;
	BlockSolver_8_1::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverDense<BlockSolver_8_1::PoseMatrixType>();

	BlockSolver_8_1 * solver_ptr = new BlockSolver_8_1(linearSolver);

	g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	g2o::SparseOptimizerTerminateAction* action;
	action = new g2o::SparseOptimizerTerminateAction();
	action->setGainThreshold(0.00001);
	action->setMaxIterations(10);
	optimizer.addPostIterationAction(action);

	VertexSL3* vSL3 = new VertexSL3();
	vSL3->setEstimate(SL3());
	vSL3->setId(0);
	optimizer.addVertex(vSL3);

	for (int i = 0; i < im1.size().height*im1.size().width; i++)
	{
		EdgeSL3* e = new EdgeSL3();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		e->setMeasurement(*(im1.data + i));
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		rk->setDelta(thHuberDeltaI);
		e->setRobustKernel(rk);

		e->loc[0] = i%im1.size().width;
		e->loc[1] = i/im1.size().width;
		e->image = &im2;
		e->xgradient = &xgradient;
		e->ygradient = &ygradient;
		e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
		optimizer.addEdge(e);
	}

	auto it = fr2->matchedGroup.find(kf);
	// Must be in its overlapping set.
	assert(it != fr2->matchedGroup.end());
	Eigen::Vector2d kpmea;
	for (auto it2 = it->second.begin();it2!=it->second.end();it2++)
	{
		kpmea[0] = it2->first->pt.x;
		kpmea[1] = it2->first->pt.y;
		EdgeProjection *e = new EdgeProjection();
		e->setVertex(0, optimizer.vertex(0));
		e->setMeasurement(kpmea);
		g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
		rk->setDelta(thHuberDeltaX);
		e->setRobustKernel(rk);

		e->loc[0] = it2->second->pt.x;
		e->loc[1] = it2->second->pt.y;
		e->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
		optimizer.addEdge(e);
	}
	optimizer.initializeOptimization();
	cv::Mat result;
	optimizer.setVerbose(true);
	optimizer.optimize(50);
	SL3 est;
	VertexSL3* sl3d = static_cast<VertexSL3*>(optimizer.vertex(0));
	est = sl3d->estimate();
	cv::eigen2cv(est._mat, result);
	fr2->keyFrameSet.insert(std::make_pair(kf, result));
	std::cout << "optimize with keypoint:"<< result << "\n";
	return result;
}

cv::Mat Tracking::PoseEstimation(Frame* fr)
{
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
	Block* solver_ptr = new Block ( linearSolver );
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm ( solver );

	// vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
	pose->setEstimate ( g2o::SE3Quat ());
	optimizer.addVertex ( pose );

	int index = 1;
	for ( const MapPoint* mp : fr->mappoints)   // landmarks
	{
		if (mp== nullptr)
		{
			continue;
		}
		g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
		point->setId ( index++ );
		point->setEstimate ( Eigen::Vector3d ( mp->Tw(0), mp->Tw(1), mp->Tw(2) ) );
		point->setMarginalized ( true );
		optimizer.addVertex ( point );
	}
	// parameter: camera intrinsics
	g2o::CameraParameters* camera = new g2o::CameraParameters (
			mK.at<double> ( 0,0 ), Eigen::Vector2d ( mK.at<double> ( 0,2 ), mK.at<double> ( 1,2 ) ), 0
	);
	camera->setId ( 0 );
	optimizer.addParameter ( camera );

	// edges
	index = 1;
	for ( int i = 0;i<fr->keypoints.size();i++)
	{
		if (fr->mappoints[i]== nullptr)
		{
			continue;
		}
		g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
		edge->setId ( index );
		edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
		edge->setVertex ( 1, pose );
		edge->setMeasurement ( Eigen::Vector2d ( fr->keypoints[i].pt.x, fr->keypoints[i].pt.y ) );
		edge->setParameterId ( 0,0 );
		edge->setInformation ( Eigen::Matrix2d::Identity() );
		optimizer.addEdge ( edge );
		index++;
	}

	optimizer.setVerbose ( true );
	optimizer.initializeOptimization();
	optimizer.optimize(10);
	std::cout<<"T="<<std::endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<std::endl;
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
