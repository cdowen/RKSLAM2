//
// Created by user on 17-5-9.
//

#include "LocalMap.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <chrono>
#include <vector>

LocalMap::LocalMap(Tracking* tracking)
{
	AllFrameInWindow.clear();
	mK=tracking->mK;
}

void LocalMap::AddFrameInWindow(Frame *fr1)
{
	AllFrameInWindow.push_back(fr1);
	if(AllFrameInWindow.size()>FrameSizeInWindow)
	{
		std::vector<Frame*>::iterator k=AllFrameInWindow.begin();

		//erase observation of Mappoints in the deleted frame.
		Frame* Framek=*k;
		int PointNum=Framek->mappoints.size();
		for(int i=0;i<PointNum;i++)
		{
			MapPoint* MP=Framek->mappoints.at(i);
			for(std::map<Frame*, cv::KeyPoint*>::iterator iter=MP->allObservation.begin();iter!=MP->allObservation.end();++iter)
			{
				if(iter->first==Framek)
				{
					MP->allObservation.erase(Framek);
				} else continue;
			}
		}

		AllFrameInWindow.erase(k);
	}
}

double LocalMap::ComputeParallax(cv::KeyPoint kp1, cv::KeyPoint kp2, KeyFrame *kf1, Frame *fr2)
{
	cv::Mat keypoint1=(cv::Mat_<double >(3,1)<< kp1.pt.x,kp1.pt.y,1);
	cv::Mat keypoint2=(cv::Mat_<double >(3,1)<< kp2.pt.x,kp2.pt.y,1);
	cv::Mat mR1=kf1->mTcw.rowRange(0,4).colRange(0,4);
	cv::Mat mR2=fr2->mTcw.rowRange(0,4).colRange(0,4);

	cv::Mat r1=mR1.t()*mK.inv()*keypoint1;
	cv::Mat r2=mR2.t()*mK.inv()*keypoint2;
	double dist1=cv::norm(r1);
	double dist2=cv::norm(r2);

	double RayAngle=acos(r1.dot(r2)/(dist1*dist2));

	if(RayAngle==NAN){
		return -1;
	} else return RayAngle*180/M_PI;
}

MapPoint* LocalMap::Triangulation(cv::KeyPoint kp1, cv::KeyPoint kp2,KeyFrame *kf1, Frame *fr2)
{
	cv::Mat A(4,4,CV_64F);

	cv::Mat P1=mK*kf1->mTcw;
	cv::Mat P2=mK*fr2->mTcw;
	A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
	A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
	A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
	A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

	cv::Mat u,w,vt;
	cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

	MapPoint* MP=new MapPoint();
	MP->Tw = vt.row(3).t();
	MP->Tw = MP->Tw.rowRange(0,3)/MP->Tw.at<double>(3);
	MP->allObservation[kf1]=&kp1;

	return MP;
}

MapPoint* LocalMap::GetPositionByOptimization(cv::KeyPoint kp1, cv::KeyPoint kp2, KeyFrame *kf1, Frame *fr2)
{
	MapPoint* MP=new MapPoint();
	cv::Mat keypoint1=(cv::Mat_<float >(3,1)<< kp1.pt.x,kp1.pt.y,1);

	//choose the mean depth value d of keyframe kf1 to initialize mappoint.
	double SumDepth=0;
	for(std::vector<MapPoint*>::iterator vit=kf1->mappoints.begin(),vend=kf1->mappoints.end();vit!=vend;vit++)
	{
		MapPoint* mMP=*vit;
		cv::Mat Coordinate(4,1,CV_64F,1);
		Coordinate.rowRange(0,3)=mMP->Tw;
		Coordinate= kf1->mTcw*Coordinate;
		double d =Coordinate.at<double>(2);
		SumDepth += d;
	}
	double MeanDepth=SumDepth/kf1->mappoints.size();

	MP->Tw=MeanDepth*kf1->mTcw.t()*mK.inv()*keypoint1;

	//optimize.
		//Initialize g2o.
	typedef g2o::BlockSolver_6_3 Block;
	Block::LinearSolverType* linearSolver=new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
	Block* solver_ptr=new Block(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* algorithmLevenberg=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(algorithmLevenberg);

		//set vertex:camera pose.
	g2o::VertexSE3Expmap* pose1 =new g2o::VertexSE3Expmap();
	Eigen::Matrix3d R_mat;
	R_mat<<
		 	kf1->mTcw.at<double>(0,0),kf1->mTcw.at<double>(0,1),kf1->mTcw.at<double>(0,2),
			kf1->mTcw.at<double>(1,0),kf1->mTcw.at<double>(1,1),kf1->mTcw.at<double>(1,2),
			kf1->mTcw.at<double>(2,0),kf1->mTcw.at<double>(2,1),kf1->mTcw.at<double>(2,2);
	pose1->setId(0);
	pose1->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(kf1->mTcw.at<double>(0,3),kf1->mTcw.at<double>(1,3),kf1->mTcw.at<double>(2,3))));
	pose1->setFixed(true);
	optimizer.addVertex(pose1);
	g2o::VertexSE3Expmap* pose2 =new g2o::VertexSE3Expmap();
	Eigen::Matrix3d R_mat2;
	R_mat2<<
		 	fr2->mTcw.at<double>(0,0),fr2->mTcw.at<double>(0,1),fr2->mTcw.at<double>(0,2),
			fr2->mTcw.at<double>(1,0),fr2->mTcw.at<double>(1,1),fr2->mTcw.at<double>(1,2),
			fr2->mTcw.at<double>(2,0),fr2->mTcw.at<double>(2,1),fr2->mTcw.at<double>(2,2);
	pose1->setId(1);
	pose1->setEstimate(g2o::SE3Quat(R_mat2,Eigen::Vector3d(fr2->mTcw.at<double>(0,3),fr2->mTcw.at<double>(1,3),fr2->mTcw.at<double>(2,3))));
	pose1->setFixed(true);
	optimizer.addVertex(pose1);

		//set vertices:3D features.
	g2o::VertexSBAPointXYZ* point=new g2o::VertexSBAPointXYZ();
	point->setId(2);
	point->setEstimate(Eigen::Vector3d(MP->Tw.at<double>(0),MP->Tw.at<double>(1),MP->Tw.at<double>(2)));
	optimizer.addVertex(point);

		//set parameter:camera intrinsic.
	g2o::CameraParameters* camera=new g2o::CameraParameters(mK.at<double>(0,0),Eigen::Vector2d(mK.at<double>(0,2),mK.at<double>(1,2)),0);
	camera->setId(0);
	optimizer.addParameter(camera);

		//set edges.
	g2o::EdgeProjectXYZ2UV* edge1=new g2o::EdgeProjectXYZ2UV();
	edge1->setId(1);
	edge1->setVertex(0,point);
	edge1->setVertex(1,pose1);
	edge1->setMeasurement(Eigen::Vector2d(kp1.pt.x,kp1.pt.y));
	edge1->setInformation(Eigen::Matrix2d::Identity());
	optimizer.addEdge(edge1);
	g2o::EdgeProjectXYZ2UV* edge2=new g2o::EdgeProjectXYZ2UV();
	edge2->setId(2);
	edge2->setVertex(0,point);
	edge2->setVertex(1,pose2);
	edge2->setMeasurement(Eigen::Vector2d(kp2.pt.x,kp2.pt.y));
	edge2->setInformation(Eigen::Matrix2d::Identity());
	optimizer.addEdge(edge2);

		//optimize!
	std::chrono::steady_clock::time_point t1=std::chrono::steady_clock::now();
	optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(100);
	std::chrono::steady_clock::time_point t2=std::chrono::steady_clock::now();
	std::chrono::duration<double > time_used=std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	std::cout<<"3D point initialization by optimization costs time: "<<time_used.count()<<" seconds\n";

	//Recover the optimized point.
	g2o::VertexSBAPointXYZ* vPoint=static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(2));
	MP->Tw<<(vPoint->estimate())(0),(vPoint->estimate())(1),(vPoint->estimate())(2);
	return MP;
}

void LocalMap::LocalOptimize(Frame *fr1)
{
	//Initialize g2o:
	typedef g2o::BlockSolver_6_3 Block;
	Block::LinearSolverType* linearSolver=new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
	Block* solver_ptr=new Block(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* algorithmLevenberg=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		//Only optimize points first.
	g2o::SparseOptimizer OptimizerPoint;
	OptimizerPoint.setAlgorithm(algorithmLevenberg);
		//Only optimize camera pose.
	g2o::SparseOptimizer OptimizerPose;
	OptimizerPose.setAlgorithm(algorithmLevenberg);


	//Set vertices.
	int i=0;//vertex id in optimizerPoint.
	int i1=0;//vertex id in optimizerPose.
	int EdgeId=1;//vertex id in optimizerPoint.
	int EdgeId1=1;//vertex id in optimizerPose.
	std::map<unsigned long,int> OptimizedFrame={};//optimized frame in optimizerPoint.
	for (std::vector<Frame*>::iterator vitF=AllFrameInWindow.begin(),vendF=AllFrameInWindow.end();vitF!=vendF;vitF++)
	{
		Frame* Fr=*vitF;

		//add camera pose in the window into optimizerPoint.
		g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
		Eigen::Matrix3d R_mat;
		R_mat<<
			 	Fr->mTcw.at<double>(0,0),Fr->mTcw.at<double>(0,1),Fr->mTcw.at<double>(0,2),
				Fr->mTcw.at<double>(1,0),Fr->mTcw.at<double>(1,1),Fr->mTcw.at<double>(1,2),
				Fr->mTcw.at<double>(2,0),Fr->mTcw.at<double>(2,1),Fr->mTcw.at<double>(2,2);
		pose->setId(i++);
		pose->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(Fr->mTcw.at<double>(0,3),Fr->mTcw.at<double>(1,3),Fr->mTcw.at<double>(2,3))));
		pose->setFixed(true);
		OptimizerPoint.addVertex(pose);
		OptimizedFrame.insert(std::make_pair(Fr->id,i));

		//add camera pose in the window into optimizerPose.
		g2o::VertexSE3Expmap* pose1=new g2o::VertexSE3Expmap();
		pose1->setId(i1++);
		pose1->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(Fr->mTcw.at<double>(0,3),Fr->mTcw.at<double>(1,3),Fr->mTcw.at<double>(2,3))));
		pose1->setFixed(false);
		OptimizerPose.addVertex(pose1);

		for(unsigned long mp=0;mp<Fr->mappoints.size();mp++)
		{
			MapPoint* MP=Fr->mappoints.at(mp);

			//set vertex in optimizerPoint: point.
			g2o::VertexSBAPointXYZ* Point=new g2o::VertexSBAPointXYZ();
			Point->setId(i++);
			Point->setEstimate(Eigen::Vector3d(MP->Tw.at<double>(0),MP->Tw.at<double>(1),MP->Tw.at<double>(2)));
			if(MP->WellConstrained)//fix the well-constrained points.
			{
				Point->setFixed(true);
			} else Point->setFixed(false);
			OptimizerPoint.addVertex(Point);

			//set vertex in optimizerPose: point.
			g2o::VertexSBAPointXYZ* Point1=new g2o::VertexSBAPointXYZ();
			Point1->setId(i1++);
			Point1->setEstimate(Eigen::Vector3d(MP->Tw.at<double>(0),MP->Tw.at<double>(1),MP->Tw.at<double>(2)));
			Point1->setFixed(true);
			OptimizerPose.addVertex(Point1);

			//set edge in optimizerPose.
			g2o::EdgeProjectXYZ2UV* edge1=new g2o::EdgeProjectXYZ2UV();
			edge1->setId(EdgeId1++);
			edge1->setVertex(0,Point1);
			edge1->setVertex(1,pose1);
			edge1->setMeasurement(Eigen::Vector2d(Fr->keypoints.at(mp).pt.x,Fr->keypoints.at(mp).pt.y));
			edge1->setInformation(Eigen::Matrix2d::Identity());
			OptimizerPose.addEdge(edge1);


			//set vertex in optimizerPoint: pose which observes the points
			for (std::map<Frame*, cv::KeyPoint*>::iterator iter=MP->allObservation.begin();iter!=MP->allObservation.end();++iter)
			{
				Frame* ObsFr=iter->first;

				std::map<unsigned long,int>::iterator VertexNotExist=OptimizedFrame.find(ObsFr->id);
				if(VertexNotExist==OptimizedFrame.end())
				{
					g2o::VertexSE3Expmap* ObsPose=new g2o::VertexSE3Expmap();
					R_mat<<
						 	ObsFr->mTcw.at<double>(0,0),ObsFr->mTcw.at<double>(0,1),ObsFr->mTcw.at<double>(0,2),
							ObsFr->mTcw.at<double>(1,0),ObsFr->mTcw.at<double>(1,1),ObsFr->mTcw.at<double>(1,2),
							ObsFr->mTcw.at<double>(2,0),ObsFr->mTcw.at<double>(2,1),ObsFr->mTcw.at<double>(2,2);
					ObsPose->setId(i++);
					ObsPose->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(Fr->mTcw.at<double>(0,3),Fr->mTcw.at<double>(1,3),Fr->mTcw.at<double>(2,3))));
					ObsPose->setFixed(true);
					OptimizerPoint.addVertex(ObsPose);

					OptimizedFrame.insert(std::make_pair(ObsFr->id,i));
				}

				//set edge in optimizerPoint.
				g2o::EdgeProjectXYZ2UV* edge=new g2o::EdgeProjectXYZ2UV();
				edge->setId(EdgeId++);
				edge->setVertex(0,Point);
				unsigned long ObsFrVertexId=OptimizedFrame.count(ObsFr->id);
				edge->setVertex(1,OptimizerPoint.vertex(ObsFrVertexId));
				edge->setMeasurement(Eigen::Vector2d(iter->second->pt.x,iter->second->pt.y));
				edge->setInformation(Eigen::Matrix2d::Identity());
				OptimizerPoint.addEdge(edge);
			}
		}
	}

	//set parameter:camera intrinsic.
	g2o::CameraParameters* camera=new g2o::CameraParameters(mK.at<double>(0,0),Eigen::Vector2d(mK.at<double>(0,2),mK.at<double>(1,2)),0);
	camera->setId(0);
	OptimizerPoint.addParameter(camera);
	OptimizerPose.addParameter(camera);

	//optimize!
		//optimize ill-conditioned points.
	std::chrono::steady_clock::time_point t1=std::chrono::steady_clock::now();
	OptimizerPoint.setVerbose(true);
	OptimizerPoint.initializeOptimization();
	OptimizerPoint.optimize(100);
	std::chrono::steady_clock::time_point t2=std::chrono::steady_clock::now();
	std::chrono::duration<double > time_used_for_points=std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	std::cout<<"3D point Optimization costs time: "<<time_used_for_points.count()<<" seconds\n";
		//optimize camera poses in the window.
	std::chrono::steady_clock::time_point t3=std::chrono::steady_clock::now();
	OptimizerPose.setVerbose(true);
	OptimizerPose.initializeOptimization();
	OptimizerPose.optimize(100);
	std::chrono::steady_clock::time_point t4=std::chrono::steady_clock::now();
	std::chrono::duration<double > time_used_for_pose=std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	std::cout<<"Pose Optimization costs time: "<<time_used_for_pose.count()<<" seconds\n";
}
