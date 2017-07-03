#include <g2o/core/block_solver.h>
#include <opencv2/core/core.hpp>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "Optimizer.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "sl3vertex.h"
#include "sl3edge.h"
#include "ProjectionEdge.h"

typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 1>> BlockSolver_9_1;
constexpr float thHuber2D = sqrt(5.99);// from ORBSLAM
const float thHuberDeltaI = 0.1;
const float thHuberDeltaX = 10;
const int numIterations = 100;
cv::Mat Optimizer::mK;
// compute homography that transform a point in fr1 to corresponding location in fr2.
//  xfr2 = H*xfr1, consider fr1 as keyframe
Eigen::Matrix3d Optimizer::ComputeHGlobalSBI(Frame *fr1, Frame *fr2)
{
	cv::Mat im1, im2, xgradient, ygradient;
	im1 = fr1->sbiImg; im2 = fr2->sbiImg;
	im2.convertTo(im2, CV_64FC1);
	im1.convertTo(im1, CV_64FC1);
	im1 = im1-cv::mean(im1)[0];
	im2 = im2-cv::mean(im2)[0];

	Sobel(im2, xgradient, CV_64FC1, 1, 0, 1);
	Sobel(im2, ygradient, CV_64FC1, 0, 1, 1);


	g2o::SparseOptimizer optimizer;
	BlockSolver_9_1::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverDense<BlockSolver_9_1::PoseMatrixType>();

	BlockSolver_9_1 * solver_ptr = new BlockSolver_9_1(linearSolver);

	g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	g2o::SparseOptimizerTerminateAction* action;
	action = new g2o::SparseOptimizerTerminateAction();
	action->setGainThreshold(0.0001);
	action->setMaxIterations(500);
	optimizer.addPostIterationAction(action);

	VertexSL3* vSL3 = new VertexSL3();
	vSL3->setId(0);
	optimizer.addVertex(vSL3);

	for (int i = 0; i < im1.size().height*im1.size().width; i++)
	{
		EdgeSL3* e = new EdgeSL3();
		e->setVertex(0, optimizer.vertex(0));
		e->setMeasurement(*((double*)im1.data + i));
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		e->setRobustKernel(rk);
		rk->setDelta(thHuberDeltaI);

		e->loc[0] = i%im1.size().width*16;
		e->loc[1] = i/im1.size().width*16;
		e->xgradient = xgradient;
		e->ygradient = ygradient;
		e->_image = im2;
		e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
		optimizer.addEdge(e);
	}
	//optimizer.setVerbose(true);
	//solver->setWriteDebug(true);
	optimizer.initializeOptimization();
	int validCount = 0;
	optimizer.optimize(300);
	VertexSL3* sl3d = static_cast<VertexSL3*>(optimizer.vertex(0));
	std::cout << sl3d->estimate() << "\n";
	std::cout<<"mu:"<<sl3d->mu<<"\n";
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
	return sl3d->estimate();
}



Eigen::Matrix3d Optimizer::ComputeHGlobalKF(KeyFrame *kf, Frame *fr2)
{
	cv::Mat im1 = kf->sbiImg;
	cv::Mat im2 = fr2->sbiImg;
	cv::Mat xgradient, ygradient;
	im1.convertTo(im1, CV_64FC1);
	im2.convertTo(im2, CV_64FC1);
	im1 = im1-cv::mean(im1)[0];
	im2 = im2-cv::mean(im2)[0];

	Sobel(im2, xgradient, CV_64FC1, 1, 0, 1);
	Sobel(im2, ygradient, CV_64FC1, 0, 1, 1);

	g2o::SparseOptimizer optimizer;
	BlockSolver_9_1::LinearSolverType * linearSolver = new g2o::LinearSolverDense<BlockSolver_9_1::PoseMatrixType>();

	BlockSolver_9_1 * solver_ptr = new BlockSolver_9_1(linearSolver);

	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	g2o::SparseOptimizerTerminateAction* action;
	action = new g2o::SparseOptimizerTerminateAction();
	action->setGainThreshold(0.0001);
	action->setMaxIterations(20);
	optimizer.addPostIterationAction(action);

	VertexSL3* vSL3 = new VertexSL3();

	auto hit = fr2->keyFrameSet.find(kf);
	if (hit!=fr2->keyFrameSet.end())
	{
		vSL3->setEstimate(hit->second);
	}
	else
	{
		vSL3->setEstimate(Eigen::Matrix3d::Identity());
	}
	vSL3->setId(0);
	optimizer.addVertex(vSL3);

	for (int i = 0; i < im1.size().height*im1.size().width; i++)
	{
		EdgeSL3* e = new EdgeSL3();
		e->setVertex(0, optimizer.vertex(0));
		e->setMeasurement(*((double*)im1.data + i));
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		rk->setDelta(thHuberDeltaI);
		e->setRobustKernel(rk);

		e->loc[0] = i%im1.size().width*16;
		e->loc[1] = i/im1.size().width*16;
		e->_image = im2;
		e->xgradient = xgradient;
		e->ygradient = ygradient;
		e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
		//optimizer.addEdge(e);
	}
	auto it = fr2->matchedGroup.find(kf);
	// Must be in its overlapping set.
	assert(it != fr2->matchedGroup.end());
	Eigen::Vector2d kpmea;
	for (auto it2 = it->second.begin();it2!=it->second.end();it2++)
	{
		kpmea[0] = fr2->keypoints[it2->first].pt.x;
		kpmea[1] = fr2->keypoints[it2->first].pt.y;
		EdgeProjection *e = new EdgeProjection();
		e->setVertex(0, optimizer.vertex(0));
		e->setMeasurement(kpmea);
		g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
		rk->setDelta(thHuberDeltaX);
		e->setRobustKernel(rk);

		e->loc[0] = kf->keypoints[it2->second].pt.x;
		e->loc[1] = kf->keypoints[it2->second].pt.y;
		e->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
		optimizer.addEdge(e);
	}
	optimizer.initializeOptimization();
	//optimizer.setVerbose(true);
	optimizer.optimize(50);
	VertexSL3* sl3d = static_cast<VertexSL3*>(optimizer.vertex(0));
	fr2->keyFrameSet[kf] = sl3d->estimate();
	std::cout << "optimize with keypoint:"<< sl3d->estimate() << "\n";
	return sl3d->estimate();
}

// v[0-2]:t;v[3-6]:x,y,z,w
Optimizer::Vector7d Optimizer::PoseEstimation(Frame* fr)
{
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
	Block* solver_ptr = new Block ( linearSolver );
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm ( solver );

	g2o::SparseOptimizerTerminateAction* action;
	action = new g2o::SparseOptimizerTerminateAction();
	action->setGainThreshold(0.001);
	action->setMaxIterations(8);
	optimizer.addPostIterationAction(action);

	// vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
	pose->setId(0);
	pose->setEstimate ( g2o::SE3Quat ());
	optimizer.addVertex ( pose );

	int index = 1;
	for ( const MapPoint* mp : fr->mappoints)   // landmarks
	{
		if (mp!= nullptr)
		{
			g2o::VertexSBAPointXYZ *point = new g2o::VertexSBAPointXYZ();
			point->setId(index++);
			point->setEstimate(Eigen::Vector3d(mp->Tw(0), mp->Tw(1), mp->Tw(2)));
			point->setMarginalized(true);
			optimizer.addVertex(point);
		}
	}
	// parameter: camera intrinsics
	g2o::CameraParameters* camera = new g2o::CameraParameters (
			Optimizer::mK.at<double> ( 0,0 ), Eigen::Vector2d ( Optimizer::mK.at<double> ( 0,2 ), Optimizer::mK.at<double> ( 1,2 ) ), 0
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

	//optimizer.setVerbose ( true );
	optimizer.initializeOptimization();
	optimizer.optimize(10);
	std::cout<<"T="<<std::endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<std::endl;
	return pose->estimate().toVector();
}