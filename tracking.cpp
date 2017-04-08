#include <opencv2/opencv.hpp>
#include "tracking.h"
#include <Eigen/Sparse>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>
#include "sl3vertex.h"
#include "sl3edge.h"
#include <opencv2/core/eigen.hpp>
#include <math.h>

typedef g2o::BlockSolver<g2o::BlockSolverTraits<8, 1>> BlockSolver_8_1;
const float thHuber2D = sqrt(5.99);// from ORBSLAM
const int numIterations = 100;
// 计算将fr1中的点通过单应矩阵变换到fr2的矩阵
cv::Mat tracking::ComputeHGlobalSBI(Frame* fr1, Frame* fr2)
{
	cv::Mat im1, im2, xgradient, ygradient;
	im1 = fr1->image; im2 = fr2->image;
	cv::resize(im1, im1, cv::Size(40, 30));
	cv::GaussianBlur(im1, im1, cv::Size(0, 0), 0.75);
	cv::resize(im2, im2, cv::Size(40, 30));
	cv::GaussianBlur(im2, im2, cv::Size(0, 0), 0.75);

	//cv::imwrite("sbi1.png", im1);
	//cv::imwrite("sbi2.png", im2);
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
	action->setMaxIterations(50);
	optimizer.addPostIterationAction(action);

	VertexSL3* vSL3 = new VertexSL3();
	vSL3->setEstimate(SL3());
	SL3 d;
	
	d._mat << 1, 0, 0,
		0, 1, 0,
		0, 0, 1;
		
	vSL3->setEstimate(d);
	vSL3->setId(0);
	optimizer.addVertex(vSL3);

	for (int i = 0; i < im1.size().height*im1.size().width; i++)
	{
		EdgeSL3* e = new EdgeSL3();
		e->setId(i);
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		e->setMeasurement(*(im1.data + i));
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		e->setRobustKernel(rk);
		rk->setDelta(thHuber2D);

		e->loc[0] = i%im1.size().width;
		e->loc[1] = i/im1.size().width;
		e->xgradient = &xgradient;
		e->ygradient = &ygradient;
		e->image = &im2;
		e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
		optimizer.addEdge(e);
	}
	//std::cout<<reinterpret_cast<VertexSL3*> (optimizer.vertex(0))->read();
	//optimizer.vertex(0)->write(std::cout);    
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