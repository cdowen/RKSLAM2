#include <opencv2/opencv.hpp>
#include "tracking.h"
#include <Eigen/Sparse>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>
#include "sl3vertex.h"
#include "sl3edge.h"
#include <opencv2/core/eigen.hpp>

//typedef g2o::BlockSolver<g2o::BlockSolverTraits<8, 1>> BlockSolver_8_1;
const float thHuber2D = sqrt(5.99);// from ORBSLAM
const int numIterations = 100;
cv::Mat tracking::ComputeHGlobalSBI(Frame* fr1, Frame* fr2)
{
	cv::Mat im1, im2, xgradient, ygradient;
	cv::resize(fr1->image, im1, cv::Size(40, 30));
	cv::GaussianBlur(im1, im1, cv::Size(0, 0), 0.75);
	cv::resize(fr2->image, im2, cv::Size(40, 30));
	cv::GaussianBlur(im2, im2, cv::Size(0, 0), 0.75);

	Sobel(im1, xgradient, CV_8UC1, 1, 0);
	Sobel(im2, ygradient, CV_8UC1, 0, 1);

	g2o::SparseOptimizer optimizer;
	g2o::BlockSolverX::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
	g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

	g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	VertexSL3* vSL3 = new VertexSL3();
	vSL3->setEstimate(SL3());
	SL3 d;
	d._mat<<1.414/2,-1.414/2, 0, 
	1.414/2, 1.414/2, 0, 
	0,0,1; 
	vSL3->setEstimate(d);
	vSL3->setId(0);
	optimizer.addVertex(vSL3);

	for (int i = 0; i < im1.size().height*im1.size().width; i++)
	{
		EdgeSL3* e = new EdgeSL3();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
		e->setMeasurement(*(im1.data + i));
		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		e->setRobustKernel(rk);
		rk->setDelta(thHuber2D);
		e->xgradient = &xgradient;
		e->loc[0] = i%im1.size().width;
		e->loc[1] = i/im1.size().width;
		e->ygradient = &ygradient;
		e->image = &im2;

		optimizer.addEdge(e);
	}
	//std::cout<<reinterpret_cast<VertexSL3*> (optimizer.vertex(0))->read();
	//optimizer.vertex(0)->write(std::cout);
	optimizer.initializeOptimization();
	optimizer.optimize(100);
	Vector8d sl3alg;
	optimizer.vertex(0)->getEstimateData(sl3alg.data());
	
	SL3 optimizedH;
	optimizedH.fromVector(sl3alg);
	cv::Mat result;
	std::cout<<"\n"<<sl3alg<<"\n";
	cv::eigen2cv(optimizedH._mat, result);
	return result;
}