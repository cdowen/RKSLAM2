//
// Created by u on 17-4-19.
//

#include "optimization_ceres.h"
#include "sl3.h"
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
bool validateProjection(Eigen::Vector3d hLoc, cv::Size s);

void optimization_ceres::ComputeHGlobalSBI(Frame *fr1, Frame *fr2)
{
	class SBICost:public ceres::SizedCostFunction<1,8>
	{
	public:
		virtual ~SBICost(){};
		cv::Mat* image;
		cv::Mat* xgrad, *ygrad;
		Eigen::Vector2i loc;
		uchar measurement;
		virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians)const
		{
			Vector8d param(parameters[0]);
			SL3 hT;
			hT.fromVector(param);
			Eigen::Vector3d hLoc;
			hLoc << loc[0], loc[1], 1;

			hLoc = hT._mat*hLoc;
			hLoc = hLoc / hLoc[2];

			if (!validateProjection(hLoc, image->size()))
			{
				residuals[0] = 0;

				for (int i = 0; i < 8; i++)
				{
					if (jacobians!=NULL&&jacobians[0]!=NULL)
					{
						jacobians[0][i] = 0;
					}
				}
			}
			else
			{
				unsigned char data= image->at<unsigned char>(hLoc[1], hLoc[0]);
				residuals[0] = measurement - data;

				float xgrd = xgrad->at<float>(hLoc[1], hLoc[0]);
				float ygrd = ygrad->at<float>(hLoc[1], hLoc[0]);
				if (jacobians!=NULL&&jacobians[0]!=NULL)
				{
					jacobians[0][0] = -xgrd;
					jacobians[0][1] = -ygrd;
					jacobians[0][2] = -loc[1] * xgrd;
					jacobians[0][3] = -loc[0] * ygrd;
					jacobians[0][4] = -loc[0] * xgrd + loc[1] * ygrd;
					jacobians[0][5] = loc[0] * xgrd + 2 * loc[1] * ygrd;
					jacobians[0][6] = loc[0] * loc[0] * xgrd + loc[0] * loc[1] * ygrd;
					jacobians[0][7] = loc[0] * loc[1] * xgrd + loc[1] * loc[1] * ygrd;
				}
			}
			return true;
		}
	};
	double x[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	ceres::Problem problem;
	cv::Mat im1, im2, xgradient, ygradient;
	im1 = fr1->sbiImg;im2 = fr2->sbiImg;
	Sobel(im2, xgradient, CV_32FC1, 1, 0);
	Sobel(im2, ygradient, CV_32FC1, 0, 1);
	xgradient = xgradient / 4.0;
	ygradient = ygradient / 4.0;

	for (int i = 0; i < im1.size().height*im1.size().width; i++)
	{
		SBICost *cf = new SBICost;
		cf->loc[0] = i%im1.size().width;
		cf->loc[1] = i/im1.size().width;
		cf->xgrad = &xgradient;
		cf->ygrad = &ygradient;
		cf->image = &im2;
		ceres::HuberLoss *loss = new ceres::HuberLoss(0.1);
		problem.AddResidualBlock(cf, loss, x);
	}
		ceres::Solver::Options options;
		options.minimizer_progress_to_stdout = true;
		options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
		options.max_num_iterations = 10;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";
	std::cout << "h : "<<x<<"\n";
}
