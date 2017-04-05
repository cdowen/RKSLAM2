#include "tracking.h"
#include <opencv2/opencv.hpp>
int main()
{
	cv::Mat im1, im2;
	Frame* fr1 = new Frame();
	Frame* fr2 = new Frame();
		im1 = cv::imread("1.png");
	im2 = cv::imread("2.png");
	fr1->image = im1;
	fr2->image = im2;
	tracking tr;
	auto a = tr.ComputeHGlobalSBI(fr1, fr2);
	std::cout<<a;
	getchar();
}