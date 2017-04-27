#include "tracking.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"
#include <stdio.h>
#include"Initialization.h"
#include<fstream>
#include <iomanip>

bool testSL3();
void testMatchByH(Frame* fr1, Frame* fr2, cv::Mat H);

int main()
{

	Tracking tr; tr.mK=(cv::Mat_<double>(3,3) <<517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);
	tr.Run();
}