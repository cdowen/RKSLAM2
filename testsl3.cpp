#include "sl3.h"
#include <iostream>
#include "Frame.h"
#include "KeyFrame.h"
#include <opencv2/opencv.hpp>
#include "matcher.h"

bool testSL3()
{
	Vector8d d;
	d << 1, 2, 3, 4, 5, 6, 7, 8;
	SL3 data;
	data.fromVector(d);
	std::cout << data._mat << "\n";
	for (int i = 0; i < 100; i++)
	{
		data.regularize();
	}
	Vector8d e;
	e = data.toVector();
	std::cout << e << "\n";
	return true;
}

// fr1 as the reference frame; project points in fr2 to fr1;
void testMatchByH(Frame* fr1, Frame* fr2, cv::Mat H)
{
	Matcher Match;
	std::map<KeyFrame*, cv::Mat> vec;
	vec.insert(std::make_pair(static_cast<KeyFrame*>(fr2), H));
	int match_num = Match.SearchMatchByGlobal(fr1, vec);
	std::cout<<"matched points by H:"<<match_num<<"\n";
}