//
// Created by u on 17-4-19.
//

#ifndef RKSLAM2_OPTIMIZATION_CERES_H
#define RKSLAM2_OPTIMIZATION_CERES_H

#include "Frame.h"

class optimization_ceres
{
public:
	void ComputeHGlobalSBI(Frame* fr1, Frame* fr2);
};


#endif //RKSLAM2_OPTIMIZATION_CERES_H
