#include "sl3edge.h"

bool validateProjection(Eigen::Vector3d hLoc, cv::Size s)
{
	if (hLoc[0] >= 0 && hLoc[0] < s.width-1&&hLoc[1]>=0 && hLoc[1] < s.height-1)
	{
		return true;
	}
	return false;
}

float getPixelValuef(float x, float y, cv::Mat* image)
{
	float* data = (float*)image->ptr(int(y));
	data = data+int(x);
	float xx = x - floor ( x );
	float yy = y - floor ( y );
	float res =
			( 1-xx ) * ( 1-yy ) * data[0] +
			xx* ( 1-yy ) * data[1] +
			( 1-xx ) *yy*data[ image->step1() ] +
			xx*yy*data[image->step1()+1];
	//std::cout<<"interpoed:"<<res<<" other four:"<<*data<<" "<<*(data+1)<<" "<<float(image->at<float>(y+1, x))<<" "<<float(image->at<float>(y+1, x+1))<<std::endl;
	return res;
}

Eigen::Vector3d EdgeSL3::homo_project()
	{
		Eigen::Vector3d hLoc;
		hLoc << loc[0], loc[1], 1;
		const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
	
		hLoc = v1->estimate()*hLoc;
		hLoc = hLoc / hLoc[2];
		return hLoc;
	}
	void EdgeSL3::computeError()
	{
		Eigen::Vector3d hLoc = homo_project();
		if (!validateProjection(hLoc, _image->size()))
		{
			isValid = false;
		    _error[0] = 0;			
		    return;
		}
		isValid = true;
		float data = getPixelValue(hLoc[0], hLoc[1]);
		//float data = _image->at<uint8_t>(hLoc[1], hLoc[0]);

		const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
		_error[0] = _measurement - data+v1->mu;
	}

	void EdgeSL3::linearizeOplus()
	{
		Eigen::Vector3d hLoc = homo_project();
		if (!validateProjection(hLoc, _image->size()))
		{
			for (int i = 0; i < 8; i++)
			{
				_jacobianOplusXi[i] = 0;
			}
			isValid = false;
		    return;
		}
		float xgrd = getPixelValuef(hLoc[0], hLoc[1], xgradient);
		float ygrd = getPixelValuef(hLoc[0], hLoc[1], ygradient);
		//float xgrd = xgradient->at<float>(hLoc[1], hLoc[0]);
		//float ygrd = ygradient->at<float>(hLoc[1], hLoc[0]);

		/*
		_jacobianOplusXi[0] = -xgrd;
		_jacobianOplusXi[1] = -ygrd;
		_jacobianOplusXi[2] = -loc[1] * xgrd;
		_jacobianOplusXi[3] = -loc[0] * ygrd;
		_jacobianOplusXi[4] = -loc[0] * xgrd + loc[1] * ygrd;
		_jacobianOplusXi[5] = loc[0] * xgrd + 2 * loc[1] * ygrd;
		_jacobianOplusXi[6] = loc[0] * loc[0] * xgrd + loc[0] * loc[1] * ygrd;
		_jacobianOplusXi[7] = loc[0] * loc[1] * xgrd + loc[1] * loc[1] * ygrd;
		 */
		/*
		_jacobianOplusXi[0] = loc[0]*loc[0]*xgrd+loc[0]*loc[1]*ygrd;
		_jacobianOplusXi[1] = loc[0] * loc[1] * xgrd + loc[1] * loc[1] * ygrd;
		_jacobianOplusXi[2] = -loc[0]*ygrd;
		_jacobianOplusXi[3] = -loc[1]*xgrd;
		_jacobianOplusXi[4] = -loc[0] * xgrd + loc[1] * ygrd;
		_jacobianOplusXi[5] = loc[0] * xgrd + 2 * loc[1] * ygrd;
		_jacobianOplusXi[6] = -xgrd;
		_jacobianOplusXi[7] = -ygrd;
		 */
		float x1 = loc[0];
		float x2 = loc[1];
		/*
		_jacobianOplusXi[0] = -xgrd;
		_jacobianOplusXi[1] = -ygrd;
		_jacobianOplusXi[2] = x2*xgrd-x1*ygrd;
		_jacobianOplusXi[3] = -3*x1*xgrd-3*x2*ygrd;
		_jacobianOplusXi[4] = -x1*xgrd+x2*ygrd;
		_jacobianOplusXi[5] = -x2*xgrd-x1*ygrd;
		_jacobianOplusXi[6] = x1*x1*xgrd+x1*x2*ygrd;
		_jacobianOplusXi[7] = x1*x2*xgrd+x2*x2*ygrd;
		*/
		_jacobianOplusXi[0] = -x1*xgrd;
		_jacobianOplusXi[1] = -x2*xgrd;
		_jacobianOplusXi[2] = -xgrd;
		_jacobianOplusXi[3] = -x1*ygrd;
		_jacobianOplusXi[4] = -x2*ygrd;
		_jacobianOplusXi[5] = -ygrd;
		_jacobianOplusXi[6] = x1*x1*xgrd+x1*x2*ygrd;
		_jacobianOplusXi[7] = x1*x2*xgrd+x2*x2*ygrd;
		_jacobianOplusXi[8] = x1*xgrd+x2*xgrd;
		_jacobianOplusXi[9] = 1;
		std::vector<double> debugN;
		isValid = true;
		for (int i = 0; i < 9; i++)
		{
			debugN.push_back(_jacobianOplusXi[i]);
		}
	}

	bool EdgeSL3::write(std::ostream& os) const
	{
		return os.good();
	}

	bool EdgeSL3::read(std::istream& is)
	{
		return true;
	}

float EdgeSL3::getPixelValue ( float x, float y)
{
	uchar* data = & _image->data[ int ( y ) * _image->step + int ( x ) ];
	float xx = x - floor ( x );
	float yy = y - floor ( y );
	float res =
		float (
			( 1-xx ) * ( 1-yy ) * data[0] +
			xx* ( 1-yy ) * data[1] +
			( 1-xx ) *yy*data[ _image->step ] +
			xx*yy*data[_image->step+1]
	);

	//std::cout<<"interpoed:"<<res<<" other four:"<<int(*data)<<" "<<int(*(data+1))<<" "<<int(_image->at<uint8_t>(y+1, x))<<" "<<int(_image->at<uint8_t>(y+1, x+1))<<std::endl;
	return res;
}

