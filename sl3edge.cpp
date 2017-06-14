#include "sl3edge.h"

bool validateProjection(Eigen::Vector2d hLoc, cv::Size s)
{
	if (hLoc[0] >= 0 && hLoc[0] < s.width-1&&hLoc[1]>=0 && hLoc[1] < s.height-1)
	{
		return true;
	}
	return false;
}

double getPixelValuef(double x, double y, cv::Mat *image)
{
	assert(image->isContinuous());
	double* data = (double*)image->ptr(int(y));
	data = data+int(x);
	double xx = x - floor ( x );
	double yy = y - floor ( y );
	double res =
			( 1-xx ) * ( 1-yy ) * data[0] +
			xx* ( 1-yy ) * data[1] +
			( 1-xx ) *yy*data[ image->step1() ] +
			xx*yy*data[image->step1()+1];
	return res;
}

Eigen::Vector2d EdgeSL3::homo_project()
	{
		const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
	
		return (v1->estimate()*loc.homogeneous()).hnormalized();
	}
	void EdgeSL3::computeError()
	{
		Eigen::Vector2d hLoc = homo_project();
		hLoc = hLoc/16.0;
		if (!validateProjection(hLoc, _image.size()))
		{
			isValid = false;
		    _error[0] = 0;			
		    return;
		}
		isValid = true;
		double data = getPixelValuef(hLoc[0], hLoc[1], &_image);
		//double data = _image->at<uint8_t>(hLoc[1], hLoc[0]);

		const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
		_error[0] = _measurement - data+v1->mu;
	}

	void EdgeSL3::linearizeOplus()
	{
		Eigen::Vector2d hLoc = homo_project();
		hLoc = hLoc/16.0;
		if (!validateProjection(hLoc, _image.size()))
		{
			for (int i = 0; i < 8; i++)
			{
				_jacobianOplusXi[i] = 0;
			}
			isValid = false;
		    return;
		}
		double xgrd = getPixelValuef(hLoc[0], hLoc[1], &xgradient);
		double ygrd = getPixelValuef(hLoc[0], hLoc[1], &ygradient);

		double x1 = loc[0];
		double x2 = loc[1];

		const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
		const Eigen::Matrix3d h = v1->estimate();
		double ww = h(2, 0)*loc[0]+h(2,1)*loc[1]+1;
		ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
		double xi = (h(0,0)*loc[0]+h(0,1)*loc[1]+h(0,2))*ww;
		double yi = (h(1,0)*loc[0]+h(1,1)*loc[1]+h(1,2))*ww;

		_jacobianOplusXi[0] = -x1*xgrd*ww;
		_jacobianOplusXi[1] = -x2*xgrd*ww;
		_jacobianOplusXi[2] = -xgrd*ww;
		_jacobianOplusXi[3] = -x1*ygrd*ww;
		_jacobianOplusXi[4] = -x2*ygrd*ww;
		_jacobianOplusXi[5] = -ygrd*ww;
		_jacobianOplusXi[6] = x1*xi*xgrd*ww+x1*yi*ygrd*ww;
		_jacobianOplusXi[7] = x2*xi*xgrd*ww+x2*yi*ygrd*ww;
		_jacobianOplusXi[8] = 1;
	}

	bool EdgeSL3::write(std::ostream& os) const
	{
		return os.good();
	}

	bool EdgeSL3::read(std::istream& is)
	{
		return true;
	}


