#include "sl3edge.h"

bool validateProjection(Eigen::Vector3d hLoc, cv::Size s)
{
	if (hLoc[0] >= 0 && hLoc[0] < s.width&&hLoc[1]>=0 && hLoc[1] < s.height)
	{
		return true;
	}
	return false;
}

Eigen::Vector3d EdgeSL3::homo_project()
	{
		Eigen::Vector3d hLoc;
		hLoc << loc[0], loc[1], 1;
		const VertexSL3* v1 = static_cast<const VertexSL3*>(_vertices[0]);
	
		hLoc = v1->estimate()._mat*hLoc;
		hLoc = hLoc / hLoc[2];
		return hLoc;
	}
	void EdgeSL3::computeError()
	{
		Eigen::Vector3d hLoc = homo_project();
		if (!validateProjection(hLoc, image->size()))
		{
			isValid = false;
		    _error[0] = 0;			
		    return;
		}
		isValid = true;
		unsigned char data= image->at<unsigned char>(hLoc[1], hLoc[0]);
		_error[0] = _measurement - data;
	}

	void EdgeSL3::linearizeOplus()
	{
		Eigen::Vector3d hLoc = homo_project();
		if (!validateProjection(hLoc, image->size()))
		{
			for (int i = 0; i < 8; i++)
			{
				_jacobianOplusXi[i] = 0;
			}
			isValid = false;
		    return;
		}
		float xgrd = xgradient->at<float>(hLoc[1], hLoc[0]);
		float ygrd = ygradient->at<float>(hLoc[1], hLoc[0]);
		_jacobianOplusXi[0] = -xgrd;
		_jacobianOplusXi[1] = -ygrd;
		_jacobianOplusXi[2] = -loc[1] * xgrd;
		_jacobianOplusXi[3] = -loc[0] * ygrd;
		_jacobianOplusXi[4] = -loc[0] * xgrd + loc[1] * ygrd;
		_jacobianOplusXi[5] = loc[0] * xgrd + 2 * loc[1] * ygrd;
		_jacobianOplusXi[6] = loc[0] * loc[0] * xgrd + loc[0] * loc[1] * ygrd;
		_jacobianOplusXi[7] = loc[0] * loc[1] * xgrd + loc[1] * loc[1] * ygrd;
		std::vector<double> debugN;
		isValid = true;
		for (int i = 0; i < 8; i++)
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