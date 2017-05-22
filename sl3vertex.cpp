#include "sl3vertex.h"

bool VertexSL3::read(std::istream& is)
{
	Eigen::Matrix3d data;
	for (int i = 0; i<9; i++){
		is>>data(i);
	}

	setEstimate(data);
	return true;
}

bool VertexSL3::write(std::ostream& os) const
{
	Eigen::Matrix3d data = estimate();
	for (int i = 0; i < 9; i++)
	{
		os << data(i);
	}
	return os.good();
}

void VertexSL3::oplusImpl(const double *update)
{
	Eigen::Matrix3d upv;
	for (int i = 0;i<8;i++)
	{
		upv(i/3, i%3) = update[i];
	}
	upv(2,2) = 0;
	_estimate = _estimate+upv;
}