#include "sl3vertex.h"

bool VertexSL3::read(std::istream& is)
{
	Vector8d data;
	for (int i = 0; i<8; i++){
		is >> data[i];
	}
	SL3 init;
	init.fromVector(data);
	setEstimate(init);
	return true;
}

bool VertexSL3::write(std::ostream& os) const
{
	SL3 data = estimate();
	Vector8d vec = data.toVector();
	for (int i = 0; i < 8; i++)
	{
		os << vec[i];
	}
	return os.good();
}

void VertexSL3::oplusImpl(const double *update)
{
	Eigen::Map<const Vector8d> v(update);
	SL3 upv;
	upv.fromVector(v);
	_estimate = _estimate*upv;
	if (++_numOplusCalls > orthogonalizeAfter)
	{
		_numOplusCalls = 0;
		_estimate.regularize();
	}
}