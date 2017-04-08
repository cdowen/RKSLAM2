#ifndef SL3VERTEX_H
#define SL3VERTEX_H
#include "sl3.h"
#include <g2o/core/base_vertex.h>
class VertexSL3 :public g2o::BaseVertex<8, SL3>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		VertexSL3(){_numOplusCalls=0; }
	static const int orthogonalizeAfter = 1000; 
	inline virtual void setToOriginImpl()
	{
		_estimate = SL3();
	}

	virtual void oplusImpl(const double* update);

	virtual bool read(std::istream& is);
	Vector8d read();
	virtual bool write(std::ostream& os) const;
protected:
	int _numOplusCalls;
};
#endif