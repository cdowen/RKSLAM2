#include "sl3.h"
#include <iostream>
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