#ifndef MAP_H
#define MAP_H
#include "KeyFrame.h"
#include <vector>
class Map
{
public:
		Map();
		static Map* instance;
		std::vector<KeyFrame*> allKeyFrame;
};
#endif