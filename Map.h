#ifndef MAP_H
#define MAP_H
#include "KeyFrame.h"
#include <vector>
class Map
{
public:
	Map(){};
	static Map* getInstance();
	std::vector<KeyFrame*> allKeyFrame;
	std::vector<MapPoint*> allMapPoint;
	bool addKeyFrame(KeyFrame*);

private:
	static Map* instance;
};
#endif