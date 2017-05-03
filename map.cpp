#include "Map.h"
Map* Map::instance;

Map* Map::getInstance()
{
	if (Map::instance == nullptr)
	{
		Map::instance = new Map();
	}
	return Map::instance;
}
bool Map::addKeyFrame(KeyFrame *kf)
{
	if (kf!= nullptr)
	{
		allKeyFrame.push_back(kf);
	}
}