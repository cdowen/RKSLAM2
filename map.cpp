#include "Map.h"
Map* Map::instance;
Map::Map()
{
	Map::instance = this;
}
