cmake_minimum_required( VERSION 2.8 )
project( rkslam2 )
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  ${CMAKE_SOURCE_DIR}/cmake_modules/ )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

find_package(OpenCV REQUIRED)
find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS} 
${G2O_INCLUDE_DIRS}
${Eigen3_INCLUDE_DIRS}
)

add_executable(rkslam2 
main.cpp
tracking.cpp
sl3.cpp
sl3vertex.cpp
sl3edge.cpp
matcher.cpp
testsl3.cpp
map.cpp
ProjectionEdge.cpp)

if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
target_link_libraries(rkslam2 ${OpenCV_LIBRARIES}
${G2O_CORE_LIBRARY_DEBUG}
${G2O_SOLVER_EIGEN_DEBUG}
${G2O_STUFF_LIBRARY_DEBUG}
)
elseif(CMAKE_BUILD_TYPE EQUAL "RELEASE")
target_link_libraries(rkslam2 ${OpenCV_LIBRARIES} 
${G2O_CORE_LIBRARY}
${G2O_SOLVER_EIGEN}
${G2O_STUFF_LIBRARY}
)
elseif (WIN32)
target_link_libraries(rkslam2 ${OpenCV_LIBRARIES}
${G2O_CORE_LIBRARY_DEBUG}
${G2O_SOLVER_EIGEN_DEBUG}
${G2O_STUFF_LIBRARY_DEBUG}
)
else()
target_link_libraries(rkslam2 ${OpenCV_LIBRARIES}
${G2O_CORE_LIBRARY}
${G2O_SOLVER_EIGEN}
${G2O_STUFF_LIBRARY}
)
endif()


