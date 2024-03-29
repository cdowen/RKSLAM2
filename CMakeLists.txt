cmake_minimum_required( VERSION 2.8 )
project( rkslam2 )
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}  ${CMAKE_SOURCE_DIR}/cmake_modules/ )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

find_package(CSparse REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Eigen3 3.2.10 REQUIRED)
find_package (Threads)
find_package(PCL 1.3 REQUIRED)
find_package(vikit_common REQUIRED)
find_package(Sophus REQUIRED)
message(STATUS "OpenCV_VERSION = ${OpenCV_VERSION}")
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

include_directories(${OpenCV_INCLUDE_DIRS}
${G2O_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${CSPARSE_INCLUDE_DIR}
${vikit_common_INCLUDE_DIR}
${PCL_INCLUDE_DIRS}
${Sophus_INCLUDE_DIR}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(rkslam2
main.cpp
tracking.cpp
sl3vertex.cpp
sl3edge.cpp
matcher.cpp
testsl3.cpp
map.cpp
Initialization.cpp
ProjectionEdge.cpp
LocalMap.cpp Optimizer.cpp Optimizer.h)

if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
target_link_libraries(rkslam2
        ${OpenCV_LIBRARIES}
        ${CSPARSE_LIBRARY}
        ${G2O_CORE_LIBRARY_DEBUG}
        ${G2O_SOLVER_EIGEN_DEBUG}
        ${G2O_STUFF_LIBRARY_DEBUG}
        ${G2O_SOLVER_CSPARSE_DEBUG}
        ${G2O_TYPES_SBA}
        ${G2O_SOLVER_CSPARSE_EXTENSION}
        ${CMAKE_THREAD_LIBS_INIT}
        ${vikit_common_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Sophus_LIBRARIES}
)
elseif(CMAKE_BUILD_TYPE EQUAL "RELEASE")
target_link_libraries(rkslam2
        ${OpenCV_LIBRARIES}
        ${CSPARSE_LIBRARY}
        ${G2O_CORE_LIBRARY}
        ${G2O_SOLVER_EIGEN}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_CSPARSE}
        ${G2O_TYPES_SBA}
        ${G2O_SOLVER_CSPARSE_EXTENSION}
        ${CMAKE_THREAD_LIBS_INIT}
        ${vikit_common_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Sophus_LIBRARIES}
)
elseif (WIN32)
target_link_libraries(rkslam2
        ${OpenCV_LIBRARIES}
        ${CSPARSE_LIBRARY}
        ${G2O_CORE_LIBRARY_DEBUG}
        ${G2O_SOLVER_EIGEN_DEBUG}
        ${G2O_STUFF_LIBRARY_DEBUG}
        ${G2O_SOLVER_CSPARSE_DEBUG}
        ${G2O_SOLVER_CSPARSE_EXTENSION}
        ${G2O_TYPES_SBA}
        ${vikit_common_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Sophus_LIBRARIES}
)
else()
target_link_libraries(rkslam2
        ${OpenCV_LIBRARIES}
        ${CSPARSE_LIBRARY}
        ${G2O_CORE_LIBRARY}
        ${G2O_SOLVER_EIGEN}
        ${G2O_STUFF_LIBRARY}
        ${G2O_SOLVER_CSPARSE}
        ${G2O_TYPES_SBA}
        ${G2O_SOLVER_CSPARSE_EXTENSION}
        ${CMAKE_THREAD_LIBS_INIT}
        ${vikit_common_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Sophus_LIBRARIES}
)
endif()
