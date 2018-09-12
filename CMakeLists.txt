cmake_minimum_required(VERSION 3.12)
project(ORB_SLAM2_Lightol)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(mono_tum main.cpp System.h System.cpp ORBVocalbulary.h MapPoint.h MapPoint.cpp ORBextractor.h ORBextractor.cpp ORBmatcher.h ORBmatcher.cpp KeyFrame.h Frame.h Frame.cpp Tracking.h Tracking.cpp Initializer.h Initializer.cpp)
target_link_libraries(mono_tum ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)

#add_executable(testORB testORB.cpp System.h System.cpp ORBVocalbulary.h MapPoint.h MapPoint.cpp ORBextractor.h ORBextractor.cpp ORBmatcher.h ORBmatcher.cpp KeyFrame.h Frame.h Frame.cpp Tracking.h Tracking.cpp Initializer.h Initializer.cpp)
#target_link_libraries(testORB ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)