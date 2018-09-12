//
// Created by lightol on 18-8-23.
//

#ifndef ORB_SLAM2_LIGHTOL_SYSTEM_H
#define ORB_SLAM2_LIGHTOL_SYSTEM_H

#include <string>
#include <iostream>

#include "ORBVocalbulary.h"
#include "Tracking.h"

using std::cout;
using std::endl;
using std::cerr;

class System
{
public:

    System(const std::string &vocFilepath, const std::string &camSetFilepath);
    Eigen::Matrix4d TrackMonocular(const cv::Mat& image) const;

private:
    ORB_Vocalbulary* vocabulary_;
    Tracking* pTracker;


};

#endif //ORB_SLAM2_LIGHTOL_SYSTEM_H
