//
// Created by lightol on 18-9-7.
//

#ifndef ORB_SLAM2_LIGHTOL_INITIALIZER_H
#define ORB_SLAM2_LIGHTOL_INITIALIZER_H

#include <vector>

#include <opencv2/core/core.hpp>

class Initializer
{
public:
    Initializer();

    std::vector<cv::KeyPoint> keypoints1;   // 初始帧上的keypoint
    std::vector<cv::KeyPoint> keypoints2;   // 第二帧(当前帧)上的keypoint

    bool bIniFrameCreated;
};

#endif //ORB_SLAM2_LIGHTOL_INITIALIZER_H
