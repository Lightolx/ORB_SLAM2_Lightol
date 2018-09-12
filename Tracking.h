//
// Created by lightol on 18-9-7.
//

#ifndef ORB_SLAM2_LIGHTOL_TRACKING_H
#define ORB_SLAM2_LIGHTOL_TRACKING_H

#include <string>

#include <opencv2/core/core.hpp>
#include "ORBextractor.h"
#include "Frame.h"
#include "Initializer.h"

class Tracking
{
public:
    enum TrackingState
    {
        NO_IMAGE_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    TrackingState trackingState;

    Eigen::Matrix4d Run(const cv::Mat &image);

    // 因为一次track是对相机pose的track，所以需要知道相机的内参
    Tracking(const std::string &camSetFilepath);

    ORBextractor OrbExtractor;
    ORBextractor iniOrbExtractor;

    // 把输入的图片转换为Frame对象，完成提取keypoint以及计算ORB特征等工作
    void PreProcessImage(cv::Mat image);
    void track();

    Frame currentFrame; // 这个Frame用指针应该要好点，毕竟一个Frame对象包含了很多成员

    Initializer initializer;

private:
    cv::Mat K;
    cv::Mat distortCoef;
    int fps;
    int bRGB;  // 这个相机拍出来的图片的颜色通道顺序，是不是BRG
    void MonoInitialize();
};

#endif //ORB_SLAM2_LIGHTOL_TRACKING_H
