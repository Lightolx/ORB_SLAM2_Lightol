//
// Created by lightol on 18-9-6.
//

#ifndef ORB_SLAM2_LIGHTOL_KEYFRAME_H
#define ORB_SLAM2_LIGHTOL_KEYFRAME_H

#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>

#include "MapPoint.h"

class KeyFrame
{
public:
    KeyFrame() {}

    Eigen::Vector3d GetCameraCenter() { return Oc; }

    int id;
    unsigned long numKFs;

    std::vector<cv::KeyPoint> keypoints;   // 这个Frame上提取出来的keypoint的集合
    std::vector<Eigen::Matrix<unsigned char, 1, 32>> vDescriptors;  // 上面那些keypoint的ORB描述符
    std::vector<MapPoint*> vMapPoints;    // 上面那些keypoint所对应的MapPoint，如果某个2D点没有三角化MapPoint，或者没有匹配到局部地图中已经存在的MapPoint，那么它在这个vector中的对应位置就为空指针

    bool isBad() const { return bBad; }

private:

    bool bBad;



    Eigen::Matrix4d Tcw;        // Tcw * Xw = Xc; 把世界坐标系下的点的坐标转换到相机坐标系
    Eigen::Matrix4d Twc;        // Twc = Tcw.inv(), 把相机坐标系下的点的坐标转换到世界坐标系，它的右上角Twc.topRightCorner(3,1) = Oc
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Vector3d Oc;         // 光心的位置

};

#endif //ORB_SLAM2_LIGHTOL_KEYFRAME_H
