//
// Created by lightol on 18-9-6.
//

#ifndef ORB_SLAM2_LIGHTOL_FRAME_H
#define ORB_SLAM2_LIGHTOL_FRAME_H

#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>

#include "MapPoint.h"
#include "ORBmatcher.h"
#include "ORBextractor.h"

// todo:: 把很多字段， 比如K等换成静态成员变量，因为可以认为一批图像它们的内参都是一样的，免得反复计算浪费时间

class Frame
{
public:
    Frame() {}
    // Constructor for Monocular cameras
    Frame(const cv::Mat &image, const ORBextractor &OrbExtractor, const cv::Mat &K, const cv::Mat &distCoef);


    static unsigned long numFrame;
    static int grid_cols;
    static int grid_rows;
    int id;

    cv::Mat K;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat distCoef;

    static float minBorderX;
    static float minBorderY;
    static float maxBorderX;
    static float maxBorderY;

    std::vector<std::vector<std::vector<int>>> grid;  // 二维数组的每一个元素包含了在这个网格内的keypoint的id序列，注意是先cols再rows

//    cv::Mat image;　　　　　　　　　　图像都不用存，因为要用到它灰度的地方就是提特征点，在后面的slam过程中用不到它
    ORBextractor OrbExtractor;

    std::vector<cv::KeyPoint> rawKeypoints;   // 这个Frame上提取出来的keypoint的集合，其坐标还没经过去畸变
    std::vector<cv::KeyPoint> keypoints;      // 坐标去畸变后的keypoints
    Eigen::Matrix<uchar, Eigen::Dynamic, 32> descriptors;  // 上面那些keypoint的ORB描述符
    std::vector<MapPoint*> vMapPoints;    // 上面那些keypoint所对应的MapPoint，如果某个2D点没有三角化MapPoint，或者没有匹配到局部地图中已经存在的MapPoint，那么它在这个vector中的对应位置就为空指针
    int numKeypoints;                     // 这张图上提取到的keypoint的数量

    // 在图中找出到2D点O的距离小于r的keypoint的ID
    std::vector<int> GetKptsInArea(Eigen::Vector2f p, float r) const;

private:
    // 对提取的rawKeypoints的坐标去畸变
    void UndistortKeypints();

    // 把图像栅格化成一个个方块，计算出每个方块的宽度以及网格的行列数
    void ConstructGrid(const cv::Mat &image);





    Eigen::Matrix4d Tcw;        // Tcw * Xw = Xc; 把世界坐标系下的点的坐标转换到相机坐标系
    Eigen::Matrix4d Twc;        // Twc = Tcw.inv(), 把相机坐标系下的点的坐标转换到世界坐标系，它的右上角Twc.topRightCorner(3,1) = Oc
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Vector3d Oc;         // 光心的位置
    void AssignKeypointsToGrid();

};

#endif //ORB_SLAM2_LIGHTOL_FRAME_H
