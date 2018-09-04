//
// Created by lightol on 18-8-27.
//

#include <eigen3/Eigen/Eigen>

#ifndef ORB_SLAM2_LIGHTOL_MAPPOINT_H
#define ORB_SLAM2_LIGHTOL_MAPPOINT_H

class MapPoint
{
public:
    MapPoint();

public:
    long unsigned int id_;                  // 这个MapPoint的id,按照生成的顺序依次往后加一
    static long unsigned mapPointNum_;      // 目前总共生成了多少个mapPoint
    long int firstKFid_;
    int obsNum_;            // 能观测到这个mapPoint的keyframe的个数

private:
    Eigen::Vector3d worldPos_;  // 这个mapPoint在世界坐标系下的xyz坐标

    // mean viewing direction, 所有观测到这个mapPoint的帧的视线的矢量和
    Eigen::Vector3d views_;

    bool isBad_;                // 在有些情况下已经生成的mapPoint可能被标记为bad





};

#endif //ORB_SLAM2_LIGHTOL_MAPPOINT_H
