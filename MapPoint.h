//
// Created by lightol on 18-8-27.
//

#include <eigen3/Eigen/Eigen>

#ifndef ORB_SLAM2_LIGHTOL_MAPPOINT_H
#define ORB_SLAM2_LIGHTOL_MAPPOINT_H

class Map;
class Frame;
class KeyFrame;

class MapPoint
{
public:
    MapPoint() {}       // 注意不要在默认构造函数里面写mapPointNum_++，因为有时候系统构造零时对象，比如push_back的时候会调用默认构造函数，这个时候就多加了一个本来在物理世界中不存在的点

    // 当一个3D点被两帧三角化出来后，马上生成一个MapPoint对象并把它的first keyframe设为前一帧
    MapPoint(const Eigen::Vector3d &position, KeyFrame* pKF, Map* pMap);

    // 与上面一个构造函数的区别是还知道了这个MapPoint对应它的初始帧上的哪个2D点
    MapPoint(const Eigen::Vector3d &position, KeyFrame* pKF,int keypointID, Map* pMap);
    
    // 算出一个最佳的ORB描述子挂在这个MapPoint上，一个MapPoint可能被多个KeyFrame观测到，也就对应了多个keypoint，从这些关键点的ORB描述子中挑选出聚类中心来作为这个MapPoint的描述子
    void ComputeDistinctiveDescriptors();

public:
    long unsigned int id;                  // 这个MapPoint的id,按照生成的顺序依次往后加一
    static unsigned long numMapPoint;      // 目前总共生成了多少个mapPoint
    long int firstKFid;    // 第一个观测到这个mapPoint的KeyFrame的id
    KeyFrame* pFirstKF;    // 第一个观测到这个mapPoint的KeyFrame的指针，免得每次还要通过id查表获得keyframe的指针

private:
    Eigen::Vector3d worldPos;  // 这个mapPoint在世界坐标系下的xyz坐标

    Eigen::Matrix<unsigned char, 1, 32> OrbDescriptor;  // 这个MapPoint所对应的最好的那个2D点的ORB描述符

    Eigen::Vector3d viewVector;  // mean viewing direction, 所有观测到这个mapPoint的帧的视线的矢量和

    bool bBad;                // 在有些情况下已经生成的mapPoint可能被标记为bad

    int nVisible;           // 总共有多少个frame能观测到这个mapPoint

    Map* map;               // 全局地图的指针，便于通过这个指针去获得KeyFrame等对象

    std::map<KeyFrame*, int>    mObservers;   // 能够观测到这个MapPoint的Keyframe以及MapPoint在Keyframe上对应的keypoint在KeyFrame::keypoints中的索引
    int numObservers;         // 总共有多少个KeyFrame能观测到这个mapPoint





};

#endif //ORB_SLAM2_LIGHTOL_MAPPOINT_H
