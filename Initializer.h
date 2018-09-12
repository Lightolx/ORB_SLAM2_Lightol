//
// Created by lightol on 18-9-7.
//

#ifndef ORB_SLAM2_LIGHTOL_INITIALIZER_H
#define ORB_SLAM2_LIGHTOL_INITIALIZER_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <map>
#include "Frame.h"

class Initializer
{
public:
    Initializer(int maxIterations = 200);

    // 根据ORBmatcher输出的匹配点对,计算当前帧相对于初始帧的pose
    bool ComputeRelativePose(const std::map<int, int> &matches, Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<Eigen::Vector3d> &vMapPoints, std::vector<std::pair<int, int>> &MpMatches);

    void ComputeHomography(float &score, Eigen::Matrix3d &H) const;
    void ComputeFundamental(float &score, Eigen::Matrix3d &F);

    bool ReconstructF(const Eigen::Matrix3d &F, Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<Eigen::Vector3d> &vMapPoints, std::vector<std::pair<int, int>> &MpMatches);
    void SvdEssential(const Eigen::Matrix3d &E, Eigen::Matrix3d &R1, Eigen::Matrix3d &R2, Eigen::Vector3d &t) const;
    int SelectRt(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Matrix3d &K, std::vector<Eigen::Vector3d> &mapPoints, std::vector<std::pair<int, int>> &validMatches, double &parallax) const;
    Eigen::Vector3d triangulate(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Matrix3d &K) const;

    // 计算Frame2相对于Frame1的F矩阵
    Eigen::Matrix3d ComputeF(const std::vector<Eigen::Vector2d> &kpts1, const std::vector<Eigen::Vector2d> &kpts2) const;

    // 通过F求出Frame1上的关键点p1在Frame2上的对极线l2，再求出与p1匹配的点p2到l1的距离，以此来作为重投影误差衡量算出的F的好坏，当然，需要把所有的匹配点对的重投影误差加起来，同时也要把点p2重投影回Frame1计算其重投影误差
    double EvaluateF(const Eigen::Matrix3d &F, double sigma, std::vector<bool> &bValidMatch);
    double EvaluateF(const Eigen::Matrix3d &F, std::vector<int> octaMatch) const;

    // 转换成标准分布，便于之后计算方便，类似于2D平面上分布的归一化
    void NormalizeKeypoints();
    void NormalizeKptOnFrame(const Frame &frame, std::vector<Eigen::Vector2d> &kpts, Eigen::Matrix3d &T) const;
    Eigen::Matrix3d T1, T2;   // 从标准分布转回原来分布的转换矩阵

    Frame initialFrame;
    Frame currentFrame;

    std::vector<Eigen::Vector2d> keypoints1;   // 初始帧上的keypoint
    std::vector<Eigen::Vector2d> keypoints2;   // 第二帧(当前帧)上的keypoint
    std::vector<std::pair<int, int>> matches;  // 上面两帧上keypints的对应关系
    std::vector<bool> bValidMatch;             // 上面这些匹配中经过RANSAC判定为正确的匹配
    std::vector<std::vector<int>> octaMatches;        // 每一行存储8个匹配对，也就是每一行的匹配对都能拿出来做一次8点法

    bool bIniFrameCreated;
    int frameCount;

    int maxIterations;
};

#endif //ORB_SLAM2_LIGHTOL_INITIALIZER_H
