//
// Created by lightol on 18-9-4.
//

#ifndef ORB_SLAM2_LIGHTOL_ORBMATCHER_H
#define ORB_SLAM2_LIGHTOL_ORBMATCHER_H

#include <eigen3/Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include "Frame.h"

class ORBmatcher
{
public:
    ORBmatcher(float nnratio = 0.6, bool checkOri = true);

    // todo:: 这里还有加速的余地
    static int ComputeOrbDistance(Eigen::Matrix<unsigned char, 1, 32> vector1,
                                  Eigen::Matrix<unsigned char, 1, 32> vector2);

    static std::vector<int> MatchDescriptors(Eigen::Matrix<unsigned char, Eigen::Dynamic, 32> descriptors1,
                                                  Eigen::Matrix<unsigned char, Eigen::Dynamic, 32> descriptors2);

    // 找寻两幅图中的匹配点对并保存在matches中，first与second分别表示匹配的一对keypoint1与keypoint2在image1与image2中的id
    int FindMatchingPoints(const Frame &image1, const Frame &image2, int windowSize,
                           std::map<int, int> &matches) const;
private:
    float NNratio;
    bool bCheckOrientation;
    static const int TH_HAMMING_DIST = 10;
};


#endif //ORB_SLAM2_LIGHTOL_ORBMATCHER_H

