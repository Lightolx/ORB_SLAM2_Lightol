//
// Created by lightol on 18-9-4.
//

#ifndef ORB_SLAM2_LIGHTOL_ORBMATCHER_H
#define ORB_SLAM2_LIGHTOL_ORBMATCHER_H

#include <eigen3/Eigen/Eigen>

class ORBmatcher
{
public:
    ORBmatcher(float nnratio = 0.6, bool checkOri = true);

    static int ComputeOrbDistance(Eigen::Matrix<unsigned char, 1, 32> vector1,
                                  Eigen::Matrix<unsigned char, 1, 32> vector2);

    static std::vector<int> MatchDescriptors(Eigen::Matrix<unsigned char, Eigen::Dynamic, 32> descriptors1,
                                                  Eigen::Matrix<unsigned char, Eigen::Dynamic, 32> descriptors2);
private:
    float NNratio;
    bool bCheckOrientation;
    static const int TH_HAMMING_DIST = 30;
};


#endif //ORB_SLAM2_LIGHTOL_ORBMATCHER_H

