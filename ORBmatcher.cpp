//
// Created by lightol on 18-9-4.
//

#include "ORBmatcher.h"

ORBmatcher::ORBmatcher(float nnratio, bool checkOri):NNratio(nnratio), bCheckOrientation(checkOri)
{

}

int ORBmatcher::ComputeOrbDistance(Eigen::Matrix<unsigned char, 1, 32> vector1,
                                   Eigen::Matrix<unsigned char, 1, 32> vector2)
{
    int numBit = 0;

    for (int i = 0; i < 32; ++i)
    {
        unsigned char c = vector1[i]^vector2[i];
        c = c - ((c >> 1) & 0125);               // 01010101
        c = (unsigned char)(c & 063) + (unsigned char)((c >> 2) & 063);   // 00110011
        c = (unsigned char)(c*021) >> 4;                                  // 11110000
        numBit += c;
    }

    return numBit;
}

std::vector<int> ORBmatcher::MatchDescriptors(Eigen::Matrix<unsigned char, Eigen::Dynamic, 32> descriptors1,
                                              Eigen::Matrix<unsigned char, Eigen::Dynamic, 32> descriptors2)
{
    int numDes1 = descriptors1.rows();
    int numDes2 = descriptors2.rows();
    std::vector<int> result(numDes1, -1);

    std::vector<int> OrbDistances;
    for (int i = 0; i < numDes1; ++i)
    {
        Eigen::Matrix<unsigned char, 1, 32> descriptor1 = descriptors1.row(i);
        OrbDistances.clear();
        OrbDistances.resize(numDes2);
        for (int j = 0; j < numDes2; ++j)
        {
            OrbDistances[j] = ComputeOrbDistance(descriptor1, descriptors2.row(j));
        }

        std::vector<int>::iterator iter = std::min_element(OrbDistances.begin(), OrbDistances.end());

        if (*iter < TH_HAMMING_DIST)
        {
            int minId = std::distance(OrbDistances.begin(), iter);
            result[i] = minId;
        }
    }

    return result;

}
