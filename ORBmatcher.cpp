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

int ORBmatcher::FindMatchingPoints(const Frame &image1, const Frame &image2, int windowSize,
                                   std::map<int, int> &matches) const
{
    std::vector<int> matches21(image2.numKeypoints, 256);  //　用于记录image2中各个keypoint与image1的距离

    for (int i = 0; i < image1.numKeypoints; ++i)
    {
        const cv::KeyPoint &kpt1 = image1.keypoints[i];
//        if (kpt1.octave)  //　todo:: 初始化时只在原图像上找对应点, 我觉得这一条不合理
//        {
//            continue;
//        }

        // Step1: 小位移假设，假设与iamge1上keypoint1(u,v)对应的keypoint2在image2中也在相同的位置，也就是(u,v)附近
        std::vector<int> ids = image2.GetKptsInArea(Eigen::Vector2f(kpt1.pt.x, kpt1.pt.y), windowSize);

        if (ids.size() < 2)
        {
            continue;
        }

        // Step2: 对于上一步求出来的候选匹配点，逐个计算其与kpt1的ORB距离并返回最近的那一个
        Eigen::Matrix<uchar, 1, 32> descriptor1 = image1.descriptors.row(i);
        Eigen::Matrix<uchar, 1, 32> descriptor2;
        std::vector<std::pair<int,int>> id_dist; // first是keypoint1与keypoint2的ORB距离，second是keypoint2的id
        int dist;
        for (int id: ids)
        {
            descriptor2 = image2.descriptors.row(id);
            dist = ORBmatcher::ComputeOrbDistance(descriptor1, descriptor2);
            id_dist.push_back(std::make_pair(dist, id));
        }

        // Step3: 选出最小和第二小的两个keypoint2, 并根据一定条件确定是否认定该点是keypoint1的匹配点
        std::sort(id_dist.begin(), id_dist.end());
        int minDist = id_dist[0].first;
        int minID = id_dist[0].second;
        int minDist2 = id_dist[1].first;

        // step3.1: 很普通啦，要成为匹配点对的话两个keypoints的ORB距离肯定要小于一定阈值
        if (minDist < TH_HAMMING_DIST)
        {
            // step3.2: 最小的距离要足够优秀，木秀于林，一堆大的就它一个小，那么它就很可能就是唯一那个正确的匹配，而不是矮子里面拔将军选出来的平平无奇的一个
            if (minDist < NNratio*minDist2)
            {
                // 如果ID为minID的keypoint2已经和以前的一个keypoint1匹配上了，那么就看两者的ORB距离了，如果现在的这个距离比以前的小，那么可以认定现在的匹配更准确，否则认为原来的更准确，现在这个点没办法正确匹配了
                if (dist < matches21[minID])
                {
                    matches[i] = minID;
                }
            }
        }
    }

    return matches.size();
}
