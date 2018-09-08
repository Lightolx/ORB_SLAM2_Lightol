//
// Created by lightol on 18-8-27.
//

#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBmatcher.h"

unsigned long MapPoint::numMapPoint = 0;

MapPoint::MapPoint(const Eigen::Vector3d &position, KeyFrame *pKF, Map *pMap):
    worldPos(position), firstKFid(pKF->id), pFirstKF(pKF), map(pMap), bBad(false)
{
    // Step1: 每新建一个MapPoint，总的MapPoint数量加一，同时给这个MapPoint赋id
    id = numMapPoint++;

    // Step2: 确定由相机看到这个3D点的视线方向
    Eigen::Vector3d Oc = pKF->GetCameraCenter();
    viewVector = worldPos - Oc;  viewVector.normalize();
}

MapPoint::MapPoint(const Eigen::Vector3d &position, KeyFrame *pKF, int keypointID, Map *pMap):
        worldPos(position), firstKFid(pKF->id), pFirstKF(pKF), map(pMap), bBad(false)
{
    Eigen::Vector3d Oc = pKF->GetCameraCenter();
    viewVector = worldPos - Oc;  viewVector.normalize();

    // MapPoint的ORB描述子肯定是通过2D点得到的，关键是选择哪一个keyframe上对应的2D点的ORB描述子
    OrbDescriptor = pKF->vDescriptors[keypointID];

    id = numMapPoint++;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
    if (bBad)      // 坏点，也不用算了，反正其他地方也不用这个点参与BA等过程，也不会用到它的ORB描述子
    {
        return;
    }
    
    // Step1: 取出所有能观测到这个3D点的keyframe上对应的2D点的ORB描述子
    std::vector<Eigen::Matrix<unsigned char, 1, 32>> descriptors;
//    descriptors.reserve(mObservers.size());
    descriptors.reserve(numObservers);
    for (std::map<KeyFrame*, int>::iterator iter = mObservers.begin(); iter != mObservers.end(); iter++)
    {
        const KeyFrame* pKeyframe = iter->first;
        if (pKeyframe->isBad())
        {
            continue;
        }

        int keypointID = iter->second;
        descriptors.push_back(pKeyframe->vDescriptors[keypointID]);
    }

    if (descriptors.empty())
    {
        return;
    }

    // Step2: 计算任意两个描述子之间的ORB距离
    int N = descriptors.size(); // 这个descriptors.size()并不一定等于numObservers，因为有些keyframe可能是bad
    int distances[N][N];
    for (int i = 0; i < N; ++i)
    {
        for (int j = i+1; j < N; ++j)  // i = j的时候不用算，因为肯定为零
        {
            int OrbDistance = ORBmatcher::ComputeOrbDistance(descriptors[i], descriptors[j]);
            distances[i][j] = OrbDistance;
            distances[j][i] = OrbDistance;
        }
    }

    // Step3: 把这些ORB描述子作为观测值算出实际值，也就是使得所有观测数据的标准差最小的那个测量值
    std::vector<int> sumDistPerDesc(N,0);
    for (int i = 0; i < N; ++i)
    {
        // step3.1: 可以选平均值，但平均值容易受极端值的影响，如果有一个错误测量的值极大，那么会把整体也拉大
//        for (int j = 0; j < N; ++j)
//        {
//            sumDistPerDesc[i] += distances[i][j];
//        }

        //　step3.2: 中值不会受到极端值的影响，比较稳健
        std::vector<int> vDist(distances[i], distances[i]+N);
        std::sort(vDist.begin(), vDist.end());
        sumDistPerDesc[i] = vDist[N/2];
    }

    std::vector<int>::iterator iter = std::min_element(sumDistPerDesc.begin(), sumDistPerDesc.end());
    int minID = std::distance(sumDistPerDesc.begin(), iter);
    OrbDescriptor = descriptors[minID];
}
