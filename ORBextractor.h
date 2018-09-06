//
// Created by lightol on 18-8-27.
//

#ifndef ORB_SLAM2_LIGHTOL_ORBEXTRACTOR_H
#define ORB_SLAM2_LIGHTOL_ORBEXTRACTOR_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Eigen>

namespace ORB_SLAM
{
class Block
{
public:
    Block();
    void DivideBlock(Block &block1, Block &block2, Block &block3, Block &block4);

    std::vector<cv::KeyPoint> keypoints;
    Eigen::Vector2i UL, UR, BL, BR;
    static unsigned int numBlock;
    unsigned int id;

    // 只包含一个keypoint，不能再分裂了
    bool bNoMore;  // 当然你会问包含零个的block怎么处理，很简单，一旦在集合中发现有这样的block马上把它删掉

    // Block对象生成后都会被放入到一个list中，在这里记录它在list中的位置
    std::list<Block>::iterator position;

};
}

class ORBextractor
{

public:
    ORBextractor(int nFeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    // 输入一幅图像，提取它上面的关键点并计算所有关键点的ORB特征保存在descriptors中
    void operator()(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, Eigen::Matrix<uchar, Eigen::Dynamic, 32> &descriptors);

private:
    // 建立层数为8的图像金字塔，越往上该层图像的像素数越少
    void ConstructPyramid(const cv::Mat &image);

    //　在每层金字塔上都提取出足够的FAST角点，作为计算BRIEF描述子的关键点
    void ExtractKeypoint(std::vector<std::vector<cv::KeyPoint>> &vvKeypoints) const;

    // 将待分配的keypoints分布到四叉树中，保证每个子块中只有一个keypoint
    std::vector<cv::KeyPoint> DistributeKpts(const std::vector<cv::KeyPoint> &keypointsToDistribute,
                                             int minX, int maxX, int minY, int maxY, int nFeatures) const;

    //　计算出每个关键点所在patch的灰度重心，连接该关键点和该灰度重心，作为该patch的x轴方向，建立该关键点附近的局部坐标系
    void ComputeKeypointDirection(std::vector<std::vector<cv::KeyPoint>> &vvKeypoints) const;

    // 计算该keypoint的ORB描述子, 并保存在行首地址为ptr的cv::Mat的对应行中, 一行32个字节，保存256个bit
    static Eigen::Matrix<uchar, 1, 32> ComputeOrbDescriptor(const cv::KeyPoint &kpt, const cv::Mat &image, const cv::Point2i *BRIEF_bit);

    int nlevels;        // 建立多少层金字塔，代码中为８
    std::vector<cv::Mat> imagePyramids;     //　size为8的vector，每一个元素对应金字塔每一层的图像

    int nFeatures;      // 8层图像加起来，总共想提取多少个keypoint
    std::vector<int> nFeaturesPerLevel;     //　每一层应该提取多少个keypoint，与总数nFeatures成正比

    int iniThFAST;      // 提取FAST角点时，有iniThFAST个连续的点的灰度值都与该点差异超过阈值，则判定为角点，代码中为20
    int minThFAST;      // 有些地方没有那么强的角点，但又必须提取一些角点，只能降低阈值了，代码中为7

    float scaleFactor;  // 金字塔中每往上一层，cols和rows的放缩系数系数，代码中为1.2，实际效果是缩小为原来的1/1.2,大概为0.83
    std::vector<float> scaleFactors;        // size为8的vector，存储了每一层图像相对于原始图像的缩小系数
    std::vector<float> invScaleFactors;     // scaleFactors的倒数，免得在代码中每次用到都要再算一遍
    std::vector<float> sigma2s;             // scaleFactors的平方，也是为了免得每次用到都要再算一遍
    std::vector<float> invsigma2s;          // sigma2s的倒数

    std::vector<Eigen::Vector2i> BRIEFs;  // BRIEF算子用于生成0或1 bit位的点对

    std::vector<int> nPixelPerRow;        // 计算一块patch的灰度质心时，每一行应该有多少个像素，这样patch恰好是个圆
};


#endif //ORB_SLAM2_LIGHTOL_ORBEXTRACTOR_H
