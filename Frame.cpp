//
// Created by lightol on 18-9-7.
//
#include <iostream>

#include <opencv/cv.hpp>
#include "Frame.h"
using std::cout;
using std::endl;

const int w = 10;        // 网格化后每一个小方块的边长,一般设定每个方块的size应该是10*10
const float wInv = 0.1;  // 边长的倒数，因为乘法比除法快

unsigned long Frame::numFrame = 0;
int Frame::grid_cols = 0;
int Frame::grid_rows = 0;
float Frame::fx = 0.0;
float Frame::fy = 0.0;
float Frame::cx = 0.0;
float Frame::cy = 0.0;
float Frame::invfx = 0.0;
float Frame::invfy = 0.0;
float Frame::minBorderX = 0.0;
float Frame::minBorderY = 0.0;
float Frame::maxBorderX = 0.0;
float Frame::maxBorderY = 0.0;

Frame::Frame(const cv::Mat &_image, const ORBextractor &_OrbExtractor, const cv::Mat &_K, const cv::Mat &_distCoef):
        OrbExtractor(_OrbExtractor), K(_K), distCoef(_distCoef)
{
    // Step1: 每新建一帧Frame，总的Frame数量加一，同时给这个Frame赋id
    id = numFrame++;

    // Step2: 建立图像网格,只在第一帧图像读进来的时候做一次
    if (numFrame == 1)
    {
        ConstructGrid(_image);

        // Step2.5: 设置相机内参，同样只做一次，因为对同一批照片要求它们是同一个相机拍出来的
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0/fx;
        invfy = 1.0/fy;
    }

    // Step3: 提取该帧上的rawKeypoints并计算其ORB描述子
    OrbExtractor(_image, rawKeypoints, descriptors);

    // Step4: 所有keypoint的坐标去畸变
    UndistortKeypints();

    // Step5: 将图像网格化，并将提取出来的keypoint分配到各个网格中，以加速ORB特征匹配过程
    AssignKeypointsToGrid();

    // Step6: 各个成员变量的初始化
    vMapPoints = std::vector<MapPoint*>(numKeypoints, nullptr);
}

void Frame::UndistortKeypints()
{
    numKeypoints = rawKeypoints.size();

    // Step0: 如果不能获得相机的去畸变参数，直接使用keypoint的原始坐标
    if (distCoef.at<float>(0,0) == 0.0)
    {
        keypoints = rawKeypoints;
        return;
    }
    
    // Step1: 利用OpenCV的内建函数去畸变
    // step1.1: 先把关键点的坐标表示成cv::Mat，一行一个(u,v)
    cv::Mat pts(numKeypoints,2,CV_32F);
    for (int i = 0; i < numKeypoints; ++i)
    {
        pts.at<float>(i,0) = rawKeypoints[i].pt.x;
        pts.at<float>(i,1) = rawKeypoints[i].pt.y;
    }

    // step1.2: 调用cv::undistortPoints()函数，专门校准一个点而不是校准整张图像
    pts.reshape(2);         // 为undistortPoints()函数做准备
    cv::undistortPoints(pts, pts, K, distCoef, cv::Mat(), K);
    pts.reshape(1);         //　还原

    // Step2: 去畸变后的坐标值赋给keypoints
    keypoints.resize(numKeypoints);
    for (int i = 0; i < numKeypoints; ++i)
    {
        cv::KeyPoint kpt = rawKeypoints[i];
        kpt.pt.x = pts.at<float>(i,0);
        kpt.pt.y = pts.at<float>(i,1);
        keypoints[i] = kpt;
    }
}

void Frame::ConstructGrid(const cv::Mat &image)
{
    // Step1:　去畸变四个角点，算出图像的边界
    if (distCoef.at<float>(0,0) != 0.0)
    {
        // 把4个角点去畸变，认定它们四个也是去畸变后的图像的角点
        cv::Mat pts(4,2,CV_32F);
        pts.at<float>(0,0) = 0;             pts.at<float>(0,1) = 0;
        pts.at<float>(1,0) = image.cols;    pts.at<float>(1,1) = 0;
        pts.at<float>(2,0) = 0;             pts.at<float>(2,1) = image.rows;
        pts.at<float>(3,0) = image.cols;    pts.at<float>(3,1) = image.rows;

        pts.reshape(2);
        cv::undistortPoints(pts, pts, K, distCoef, cv::Mat(), K);
        pts.reshape(1);

        minBorderX = std::min(pts.at<float>(0,0),pts.at<float>(2,0));
        minBorderY = std::min(pts.at<float>(0,1),pts.at<float>(1,1));
        maxBorderX = std::max(pts.at<float>(1,0),pts.at<float>(3,0));
        maxBorderY = std::max(pts.at<float>(2,1),pts.at<float>(3,1));
    }
    else
    {
        minBorderX = 0;
        minBorderY = 0;
        maxBorderX = image.cols;
        maxBorderY = image.rows;
    }

    // Step2: 确定网格的行数与列数
    grid_cols = ceil((maxBorderX - minBorderX) / float(w));
    grid_rows = ceil((maxBorderY - minBorderY) / float(w));

}

void Frame::AssignKeypointsToGrid()
{
    // Step1: 初始化grid的行数和列数
    grid.resize(grid_rows);
    for (std::vector<std::vector<int>> &row: grid)
    {
        row.resize(grid_cols);
    }

    // Step2: 为每一个格子内的vector都reserve足够的空间
    int nKeypointPerGrid = std::ceil(float(numKeypoints) / (grid_cols*grid_rows));
    for (int i = 0; i < grid_rows; ++i)
    {
        for (int j = 0; j < grid_cols; ++j)
        {
            grid[i][j].clear();
            grid[i][j].reserve(nKeypointPerGrid);
        }
    }

    // Step3: 将当前帧的关键点分配到各个格子中
    for (int i = 0; i < numKeypoints; ++i)
    {
        const cv::KeyPoint &kpt = keypoints[i];
        float x = kpt.pt.x;
        float y = kpt.pt.y;
        if (x < minBorderX || x > maxBorderX || y < minBorderY || y > maxBorderY)
        {
            continue;
        }

        int m = std::floor((y - minBorderY) * wInv);  // 第m行, 这里用floor而不是round，因为0.9你也要归属到第0行，否则就在这里多造出来了一行
        int n = std::floor((x - minBorderX) * wInv);  // 第n列
        grid[m][n].push_back(i);
    }
}

std::vector<int> Frame::GetKptsInArea(Eigen::Vector2f p, float r) const
{
    // Step1: 首先得到这个区域包含哪些grid
    int minX = std::max((int)std::floor((p.x() - r - minBorderX) * wInv), 0);
    int maxX = std::min((int)std::floor((p.x() + r - minBorderX) * wInv), grid_cols-1);  // 之所以用floor是因为数组是从0开始计数的
    int minY = std::max((int)std::floor((p.y() - r - minBorderY) * wInv), 0);
    int maxY = std::min((int)std::floor((p.y() + r - minBorderY) * wInv), grid_rows-1);

    // Step2: 从这些网格中挑选出到点p的距离小于r的keypoint，因为范围是个圆，而我们在Step1中得到的是个矩形，相当于最小包围盒
    std::vector<int> kptIDs;
    kptIDs.reserve(numKeypoints/10);
    for (int m = minY; m <= maxY; ++m)      // 第m行
    {
        for (int n = minX; n <= maxX; ++n)  // 第n列
        {
            std::vector<int> ids = grid[m][n];

//            for (int id: ids)
//            {
//                cv::KeyPoint kpt = keypoints[id];
//                Eigen::Vector2f pt(kpt.pt.x, kpt.pt.y);
//
//                // note: 在这里原代码用的是街区距离，但我觉得你要用街区距离还不如把所有的grid都压进去算了
//                // todo:: 我觉得把所有grid里面的keypoint都压进去也没什么不行,因为每个格子也就１０个像素大小
//                if ((pt - p).norm() < r)
//                {
//                    kptIDs.push_back(id);
//                }
//            }
            kptIDs.insert(kptIDs.end(), ids.begin(), ids.end());
        }
    }

    return kptIDs;
}
