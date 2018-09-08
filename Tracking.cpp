//
// Created by lightol on 18-9-7.
//

#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv/cv.hpp>

#include "Tracking.h"
#include "Initializer.h"
#include "ORBmatcher.h"

Tracking::Tracking(const std::string &camSettingsFilepath):trackingState(NO_IMAGE_YET)
{
    cv::FileStorage fSettings(camSettingsFilepath, cv::FileStorage::READ);

    // Step1: 读取相机的内参
    // step1.1: 相机内参
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    cv::Mat K1 = cv::Mat::eye(3,3,CV_32F);
    K1.at<float>(0,0) = fx;
    K1.at<float>(1,1) = fy;
    K1.at<float>(0,2) = cx;
    K1.at<float>(1,2) = cy;
    K = K1.clone();         // 如果这里用浅复制那么这个函数结束后临时变量K1就会被销毁，它所在的内存区域也会被释放

    // step1.2: 去畸变参数
    cv::Mat distortCoef1 = cv::Mat(4,1,CV_32F);
    distortCoef1.at<float>(0) = fSettings["Camera.k1"];  // 径向畸变
    distortCoef1.at<float>(1) = fSettings["Camera.k2"];
    distortCoef1.at<float>(2) = fSettings["Camera.p1"];  // 切向畸变
    distortCoef1.at<float>(3) = fSettings["Camera.p2"];
    float k3 = fSettings["Camera.k3"];  // 鱼眼镜头有第三个去径向畸变参数
    if (k3 != 0.0)
    {
        distortCoef1.resize(5);
        distortCoef1.at<float>(4) = k3;
    }
    distortCoef = distortCoef1.clone();

    // step1.3: 相机帧数
    fps = fSettings["Camera.fps"];
    if (fps == 0)
    {
        fps = 30;
    }

    // step1.4: 颜色通道的存储顺序，RGB还是BGR，因为OpenCV内部使用BGR，所以这个顺序很重要
    bRGB = fSettings["Camera.RGB"];  // 不让使用bool型

    // Step2: 读取ORB描述子的参数设置，比如一幅图像提多少个点，FAST的阈值等等，对一次track这些当然要是一样的，要不然对前后两幅不一样的参数提出来的ORB描述子，你怎么匹配？
    int nFeatures = fSettings["ORBextractor.nFeatures"];  // 每幅图像提多少个keypoint
    float scaleFactor = fSettings["ORBextractor.scaleFactor"]; //　图像金字塔的缩放系数
    int nLevels = fSettings["ORBextractor.nLevels"];
    int iniThFAST = fSettings["ORBextractor.iniThFAST"];  // FAST提角点的初始阈值
    int minThFAST = fSettings["ORBextractor.minThFAST"];  // 初始阈值提不到FAST点时，降低阈值，再提不到就没办法了

    OrbExtractor = ORBextractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
    iniOrbExtractor = ORBextractor(2*nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
}

void Tracking::track()
{
    if (trackingState == NO_IMAGE_YET)
    {
        trackingState = NOT_INITIALIZED;
    }

    if (trackingState == NOT_INITIALIZED)
    {
        MonoInitialize();
    }
}

void Tracking::PreProcessImage(const cv::Mat &image)
{
    // step1: 检查输入的图片．如果不是灰度图就转换成灰度图
    currentImage = image.clone();

    if (currentImage.channels() == 3)
    {
        if (bRGB)
        {
            cv::cvtColor(currentImage, currentImage, CV_RGB2GRAY);
        }
        else
        {
            cv::cvtColor(currentImage, currentImage, CV_BGR2GRAY);
        }
    }
    else if (currentImage.channels() == 4)
    {
        if (bRGB)
        {
            cv::cvtColor(currentImage, currentImage, CV_RGBA2GRAY);
        }
        else
        {
            cv::cvtColor(currentImage, currentImage, CV_BGRA2GRAY);
        }
    }

    // step2: 如果是初始化则需要提取2000个关键点，如果是后续的追踪过程则只提取1000个就行
    if ( trackingState == NO_IMAGE_YET || trackingState == NOT_INITIALIZED)
    {
        currentFrame = Frame(currentImage, iniOrbExtractor, K, distortCoef);
    }
    else
    {
        currentFrame = Frame(currentImage, OrbExtractor, K, distortCoef);
    }

}

Eigen::Matrix4d Tracking::Run(const cv::Mat &image)
{
    // 图像预处理，把输入的图片提取完特征点并保存成Frame对象
    PreProcessImage(image);

    // tracking流程的主函数，追踪该帧的位置并建图
    track();
}

void Tracking::MonoInitialize()
{
    Initializer initializer;

    // Step0: 检查当前帧上keypoint的数目，如果只能提出不到100个关键点，那么这一帧肯定不能用来做初始化，无论是做第一帧还是做第二帧都不行
    if (currentFrame.numKeypoints < 100)
    {
        initializer.bIniFrameCreated = false;  // 因为初始化要求必须是连续的两帧，所以第二帧不行，第一帧能提再多的keypoint也作废
        return;
    }

    // Step1: 生成初始帧，条件比较宽松，只要能提到100个关键点就能做初始帧
    if (!initializer.bIniFrameCreated)
    {
//        std::vector<Eigen::Vector2f> keypoints1;
//        keypoints1.reserve(currentFrame.numKeypoints);
//
//        for (const cv::KeyPoint &kpt: currentFrame.keypoints)
//        {
//            Eigen::Vector2f pt(kpt.pt.x, kpt.pt.y);
//            keypoints1.push_back(pt);
//        }
        initializer.keypoints1 = currentFrame.keypoints;
        initializer.bIniFrameCreated = true;

        return;
    }

    // Step2: 生成第二帧，必须要是初始帧的下一帧，而且能和初始帧找到100个匹配点，否则重新开始初始化过程
    ORBmatcher OrbMatcher(0.9, true);

}
