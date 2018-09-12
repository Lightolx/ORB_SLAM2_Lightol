//
// Created by lightol on 18-9-7.
//
#include <iostream>
#include <chrono>

#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv/cv.hpp>

#include "Tracking.h"
#include "ORBmatcher.h"

using std::cout;
using std::endl;

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
//    fps = fSettings["Camera.fps"];
    fps = 30;
//    if (fps == 0)
//    {
//        fps = 30;
//    }

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

void Tracking::PreProcessImage(cv::Mat image)
{
    // step1: 检查输入的图片．如果不是灰度图就转换成灰度图
    if (image.channels() == 3)
    {
        if (bRGB)
        {
            cv::cvtColor(image, image, CV_RGB2GRAY);
        }
        else
        {
            cv::cvtColor(image, image, CV_BGR2GRAY);
        }
    }
    else if (image.channels() == 4)
    {
        if (bRGB)
        {
            cv::cvtColor(image, image, CV_RGBA2GRAY);
        }
        else
        {
            cv::cvtColor(image, image, CV_BGRA2GRAY);
        }
    }

    // step2: 如果是初始化则需要提取2000个关键点，如果是后续的追踪过程则只提取1000个就行
    if ( trackingState == NO_IMAGE_YET || trackingState == NOT_INITIALIZED)
    {
        currentFrame = Frame(image, iniOrbExtractor, K, distortCoef);
    }
    else
    {
        currentFrame = Frame(image, OrbExtractor, K, distortCoef);
    }

}

Eigen::Matrix4d Tracking::Run(const cv::Mat &image)
{
    // Step1: 图像预处理，把输入的图片提取完特征点并保存成Frame对象
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    PreProcessImage(image);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // Step2: tracking流程的主函数，追踪该帧的位置并建图
    track();
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

//    cout << "preProcess = " << std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() << endl;
//    cout << "track = " << std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count() << endl;
}

void Tracking::MonoInitialize()
{
    // Step0: 检查当前帧上keypoint的数目，如果只能提出不到100个关键点，那么这一帧肯定不能用来做初始化，无论是做第一帧还是做第二帧都不行
    if (currentFrame.numKeypoints < 100)
    {
        initializer.bIniFrameCreated = false;  // 因为初始化要求必须是连续的两帧，所以第二帧不行，第一帧能提再多的keypoint也作废
//        initializer.frameCount = 0;
        return;
    }

    // Step1: 生成初始帧，条件比较宽松，只要能提到100个关键点就能做初始帧
    if (!initializer.frameCount)
    {
        initializer.initialFrame = currentFrame;
        initializer.frameCount++;

        return;
    }

    // Step2: 生成第二帧，必须要是初始帧的下一帧，而且能和初始帧找到100个匹配点，否则重新开始初始化过程
//    if (++initializer.frameCount < 6)  // 隔一帧再与初始帧匹配，保证视差大一点
//    {
//        return;
//    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ORBmatcher OrbMatcher(0.9, true);
    std::map<int, int> matches;
    int nMatches = OrbMatcher.FindMatchingPoints(initializer.initialFrame, currentFrame, 10, matches);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//    cout << "findMatches = " << std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() << endl;
    if (nMatches < 100)  // 找不到足够的匹配点就重新开始初始化
    {
        initializer.initialFrame = currentFrame;
//        initializer.frameCount = 0;

        return;
    }
    else
    {
        initializer.currentFrame = currentFrame;
//        cout << "Initialization beginning..." << endl;
    }

    cv::Mat image1 = initializer.initialFrame.image;
    cv::Mat image2 = initializer.currentFrame.image;

    {
        int cols1 = image1.cols;
        int cols2 = image2.cols;
        int rows1 = image1.rows;
        int rows2 = image2.rows;

        int rows3 = rows1 + rows2 + 20;
        cv::Mat image3(rows3, cols1, image1.type());
        uchar *ptr1 = image1.data;
        uchar* ptr2 = image2.data;
        uchar* ptr3 = image3.data;
        int step03 = image3.step[0];
        int step13 = image3.step[1];
        int step01 = image1.step[0];
        int step11 = image1.step[1];
        int channel = image3.channels();
        int elemSize1 = image3.elemSize1();

        for (int i = 0; i < image1.rows; ++i)
        {
            for (int j = 0; j < image1.cols; ++j)
            {
                for (int k = 0; k < channel; ++k)
                {
                    *(ptr3 + i*step03 + j*step13 + k*elemSize1) = *(ptr1 + i*step01 + j*step11 + k*elemSize1);
                    *(ptr3 + (i+rows1+20)*step03 + j*step13 + k*elemSize1) = *(ptr2 + i*step01 + j*step11 + k*elemSize1);
                }
            }
        }

        for (const auto &match: matches)
        {
            cv::KeyPoint kpt1 = initializer.initialFrame.keypoints[match.first];
            cv::KeyPoint kpt2 = initializer.currentFrame.keypoints[match.second];
            kpt2.pt.y += rows1+20;

            uchar b = rand();uchar g = rand(); uchar r = rand();
            cv::circle(image3, kpt1.pt, 2, cv::Scalar(b,g,r));
            cv::circle(image3, kpt2.pt, 2, cv::Scalar(b,g,r));
            cv::line(image3, kpt1.pt, kpt2.pt, cv::Scalar(b,g,r));
        }


//        cv::imshow("image3", image3);
//        cv::waitKey(1000);
    }


    // Step3: 初始化流程开始，首先根据匹配点对计算当前帧相对与初始帧的pose
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    std::vector<Eigen::Vector3d> vMapPoints;       // 三角化出来的mapPoints
    std::vector<std::pair<int, int>> MpMatches;    // 上面的mapPoints在两帧中对应的keypoint的ID
    if (initializer.ComputeRelativePose(matches, R, t, vMapPoints, MpMatches))
    {
        cout << "initialization done" << endl << endl;
        return;
        // create initial map
    }
    initializer = Initializer();
}
