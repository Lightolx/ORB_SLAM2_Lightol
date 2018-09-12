#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <unistd.h>
#include <opencv/cv.hpp>

#include "System.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"

int main(int argc, char **argv)
{

    cv::Mat image1 = cv::imread("../images/1.png");
    cv::Mat image2 = cv::imread("../images/3.png");
//    cv::Mat image1 = cv::imread("1.png");
//    cv::Mat image2 = cv::imread("2.png");
    ORBextractor ORBextractor(1000, 1.2, 8, 20, 7);
    cv::Mat K1 = cv::Mat::eye(3,3,CV_32F);
    K1.at<float>(0,0) = 100;
    K1.at<float>(1,1) = 100;
    K1.at<float>(0,2) = 100;
    K1.at<float>(1,2) = 100;
    cv::Mat distortCoef1 = cv::Mat::zeros(4,1,CV_32F);
    Frame frame1(image1, ORBextractor, K1, distortCoef1);
    Frame frame2(image2, ORBextractor, K1, distortCoef1);

    std::map<int, int> matches;
    int nMatches = ORBmatcher::FindMatchingPoints(frame1, frame2, 50, matches);

//    cout << "nMatches is " << nMatches << endl;


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
        cv::KeyPoint kpt1 = frame1.keypoints[match.first];
        cv::KeyPoint kpt2 = frame2.keypoints[match.second];
        kpt2.pt.y += rows1+20;

        uchar b = rand();uchar g = rand(); uchar r = rand();
        cv::circle(image3, kpt1.pt, 2, cv::Scalar(b,g,r));
        cv::circle(image3, kpt2.pt, 2, cv::Scalar(b,g,r));
        cv::line(image3, kpt1.pt, kpt2.pt, cv::Scalar(b,g,r));
    }


    cv::imshow("image3", image3);
    cv::waitKey();


//    for (auto kpt: keypoints1)
//    {
//        cv::circle(image1, kpt.pt, 2, cv::Scalar(0,0,255));
//    }

//    cv::imshow("myimage1", image1);
//    cv::waitKey();
}
