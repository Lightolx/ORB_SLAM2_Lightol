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

bool LoadImages(const std::string &filepath,                // i
                std::vector<std::string> &imageFilenames,   // o
                std::vector<double> &timeStamps);           // o

int main(int argc, char **argv)
{

    // Step0: 从下载的TUM数据集的rgb.txt文件中读取每一帧图像的文件路径以及相应的时间戳，分别存放
    //        在vstrImageFilenames和vTimestamps两个vector中
    std::string filepath = std::string(argv[3]) + "/rgb.txt";
    std::vector<std::string> imageFilenames;
    std::vector<double> timeStamps;

    if(!LoadImages(filepath, imageFilenames, timeStamps))
    {
        cerr << "Cannot load images from " << filepath << "check it!" << endl;
    }

    // Step1: 生成一个System对象，它的初始化输入包括字典文件的路径，相机参数文件的路径，摄像头的类型
    // todo:: creare SLAM
    System SLAM(argv[1], argv[2]);

    // Step2: 对于每一帧图像，把它送入tracking线程进行处理
    std::string filePath = std::string(argv[3]);
    int imageNum = imageFilenames.size();

    cv::Mat image = cv::Mat();
    for(int i = 0; i < imageNum; i++)
    {
        image = cv::imread(filePath + "/" + imageFilenames[i], CV_LOAD_IMAGE_UNCHANGED);
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // 对该帧进行定位并返回它的pose
        Eigen::Matrix4d Tcw = SLAM.TrackMonocular(image);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        // tacking线程处理该帧花费的时间
        double tTrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        // 看这一帧处理完了之后下一帧是否已经捕捉到了，如果没有的话线程就只有等着，直到相机拍了下一帧图像
        // 其实很简单啊，就像scanf函数，如果用户没有从键盘输入字符，那么scanf函数也只能等着用户输入下一个字符
        if (i != imageNum-1)  // 最后一帧不用等下一帧的输入了，这个时候tracking线程也不用等了，可以直接结束
        {
            double shootInt = timeStamps[i+1] - timeStamps[i];

            if(tTrack < shootInt)
            {
                usleep((shootInt-tTrack)*1e6);
            }
        }

    }

    // Step3: 所有的图像都处理完了，这个时候可以关闭整个系统的３个线程了
    // todo:: SLAM.shutdown()

    // Step4: 把关键帧的光心位置以及pose保存下来，便于画出轨迹
    // todo:: SLAM.saveTrajectory()

}

bool LoadImages(const std::string &filepath,
                std::vector<std::string> &imageFilenames,
                std::vector<double> &timeStamps)
{
    std::ifstream fin(filepath.c_str());

    std::string s0;
    getline(fin, s0);
    getline(fin, s0);
    getline(fin, s0);

    double t;
    std::string filename;
    while (getline(fin, s0))
    {
        std::stringstream ss(s0);
        ss >> t;
        ss >> filename;
        timeStamps.push_back(t);
        imageFilenames.push_back(filename);
    }

    return true;
}