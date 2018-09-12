//
// Created by lightol on 18-8-24.
//

#include "System.h"


System::System(const std::string &vocFilepath, const std::string &camSetFilepath)
{
    cout << " \nThis is a simplified version of ORB_SLAM2 project provided by Lightol. " << endl;
//            "You will find that so many modules \nand functions are removed, "
//            "this project will only leave you a rough impression of the ORB_SLAM2 algorithm,\n"
//            "but it is more readable than the original project, at least in my opinion. \nWell, this is an introduction reading for guys just start learning ORB_SLAM" << endl;

    // Step1: 导入词典文件并存在System类的vocalbulary_成员变量中
    vocabulary_ = new ORB_Vocalbulary();
    vocabulary_->loadFromTextFile(vocFilepath);

    // Step2: 实例化一个Tracker对象，并且就在本线程内运行track()函数
    pTracker = new Tracking(camSetFilepath);
    
}

Eigen::Matrix4d System::TrackMonocular(const cv::Mat &image) const
{
    // todo:: 查看是建图＋定位模式还是纯定位模式，是否需要reset系统等等

    Eigen::Matrix4d Tcw = pTracker->Run(image);
}

