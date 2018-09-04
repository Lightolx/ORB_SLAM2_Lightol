//
// Created by lightol on 18-8-24.
//

#include "System.h"


System::System(const std::string &vocFilepath, const std::string &camSetFilepath)
{
    cout << "This is a simplified version of ORB_SLAM2 project provided by Lightol." << endl;
    cout << "You will find that so many modules and functions are removed, this project will\n"
         << "only leave you a rough impression of the ORB_SLAM2 algorithm, by it is more\n"
         << "readable than the original project, at least in my opinion. Well, this is an\n"
         << "introduction reading for guys just start learning ORB_SLAM" << endl;

    // Step0: 导入词典文件并存在System类的vocalbulary_成员变量中
    vocabulary_ = new ORB_Vocalbulary();
    vocabulary_->loadFromTextFile(vocFilepath);
    
    
}

