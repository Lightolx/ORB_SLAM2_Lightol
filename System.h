//
// Created by lightol on 18-8-23.
//

#ifndef ORB_SLAM2_LIGHTOL_SYSTEM_H
#define ORB_SLAM2_LIGHTOL_SYSTEM_H

#include <string>
#include <iostream>

#include "ORBVocalbulary.h"

using std::cout;
using std::endl;
using std::cerr;

class System
{
public:
    System(const std::string &vocFilepath, const std::string &camSetFilepath);


private:
    ORB_Vocalbulary* vocabulary_;
};

#endif //ORB_SLAM2_LIGHTOL_SYSTEM_H
