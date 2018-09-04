//
// Created by lightol on 18-8-24.
//

#ifndef ORB_SLAM2_LIGHTOL_ORBVOCALBULARY_H
#define ORB_SLAM2_LIGHTOL_ORBVOCALBULARY_H

#include"Thirdparty/DBoW2/DBoW2/FORB.h"
#include"Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

using ORB_Vocalbulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

#endif //ORB_SLAM2_LIGHTOL_ORBVOCALBULARY_H
