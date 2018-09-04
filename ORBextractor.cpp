//
// Created by lightol on 18-8-27.
//

#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

#include "ORBextractor.h"

using std::cout;
using std::endl;

using ORB_SLAM::Block;

const int IMAGE_MARGIN = 19;
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;

static int BRIEF_bit[256*4] =
{
        8,-3, 9,5/*mean (0), correlation (0)*/,
        4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
        -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
        7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
        2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
        1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
        -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
        -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
        -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
        10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
        -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
        -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
        7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
        -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
        -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
        -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
        12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
        -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
        -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
        11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
        4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
        5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
        3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
        -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
        -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
        -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
        -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
        -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
        -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
        5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
        5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
        1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
        9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
        4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
        2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
        -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
        -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
        4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
        0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
        -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
        -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
        -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
        8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
        0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
        7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
        -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
        10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
        -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
        10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
        -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
        -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
        3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
        5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
        -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
        3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
        2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
        -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
        -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
        -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
        -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
        6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
        -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
        -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
        -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
        3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
        -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
        -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
        2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
        -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
        -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
        5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
        -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
        -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
        -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
        10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
        7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
        -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
        -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
        7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
        -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
        -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
        -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
        7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
        -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
        1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
        2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
        -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
        -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
        7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
        1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
        9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
        -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
        -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
        7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
        12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
        6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
        5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
        2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
        3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
        2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
        9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
        -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
        -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
        1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
        6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
        2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
        6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
        3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
        7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
        -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
        -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
        -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
        -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
        8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
        4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
        -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
        4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
        -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
        -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
        7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
        -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
        -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
        8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
        -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
        1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
        7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
        -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
        11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
        -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
        3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
        5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
        0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
        -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
        0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
        -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
        5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
        3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
        -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
        -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
        -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
        6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
        -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
        -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
        1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
        4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
        -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
        2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
        -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
        4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
        -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
        -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
        7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
        4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
        -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
        7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
        7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
        -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
        -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
        -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
        2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
        10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
        -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
        8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
        2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
        -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
        -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
        -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
        5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
        -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
        -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
        -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
        -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
        -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
        2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
        -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
        -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
        -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
        -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
        6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
        -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
        11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
        7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
        -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
        -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
        -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
        -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
        -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
        -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
        -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
        -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
        1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
        1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
        9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
        5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
        -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
        -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
        -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
        -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
        8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
        2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
        7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
        -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
        -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
        4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
        3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
        -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
        5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
        4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
        -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
        0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
        -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
        3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
        -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
        8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
        -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
        2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
        10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
        6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
        -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
        -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
        -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
        -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
        -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
        4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
        2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
        6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
        3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
        11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
        -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
        4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
        2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
        -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
        -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
        -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
        6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
        0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
        -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
        -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
        -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
        5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
        2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
        -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
        9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
        11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
        3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
        -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
        3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
        -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
        5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
        8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
        7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
        -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
        7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
        9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
        7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
        -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

namespace ORB_SLAM
{
Block::Block():bNoMore(false)
{

}

void Block::DivideBlock(Block &block1, Block &block2, Block &block3, Block &block4)
{
    const int halfX = (UR.x() - UL.x()) / 2;  // todo:: ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = (BL.y() - BL.x()) / 2;

    // Step1: 确定四个子四叉树的四个边界点
    block1.UL = UL;
    block1.UR = Eigen::Vector2i(UL.x() + halfX, UL.y());
    block1.BL = Eigen::Vector2i(UL.x(), UL.y() + halfY);
    block1.BR = Eigen::Vector2i(UL.x() + halfX, UL.y() + halfY);

    block2.UL = block1.UR;
    block2.UR = UR;
    block2.BL = block1.BR;
    block2.BR = Eigen::Vector2i(UR.x(), UR.y() + halfY);

    block3.UL = block1.BL;
    block3.UR = block1.BR;
    block3.BL = BL;
    block3.BR = Eigen::Vector2i(BL.x() + halfX, BL.y());

    block4.UL = block1.BR;
    block4.UR = block2.BR;
    block4.BL = block3.BR;
    block4.BR = BR;

    // Step2: 把父四叉树上的fast角点（关键点）分配到四个子四叉树上
    int numKpt = keypoints.size();
    block1.keypoints.reserve(numKpt);
    block2.keypoints.reserve(numKpt);
    block3.keypoints.reserve(numKpt);
    block4.keypoints.reserve(numKpt);

    for (const cv::KeyPoint &pt: keypoints)
    {
        int x = pt.pt.x;
        int y = pt.pt.y;

        if (x < UL.x() + halfX)
        {
            if (y < UL.y() + halfY)
            {
                block1.keypoints.push_back(pt);
            }
            else
            {
                block3.keypoints.push_back(pt);
            }
        }
        else
        {
            if (y < UL.y() + halfY)
            {
                block2.keypoints.push_back(pt);
            }
            else
            {
                block4.keypoints.push_back(pt);
            }
        }
    }

    if (block1.keypoints.size() < 2)
    {
        block1.bNoMore = true;
    }
    if (block2.keypoints.size() < 2)
    {
        block2.bNoMore = true;
    }
    if (block3.keypoints.size() < 2)
    {
        block3.bNoMore = true;
    }
    if (block4.keypoints.size() < 2)
    {
        block4.bNoMore = true;
    }
}
}

// 计算关键点kpt所在patch（由PATCH_SIZE确定）的灰度质心与图像x轴正方向的夹角， Intensity Center Angle
static float IC_Angel(const cv::Mat &image, const cv::KeyPoint &kpt, const std::vector<int> &nPixelPerRow)
{
    int m10 = 0; // 表示x方向的图像矩，也就是每一列上，以所有行的灰度值之和作为权重，算出的x方向上的中点
    int m01 = 0; // 同上

    const uchar *ptr = &image.at<uchar>(kpt.pt);    // 因为要逐字节操作，所以声明uchar型指针，也幸好开始就把图像转成了gray，也就是CV_8UC1，image的每个元素只有一个通道，每个通道只占一个字节，也就是一个像素只占一个字节，肯定不会出错
    int step0 = image.step[0];

    // 遍历每一个点，把它们相对于x与y方向上的矩累加起来
    for (int v = -HALF_PATCH_SIZE; v <= HALF_PATCH_SIZE; ++v)
    {
        int nPixel = 0;
        if (v >= 0)
        {
            nPixel = nPixelPerRow[v];
        } else
        {
            nPixel = nPixelPerRow[-v];
        }

        for (int u = -nPixel; u <= nPixel; ++u)
        {
            m10 += *(ptr + v * step0 + u) * u;  // 前面是以灰度作为权重，后面是该点的x坐标
            m01 += *(ptr + v * step0 + u) * u;  // 前面是以灰度作为权重，后面是该点的y坐标
        }
    }

    return cv::fastAtan2(m01,m10);
}

ORBextractor::ORBextractor(int _nFeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST):
nFeatures(_nFeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    // Step0: 初始化一些成员变量
    BRIEFs = std::vector<Eigen::Vector2i>();

    // Step1: 计算每一层金字塔的尺度缩放系数，也就是0.8的n次方
    imagePyramids.resize(nlevels);
    scaleFactors.resize(nlevels);
    invScaleFactors.resize(nlevels);
    sigma2s.resize(nlevels);
    invsigma2s.resize(nlevels);
    for (int i = 0; i < nlevels; ++i)
    {
        scaleFactors[i] = std::pow(scaleFactor, i);
        invScaleFactors[i] = 1.0/scaleFactors[i];
        sigma2s[i] = std::pow(scaleFactors[i], 2);
        invsigma2s[i] = 1.0/sigma2s[i];
    }

    // Step2: 计算在每一层金字塔上理论上应该提取多少个关键点，越往上图像的像素数越少，提取的特征点应该越少
    nFeaturesPerLevel.resize(nlevels);
    float s = 1.0/scaleFactor;
    nFeaturesPerLevel[0] = nFeatures * (1 - s) / (1 - pow(s, nlevels)) ; // 等比数列求和, nFeatures = n0 * (1-s)/(1-s^n), n0是金字塔最底层应该提取的keypoint的数目
    int sumFeatures = nFeaturesPerLevel[0];

    for (int i = 1; i < nlevels-1; ++i)
    {
        nFeaturesPerLevel[i] = nFeaturesPerLevel[0] * invScaleFactors[i];
        sumFeatures += nFeaturesPerLevel[i];
    }

    nFeaturesPerLevel[nlevels-1] = std::max(nFeatures - sumFeatures, 0); // nFeatures = 1000时，第一层应该有217个点，每往上一层乘以0.8，最顶层用来兜底，应该有60个点

    // Step3: 确定BRIEF算子的256对比较点，这是预先就按照一定的概率分布确定好了的
    const Eigen::Vector2i* BRIEFs0 = (const Eigen::Vector2i*)BRIEF_bit;
    std::copy(BRIEFs0, BRIEFs0+512, std::back_inserter(BRIEFs));

    // Step4: 预先计算出求解特征点方向的patch的范围，它应该是个圆，所以计算一下从上至下每一行有多宽
    nPixelPerRow.resize(HALF_PATCH_SIZE + 1);    // 只计算上半部分，因为上下是对称的，算完了上半部分下半部分直接拷贝就行了

    // step4.1: 首先计算45°对角线以下与x轴正方向所夹的范围
    int v45 = std::ceil(double(HALF_PATCH_SIZE) / std::sqrt(2));   // 计算45°对角线以下总共有多少行，用ceil而不是floor是免得对角线上这一行恰好被漏了
    const int R2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;              // 半径的平方，先算出来免得下面又要算

    for (int i = 0; i <= v45 ; ++i)
    {
        nPixelPerRow[i] = int(std::sqrt(R2 - i*i));
    }

    // step4.2: 再算45°对角线以上的部分，其实不用算，因为它必须是个对称的，所以横着有几行，竖着就有几列
    for (int row = HALF_PATCH_SIZE, rowId = 0; row > v45; --row)  // rowId, 宽度改变时的行标
    {
        while (nPixelPerRow[rowId] == nPixelPerRow[rowId + 1])
        {
            ++rowId;
        }

        nPixelPerRow[row] = rowId;
        ++rowId;
    }
}

void ORBextractor::ConstructPyramid(const cv::Mat &image)
{
    // todo:: cv::Mat 的浅复制与深复制
    imagePyramids[0] = image.clone();

    int ncols = image.cols;
    int nrows = image.rows;

    // Step1: 对于每一层图像，根据其缩放因子resize图像大小
    for (int i = 1; i < nlevels; ++i)
    {
        float scale = invScaleFactors[i];
        cv::Size sz(int(ncols*scale), int(nrows*scale));
        cv::resize(image, imagePyramids[i], sz, 0, 0, cv::INTER_LINEAR);
    }

    // Step2: 给所有的图像加上边缘，便于求FAST角点时对边缘线3像素以内的点也能进行计算
    //        边缘margin以内的19个像素就不参与提取特征点了
//    for (int i = 0; i < nlevels; ++i)
//    {
//        cv::copyMakeBorder(imagePyramids[i], imagePyramids[i], IMAGE_MARGIN, IMAGE_MARGIN,
//                           IMAGE_MARGIN, IMAGE_MARGIN, cv::BORDER_REFLECT_101);
//    }
}

void ORBextractor::ExtractKeypoint(std::vector<std::vector<cv::KeyPoint>> &vvKeypoints) const
{
    vvKeypoints.reserve(nlevels);

    const int cellSize = 30;           // 将整幅图像划分为无数个大小为30*30px的cell

    for (int level = 0; level < nlevels; ++level)
    {
        // Step1: 上下左右都往里收缩19个像素，裁剪出ROI区域,并将ROI区域在横竖方向上划分为多个网格,以及确定每个cell的size
        const cv::Mat image = imagePyramids[level];
        const int ROIwidth = image.cols - 2*IMAGE_MARGIN ;
        const int ROIheight = image.rows - 2*IMAGE_MARGIN;
        // 从ROI区域上下左右各扩出去3个像素，由于上一步的copyMakeBorder()所以这些ROI以外的地方是有像素值的
//        const int minROIX = IMAGE_MARGIN - 3;
//        const int minROIY = IMAGE_MARGIN - 3;
//        const int maxROIX = image.cols - IMAGE_MARGIN + 3;
//        const int maxROIY = image.rows - IMAGE_MARGIN + 3;

        const int minROIX = IMAGE_MARGIN;
        const int minROIY = IMAGE_MARGIN;
        const int maxROIX = image.cols - IMAGE_MARGIN;
        const int maxROIY = image.rows - IMAGE_MARGIN;

        std::vector<cv::KeyPoint> kptsToDistribute;   // 留待分配位置,建立四叉树的keypoints

        // 检测出keypoint后还会删除一些，并不是所有的keypoint最终都会成为ORB特征点
        kptsToDistribute.reserve(10*nFeatures);

        const int nCols = ROIwidth / cellSize;      // 在横轴方向，应该有nCols个cell
        const int nRows = ROIheight / cellSize;     // 在纵轴方向，应该有nRows个cell

        // note: 由于这里采用了ceil函数，所以每个cell的size比应该的size是要大半个像素左右的，这样会造成　
        //        nCols*xCellSize > ROIwidth，在下面的循环中应该注意
        const int xCellSize = ceil(float(ROIwidth) / nCols);   // 每个cell的横轴方向的大小为xCellSize
        const int yCellSize = ceil(float(ROIheight) / nRows);  // 每个cell的纵轴方向的大小为yCellSize

        // Step2: 对每个cell都提取FAST角点，这样能保证在整张图上比较均匀的采集keypoint
        bool bBottom = false;           // 如果划窗已经触及到了底面，则迭代停止
        bool bRightBoundary = false;
        for (int i = 0; i < nRows; ++i)
        {
            const int minY = minROIY + i*yCellSize;
            int maxY = minY + yCellSize;

            if (maxY >= maxROIY)
            {
                maxY = maxROIY;
                bBottom = true;  // 如果触到了底面，则只检测ROI区域包括的部分
            }

            for (int j = 0; j < nCols; ++j)
            {
                const int minX = minROIX + j*xCellSize;
                int maxX = minX + xCellSize;

                if (maxX >= maxROIX)
                {
                    maxX = maxROIX;
                    bRightBoundary = true;
                }

                std::vector<cv::KeyPoint> keypoints;
                // 因为colRange()和rowRange()都是左闭右开的，所以需要加一
                cv::FAST(image.rowRange(minY-3, maxY+4).colRange(minX-3, maxX+4), keypoints, iniThFAST, true);

                // 如果在这个cell内提不到点，那么就适当放宽FAST角点的限制，从连续20个点变为连续7个点
                if (keypoints.empty())
                {
                    cv::FAST(image.rowRange(minY-3, maxY+4).colRange(minX-3, maxX+4), keypoints, iniThFAST, true);
                }

                if (!keypoints.empty())
                {
                    for (cv::KeyPoint &kpt: keypoints)
                    {
                        // FAST函数返回的坐标是相对于点（minX-3, minY-3)的
                        kpt.pt.x += minX-3;
                        kpt.pt.y += minY-3;
                        kptsToDistribute.push_back(kpt);
                    }
                }

                if (bRightBoundary)
                {
                    break;
                }
            }

            if (bBottom)
            {
                break;
            }
        }

        // Step3: 所有的cell中提取到的特征点，用四叉树的形式一层一层地分割，保证最后得到的keypoint在整张图上分布得比较均匀
        vvKeypoints[level] = DistributeKpts(kptsToDistribute, minROIX, maxROIX, minROIY, maxROIY, nFeaturesPerLevel[level]);

        for (cv::KeyPoint &kpt: vvKeypoints[level])
        {
            kpt.octave = level;  // 表示这个keypoint是在第几层金字塔上提取出来的
        }
    }
}

std::vector<cv::KeyPoint> ORBextractor::DistributeKpts(const std::vector<cv::KeyPoint> &keypoints,
                                        int minX, int maxX, int minY, int maxY,int nFeatures) const
{
    // Step0: 输入数据检查
    if (keypoints.empty())
    {
        std::vector<cv::KeyPoint> temp;
        temp.clear();
        return temp;
    }

    if ((maxX-minX) / (maxY-minY) >1)
    {
        // todo:: 默认X方向大于Y方向，初始块需要分为两块
    }

    // 使用list便于后面删除某个节点的操作，因为一个父block划分为4个子block后，父block需要删除
    std::list<Block> lBlocks;

    // Step1: 将整个ROI区域设为初始block并填写其所有的成员变量
    Block block0;  // 一般图像长和宽差不多，初始时只有一个块
    block0.UL = Eigen::Vector2i(minX, minY);
    block0.UR = Eigen::Vector2i(maxX, minY);
    block0.BL = Eigen::Vector2i(minX, maxY);
    block0.BR = Eigen::Vector2i(maxX, maxY);
    block0.keypoints.assign(keypoints.begin(), keypoints.end());

    if (block0.keypoints.size() == 1)
    {
        block0.bNoMore = true;
    }

    lBlocks.push_back(block0);
    lBlocks.back().position = lBlocks.end();

    // Step2: 将Block集合不断分裂直至list中包含的block的个数大于等于nFeatures，因为最终
    //        一个Block中只会保留一个response最强的FAST角点
    bool bFinish = false;

    while (!bFinish)
    {
//        cout << "starting a new loop" << endl;
        int nToDivede = 0;
        int nBlocksIni = 0;
        std::vector<std::pair<int, Block*>> vBlocksToDivide;

        // Step2.1: 第一轮分裂，把list中现有的blocks分裂完，记得一个block分裂后就该把它从list中erase掉
        for (std::list<Block>::iterator it = lBlocks.begin(); it != lBlocks.end();)
        {
            if (it->bNoMore)  // 如果已无法再分裂，那么不分裂也不擦除它
            {
                it++;         // 因为for的第三部分不是递加，所以在每条分支后面都得手动把指针加一
                continue;
            }

            Block block1, block2, block3, block4;
            it->DivideBlock(block1, block2, block3, block4);
            std::vector<Block*> blocks;
            blocks.reserve(4);
            blocks.push_back(&block1);
            blocks.push_back(&block2);
            blocks.push_back(&block3);
            blocks.push_back(&block4);

            // Process1: 新生成的block加入到list中，当然仅仅对于那些包含keypoint的block，
            //           比如在block0中，它的左下角区域一个keypoint也没有，那么它分裂出来的block3中
            //           也不包含keypoint，那么block3就不加入到list中
            for (Block* pBlock: blocks)
            {
                if (pBlock->keypoints.empty())
                {
                    continue;
                }

                // 注意这里是深拷贝，也就是push_front的是一个新的对象，此时lBlocks.front()
                // 与*pBlock两个对象虽然包含的成员变量的值都相同，但它们是不同的Block对象，
                // 存储在不同的位置，此后再操作也只能是操作lBlocks.front()，因为*pBlock是
                // 一个临时变量，可能下一轮循环时就已经被销毁了
                lBlocks.push_front(*pBlock);
                lBlocks.front().position = lBlocks.begin();

                // 如果这个子block还有继续分裂的可能，那么很大概率它会1变4，总的block数翻3倍
                if (!pBlock->bNoMore)
                {
                    nToDivede++;
                    vBlocksToDivide.push_back(std::make_pair(pBlock->keypoints.size(), &lBlocks.front()));
                }
            }

            // Process2: 老的block从list中去除掉
            it = lBlocks.erase(it);
        }

        // terminal conditon: 如果此时list中block的数目已经达到了nFeatures,那么就可以停止继续分裂成更小块了,
        //                    或者这一波分裂完全没有增加blocks数目, list中所有的block已经都只包含一个keypoint,
        //                    已经分裂不动了,这个时候也要停了
//        cout << "in the first loop, lBlocks.size() = " << lBlocks.size() << endl;
        if (lBlocks.size() >= nFeatures || lBlocks.size() == nBlocksIni)
        {
            cout << "break first loop" << endl;
            break;
        }

        // Step2.2: 第二波分裂,如果分裂即将进入尾声(把刚才分出的几个包含keypoint较多的字块再分一下,有可能使得
        //          总block数达到nfeatures),那么此时采取精英原则,就按照包含keypoints数从多到少的顺序分裂这些子块,
        //          这是遵循keypoint密集的地方优先分裂的原则,毕竟keypoint多的地方也应该最终取
        //          较多的block块生成较多的keypointDistributed
        if (lBlocks.size() + 3*nToDivede > nFeatures)
        {
            while (!bFinish)
            {
                int nBlocksIni2 = lBlocks.size();
                std::vector<std::pair<int, Block*>> vBlocksToDivide2 = vBlocksToDivide;
                vBlocksToDivide.clear();
                std::sort(vBlocksToDivide2.begin(), vBlocksToDivide2.end(), std::greater<std::pair<int, Block*>>());

                int itpIntBlock = 0;
                for (const std::pair<int, Block*> &pIntBlock: vBlocksToDivide2)
                {
                    itpIntBlock++;
                    Block block1, block2, block3, block4;
                    pIntBlock.second->DivideBlock(block1, block2, block3, block4);
                    std::vector<Block*> blocks;
                    blocks.reserve(4);
                    blocks.push_back(&block1);
                    blocks.push_back(&block2);
                    blocks.push_back(&block3);
                    blocks.push_back(&block4);

                    for (Block* pBlock: blocks)
                    {
                        if (pBlock->keypoints.empty())
                        {
                            continue;
                        }

                        lBlocks.push_front(*pBlock);
                        lBlocks.front().position = lBlocks.begin();

                        if (!pBlock->bNoMore)
                        {
                            vBlocksToDivide.push_back(std::make_pair(pBlock->keypoints.size(), &lBlocks.front()));
                        }
                    }

                    // Process2: 老的block从list中去除掉
                    lBlocks.erase(pIntBlock.second->position);

                    if(lBlocks.size() >= nFeatures)
                    {
                        break;
                    }
                }

                // terminal condition: 在这里也把第二个条件加进来，因为的确可能这张图就提不到nFeatures个keypoint，
                //                     这个时候再分裂也没有意义，你还不让他停？
                cout << "in the second loop, lBlocks.size() = " << lBlocks.size() << endl;
                if (lBlocks.size() >= nFeatures || lBlocks.size() == nBlocksIni2)
                {
                    bFinish = true;
                }

            }
        }
    }

    // Step3: 从list中的每个block中挑选出respond最大的那个keypoint保留下来
    std::vector<float> responses;
    std::vector<cv::KeyPoint> vkptsDistributed;

    for (const Block &block: lBlocks)
    {
        const std::vector<cv::KeyPoint> &keypoints = block.keypoints;
        responses.clear();
        responses.reserve(keypoints.size());

        for (const cv::KeyPoint &kpt: keypoints)
        {
            responses.push_back(kpt.response);
        }

        std::vector<float>::iterator maxIter = std::max_element(responses.begin(),responses.end());
        int maxId = std::distance(responses.begin(), maxIter);
        vkptsDistributed.push_back(keypoints[maxId]);
    }

    cout << "finish the keypoints assignment" << endl;

    return vkptsDistributed;


    /*
    for(std::list<Block>::iterator lit=lBlocks.begin(); lit!=lBlocks.end(); lit++)
    {
        std::vector<cv::KeyPoint> &vNodeKeys = lit->keypoints;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        keypointsDistributed.push_back(*pKP);
    }
     */
    
}

void ORBextractor::ComputeKeypointDirection(std::vector<std::vector<cv::KeyPoint>> &vvKeypoints) const
{
    for (int level = 0; level < nlevels; ++level)
    {
        std::vector<cv::KeyPoint> &vKeypoints = vvKeypoints[level];  //　该层图像上所有的关键点

        for (cv::KeyPoint &kpt: vKeypoints)
        {
            kpt.angle = IC_Angel(imagePyramids[level], kpt, nPixelPerRow);
        }
    }
}

void ORBextractor::ComputeOrbDescriptor(const cv::KeyPoint &kpt, const cv::Mat &image, const cv::Point2i *BRIEF_bit,
                                        uchar* rowAdress)
{
    int step0 = image.step[0];
    double theta = kpt.angle * M_PI / 360;
    float a = std::sin(theta);
    float b = std::cos(theta);
    uchar* ptr = image.data + int(kpt.pt.y)*step0 + int(kpt.pt.x);  // OpenCV为啥非要定义kpt.pt为Point2f类型？

    auto Intensity = [&](int id)     // 这里按引用传递而不是按值传递，因为接下来我们会改变BRIEF_bit的值
    {
        cv::Point2i pt = *(BRIEF_bit + id);   // BRIEF描述子中的一个采样点
        int u = int(pt.x * b - pt.y * a);     // 旋转后的坐标系下的坐标变换到旋转之前, Xw = Twc*Xc;
        int v = int(pt.x * a + pt.y * b);
        return *(ptr + v*step0 + u);
    };

    for (int i = 0; i < 32; ++i)    // descriptor的size是 32x8, 刚好表示256个点对的大小, 并且每一行是一个字节
    {
        uchar t0 = 0, t1 = 0;   // 反正gray的颜色范围也可以用一个uchar表示
        uchar val = 0;          // 8个bit刚好是一个uchar

        for (int j = 0; j < 16; j += 2)
        {
            t0 = Intensity(j); t1 = Intensity(j+1);
            val |= (t0 < t1) << j/2;
        }

        *(rowAdress + i*8) = val;       // 将这8个bit赋给descriptor的对应行
        BRIEF_bit += 16;                // 8个点对，16个cv::Point2i
    }

}

// InputArray 和 OutputArray自带引用&，不信你按住ctrl点过去看, 而且InputArray还是常引用，不允许修改
void ORBextractor::operator()(cv::InputArray &_image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray _descriptors)
{
    cv::Mat image = _image.getMat();
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    assert(image.type() == CV_8UC1);  //　后面算BRIEF提取像素值的时候那么随意，就是因为输入的是最简单的CV_8UC1

    // Step1: 建立图像金字塔，按照0.8的系数依次resize出来8层
    ConstructPyramid(image);
    cout << "step1 end" << endl;

    // Step2: 在所有层上提取关键点，也就是keypoint保存在vector<<vector>>中，第一个vector表示金字塔的层，第二个表示一层上的keypoints
    std::vector<std::vector<cv::KeyPoint>> vvKeypoints;
    ExtractKeypoint(vvKeypoints);
    cout << "step2 end" << endl;

    // Step3: 在所有层上用灰度质心法计算关键点的方向，便于后面计算BRIEF特征时建立局部坐标系
    ComputeKeypointDirection(vvKeypoints);
    cout << "step3 end" << endl;

    // Step4: 计算所有层上提取出的keypoints的ORB特征
    // step4.1: 首先统计总共有多少个keypoints,　便于初始化keypoints和descriptor
    int nKeyPoints = 0;

    for (int level = 0; level < nlevels; ++level)
    {
        nKeyPoints += vvKeypoints[level].size();
    }

    cv::Mat descriptors;
    if (nKeyPoints == 0)
    {
        _descriptors.release();  // release all inner buffers
        return;
    }
    else
    {
        _descriptors.create(nKeyPoints, 32, CV_8UC1);   // 每一个keypoint的特征子都是256位bit，所以需要32个字节
        descriptors = _descriptors.getMat();     // 在getMat()前必须先调用create()分配空间
    }

    keypoints.clear();
    keypoints.reserve(nKeyPoints);

    // step4.2: 在每一层上计算keypoint的ORB特征
    int offset = 0;  // descriptors已经记录了多少个keypoint了
    for (int level = 0; level < nlevels; ++level)
    {
        std::vector<cv::KeyPoint> vKeypoints = vvKeypoints[level];  // 该层上提取出的所有的keypoints，声明引用是为了节省存储空间
        int nKeypointsLevel = vKeypoints.size();  // 这一层图像总共提取出了多少个keypoints
        if (nKeypointsLevel == 0)
        {
            continue;
        }

        // step4.2.1: 对该层图像进行高斯模糊，模糊是为了降噪，加强算子对噪声的鲁棒性，这是在为求BRIEF描述子做准备，BRODER_REFLECT_101是指在求边界的卷积时在边界以外的那些像素的插值方式，详见https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
        cv::Mat imageC = imagePyramids[level].clone();   // 注意这里clone()一个，不要把原来的图像也模糊了, 要不后面可能会有麻烦
        cv::GaussianBlur(imageC, imageC, cv::Size(7,7), 2, 2, cv::BORDER_REFLECT_101);  // cv::Size(7,7)指高斯卷积核的大小，2表示sigmaX,sigmaY

        // step4.2.2: 计算该层上keypoint的ORB特征并保存到descriptors的相应行中
        for (int i = 0; i < nKeypointsLevel; ++i)
        {
            ComputeOrbDescriptor(vKeypoints[i], imageC, (const cv::Point2i*)BRIEF_bit, descriptors.ptr(offset + i));
        }

        offset += nKeypointsLevel;

        // 将缩小后的图像上提出的keypoint的(u,v)坐标还原到原图中，因为它本来就是代表原图中的这个点的。所谓的尺度不变性，指的是在不同的尺度上提取这个点的ORB特征，既然提取完了那么自然应该把坐标还原到原图中。换句话说，原图中的点可能拥有多个ORB特征，只要它在不同的层上都被判定为关键点
        if (level != 0)
        {
            float scale = scaleFactors[level];

            for (cv::KeyPoint &kpt: vKeypoints)
            {
                kpt.pt *= scale;    // 就比如现图像缩小为原图像的一半，此时scale = 2，对于现图像上的一个特征点(15,20),　它其实是来自于原图像上的点(30,40), 因此这个点在现图像上计算出来的ORB特征是指原图像上的点(30,40)在某个尺度下的ORB特征，它还是属于点(30,40)的属性，所以这个keypoint的坐标要还原到原图像中的(30,40)
            }
        }

        keypoints.insert(keypoints.end(), vKeypoints.begin(), vKeypoints.end());

    }

}