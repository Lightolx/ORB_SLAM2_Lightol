//
// Created by lightol on 18-9-7.
//
#include <iostream>
#include <thread>

#include <sophus/so3.h>

#include "Initializer.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"


using std::cout;
using std::endl;

Initializer::Initializer(int _maxIteration):bIniFrameCreated(false), maxIterations(_maxIteration), frameCount(0)
{

}

bool Initializer::ComputeRelativePose(const std::map<int, int> &_matches, Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<Eigen::Vector3d> &vMapPoints)
{
    // Step0: 把map转换成vector，便于等会按id索引其元素
    int nMatches = _matches.size();
    matches.reserve(nMatches);

    for (const std::pair<int, int> &match: _matches)
    {
        matches.push_back(match);
    }

    std::vector<int> allIDs(nMatches, 0);  // 记录所有匹配对的id
    for (int i = 0; i < nMatches; ++i)
    {
        allIDs[i] = i;
    }

    // Step1: 从所有的匹配点对中随机选出8对用于计算F或H
    DUtils::Random::SeedRandOnce(0);
    octaMatches.resize(maxIterations);
    for (std::vector<int> &octaMatch: octaMatches)
    {
        octaMatch.resize(8);
    }

    for (int i = 0; i < maxIterations; ++i)
    {
        std::vector<int> allIDs1 = allIDs;

        for (int j = 0; j < 8; ++j)
        {
            int randi = DUtils::Random::RandomInt(0, allIDs1.size()-1);
            int id = allIDs1[randi];
            octaMatches[i][j] = id;

            // todo::其实这里可以这样干，因为有些keypoint离得很近，所以取8个匹配对时应该要保证它们相互之间在图像上不能离得太近，在这里算距离总比之后算F或H再验证要快
            allIDs1[randi] = allIDs1.back();  // 为了避免下一轮又取到allIDs中的第randi这个元素，把最末尾的元素赋给allIDs[randi]，再把最末尾的元素删除掉
            allIDs1.pop_back();
        }
    }

    // Step3: 对两帧上的keypoints做归一化操作，即调整成在x和y方向上都是均值为0，标准差为1的分布，这是为了计算的方便
    NormalizeKeypoints();

    // Step2: 同时开启两条线程计算H和F，根据两者的打分决定选用哪个模型
    //        两个子线程并行执行，谁先完不一定，主线程等着两个都完了再继续执行
    Eigen::Matrix3d H, F;
    float scoreH, scoreF;

    // 第一个参数是函数的地址，如果这个函数是类的成员函数，第二个参数应该是类的某个实例的指针或引用，因为类的成员函数可能会去调用成员变量，它怎么知道去调用哪个实例的成员变量，你必须告诉它
    // 用std::ref()是因为thread()的构造函数是模版函数，用这种方式告诉模板实例化出一个形参是引用变量的函数，这样在thread对象在内部再把这个变量传给ComputeHomography()函数时，修改的是外部的这个scoreH，而不是构造函数生成的一个临时变量
    std::vector<bool> bValidMatch;
    std::thread threadH(&Initializer::ComputeHomography, this, std::ref(scoreH), std::ref(H));
    std::thread threadF(&Initializer::ComputeFundamental, this, std::ref(scoreF), std::ref(F));

    threadH.join();  //　其实两个谁先完不一定，谁先完谁先合入主线程
    threadF.join();

//    cout << "final scoreF = " << scoreF << endl;
    ReconstructF(F, R, t, vMapPoints);


    // Step3: 根据F与H两个模型的打分决定使用哪一个模型
    if (scoreH / (scoreF + scoreH) > 0.4)
    {
//        ReconstructFromH();
    }
    else
    {
//        ReconstructF(F);
    }

}

void Initializer::ComputeHomography(float &score, Eigen::Matrix3d &H) const
{

}

void Initializer::ComputeFundamental(float &score, Eigen::Matrix3d &F_)
{
    score = 0.0;
    std::vector<Eigen::Vector2d> kpts1(8);
    std::vector<Eigen::Vector2d> kpts2(8);

    for (const std::vector<int> &octaMatch: octaMatches)
    {
        Eigen::Matrix3d iniF = Eigen::Matrix3d::Zero();

        for (int i = 0; i < 8; ++i)
        {
            kpts1[i] = keypoints1[matches[octaMatch[i]].first];
            kpts2[i] = keypoints2[matches[octaMatch[i]].second];
        }

        Eigen::Matrix3d F = ComputeF(kpts1, kpts2);
//        Eigen::Matrix3d F1 = T2.transpose() * F * T1;

        std::vector<bool> bValidMatch1(matches.size(), true);
        double currentScore = EvaluateF(F, 1.0, bValidMatch1);

        if (currentScore > score)
        {
            score = currentScore;
            F_ = F;
            bValidMatch = bValidMatch1;
        }
    }
}

void Initializer::NormalizeKeypoints()
{
    NormalizeKptOnFrame(initialFrame, keypoints1, T1);
    NormalizeKptOnFrame(currentFrame, keypoints2, T2);
}

void Initializer::NormalizeKptOnFrame(const Frame &frame, std::vector<Eigen::Vector2d> &kpts, Eigen::Matrix3d &T) const
{

//    // Step1: 计算x,y方向的分布均值
//    float sumX(0.0), sumY(0.0);
//    int N = frame.numKeypoints;
//
//    for (const cv::KeyPoint &kpt: frame.keypoints)
//    {
//        // todo:: 在这里查看一下kpt的坐标是否都为int
//        sumX += kpt.pt.x;
//        sumY += kpt.pt.y;
//    }
//
//    float meanX = sumX / N;
//    float meanY = sumY / N;
//
//    //　Step2: x,y方向均值都调整为0，同时计算两个方向的标准差
//    float sumDevX(0.0), sumDevY(0.0);     // deviation　偏差
//    kpts.clear();
//    kpts.resize(N);
//    for (int i = 0; i < N; ++i)
//    {
//        kpts[i].x() = frame.keypoints[i].pt.x - meanX;
//        kpts[i].y() = frame.keypoints[i].pt.y - meanY;
//
//        sumDevX += fabs(kpts[i].x());
//        sumDevY += fabs(kpts[i].y());
//    }
//
//    float devX = sumDevX / N;
//    float devY = sumDevY / N;
//    float sX = 1 / devX;
//    float sY = 1 / devY;
//
//    // Step3: 把两个方向上分布的标准差都调整为1
//    for (Eigen::Vector2d &kpt: kpts)
//    {
//        kpt.x() *= sX;
//        kpt.y() *= sY;
//    }
//
//    T(0,0) = sX;
//    T(1,1) = sY;
//    T(0,2) = -meanX*sX;
//    T(1,2) = -meanY*sY;


    int N = frame.numKeypoints;
    kpts.clear();
    kpts.resize(N);
    for (int i = 0; i < N; ++i)
    {
        kpts[i].x() = frame.keypoints[i].pt.x;
        kpts[i].y() = frame.keypoints[i].pt.y;
    }
}

Eigen::Matrix3d Initializer::ComputeF(const std::vector<Eigen::Vector2d> &kpts1, const std::vector<Eigen::Vector2d> &kpts2) const
{
    // Step1: 构造系数矩阵A,由8对匹配点的2D坐标组成
    Eigen::Matrix<double, 8, 9> A;
    float u1(0.0), v1(0.0), u2(0.0), v2(0.0);
    Eigen::Matrix<double, 1, 9> a = Eigen::Matrix<double, 1, 9>::Zero();

    for (int i = 0; i < 8; ++i)
    {
        u1 = kpts1[i].x(); v1 = kpts1[i].y();
        u2 = kpts2[i].x(); v2 = kpts2[i].y();
//        cout << u1 << ", " << v1 << ", " << endl;
//        cout << u2 << ", " << v2 << ", " << endl;
        a << u1*u2, u2*v1, u2, u1*v2, v1*v2, v2, u1, v1, 1;
        A.row(i) = a;
    }

    // Step2: 解出基础矩阵F，当然由于误差的存在可能不精确
    Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 9> V1 = svd.matrixV();
    Eigen::Matrix<double, 8, 8> U1 = svd.matrixU();
    Eigen::Matrix<double, 8, 1> sigmas1 = svd.singularValues();
    Eigen::Matrix<double, 9, 1> h = V1.col(8);
    Eigen::Matrix3d iniF = Eigen::Matrix3d::Zero();
    iniF.row(0) = h.topRows(3).transpose();
    iniF.row(1) = h.block<3,1>(3,0).transpose();
    iniF.row(2) = h.bottomRows(3).transpose();

    // Step3: F的特征值肯定是[1, 1, 0]形式的，所以把初始的F调整成这种新式
    Eigen::JacobiSVD<Eigen::Matrix3d> svd2(iniF, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d V = svd2.matrixV();
    Eigen::Matrix3d U = svd2.matrixU();
    Eigen::Vector3d sigmas = svd2.singularValues();
    float sigma1 = sigmas(0);
    float sigma2 = sigmas(1);
//    float sigma = (sigma1 + sigma2);
//    Eigen::Matrix3d S = U.inverse() * iniF * V;
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
//    S(0,0) = S(1,1) = sigma;    // 前两个特征值应该相等
    S(0,0) = sigma1;
    S(1,1) = sigma2;
    Eigen::Matrix3d F = U * S * V.transpose();

//    cout << "F result is\n";
//    for (int i = 0; i < 8; ++i)
//    {
//        cout << Eigen::Vector3d(kpts2[i].x(), kpts2[i].y(), 1).transpose() * iniF * Eigen::Vector3d(kpts1[i].x(), kpts1[i].y(), 1) << " ";
//    }
//    cout << endl << endl;

    return iniF;
}

double Initializer::EvaluateF(const Eigen::Matrix3d &F, double sigma, std::vector<bool>& bValidMatch)
{
    const float thresh = 3.841;
    const float thScore = 5.991;
    const double invSigmaSquare = 1/(sigma*sigma);
    double sumScore = 0.0;

    for (int i = 0; i < matches.size(); ++i)
    {
        const std::pair<int, int> &match = matches[i];
        Eigen::Vector2d p10 = keypoints1[match.first];
        Eigen::Vector2d p20 = keypoints2[match.second];
        Eigen::Vector3d p1 = Eigen::Vector3d(p10.x(), p10.y(), 1);  //　非齐次转化为齐次坐标
        Eigen::Vector3d p2 = Eigen::Vector3d(p20.x(), p20.y(), 1);

        // Step1: 通过F把p1投影到Frame2上，算出其对极线方程
        Eigen::Vector3d l2 = F*p1;

        // Step2: 算出p2到l1的距离作为重投影误差
        double a = l2.x();
        double b = l2.y();

//        double error = fabs(Eigen::Vector3d(u2, v2, 1).dot(l2)) / std::sqrt(pow(a, 2) + pow(b, 2));
        double error1 = pow(fabs(p2.dot(l2)), 2) / (pow(a, 2) + pow(b, 2));  // 开方运算的代价很高，所以只计算误差的平方
        double chiError1 = error1 * invSigmaSquare;

        // Step3: 跟上面一样，求出p2在Frame1中的对极线
        Eigen::Vector3d l1 = p2.transpose()*F;

        // Step4: p1到l1的距离
        a = l1.x();
        b = l1.y();
        double error2 = pow(fabs(l1.dot(p1)), 2) / (pow(a, 2) + pow(b, 2));
        double chiError2 = error2 * invSigmaSquare;

        if (chiError1 < thresh && chiError2 < thresh)
        {
            sumScore += 2*thScore - chiError1 - chiError2;        // error越小，score越高
        }
        else
        {
            bValidMatch[i] = false;
        }

//        cout << "error1 is " << error1 << endl;
//        cout << "error2 is " << error2 << endl;
    }

    return sumScore;
}

double Initializer::EvaluateF(const Eigen::Matrix3d &F, std::vector<int> octaMatch) const
{
    const float thresh = 3.841;
    const float thScore = 5.991;
    double sumScore = 0.0;

    for (int i = 0; i < 8; ++i)
    {
        Eigen::Vector2d p10 = keypoints1[matches[octaMatch[i]].first];
        Eigen::Vector2d p20 = keypoints2[matches[octaMatch[i]].second];

        Eigen::Vector3d p1 = Eigen::Vector3d(p10.x(), p10.y(), 1);  //　非齐次转化为齐次坐标
        Eigen::Vector3d p2 = Eigen::Vector3d(p20.x(), p20.y(), 1);

        // Step1: 通过F把p1投影到Frame2上，算出其对极线方程
        Eigen::Vector3d l2 = F * p1;

        // Step2: 算出p2到l1的距离作为重投影误差
        double a = l2.x();
        double b = l2.y();
        double error1 = pow(fabs(p2.dot(l2)), 2) / (pow(a, 2) + pow(b, 2));
        double chiError1 = error1;
        cout << "error1 is " << error1 << endl;

        if (chiError1 < thresh)
        {
            sumScore += thScore - chiError1;        // error越小，score越高
        }

        // Step3: 跟上面一样，求出p2在Frame1中的对极线
        Eigen::Vector3d l1 = p2.transpose() * F;

        // Step4: p1到l1的距离
        a = l1.x();
        b = l1.y();
        double error2 = pow(fabs(l1.dot(p1)), 2) / (pow(a, 2) + pow(b, 2));
        double chiError2 = error2;
        cout << "error2 is " << error2 << endl;

        if (chiError2 < thresh)
        {
            sumScore += thScore - chiError2;
        }
    }


    return sumScore;
}

bool Initializer::ReconstructF(const Eigen::Matrix3d &F, Eigen::Matrix3d &R, Eigen::Vector3d &t, std::vector<Eigen::Vector3d> &vMapPoints)
{
    // Step1: 根据F矩阵求出E矩阵
    cv::Mat K = initialFrame.K;
    Eigen::Matrix3d eigK = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            eigK(i,j) = K.at<float>(i,j);
        }
    }
    const Eigen::Matrix3d E = eigK.transpose()*F*eigK;

    // Step2: SVD分解E矩阵求出R,t
    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t1;
    SvdEssential(E, R1, R2, t1);

    // Step3: 从4种可能的R,t都重建一次，然后选择重建出可靠mapPoint最多的那对R,t
    std::vector<Eigen::Vector3d> mapPoints1, mapPoints2, mapPoints3, mapPoints4;
    double parallax1(0), parallax2(0), parallax3(0), parallax4(0);
    int n1 = SelectRt(R1, t1, eigK, mapPoints1, parallax1);
    int n2 = SelectRt(R1, -t1, eigK, mapPoints2, parallax2);
    int n3 = SelectRt(R2, t1, eigK, mapPoints3, parallax3);
    int n4 = SelectRt(R2, -t1, eigK, mapPoints4, parallax4);

    std::vector<int> vNumMapPoints(4, 0);
    vNumMapPoints[0] = n1;
    vNumMapPoints[1] = n2;
    vNumMapPoints[2] = n3;
    vNumMapPoints[3] = n4;
    std::sort(vNumMapPoints.begin(), vNumMapPoints.end());
    if (vNumMapPoints[3] < 40)
    {
        return false;
    }

    if (double(vNumMapPoints[2])/double(vNumMapPoints[3]) > 0.7)
    {
        return false;
    }

    if (n1 == vNumMapPoints[3])
    {
        cout << "maxGood = " << n1 << "\nparallax = " << parallax1 << endl;
//        cout << "R is\n" << R1 << "\nt is " << t1.transpose() << endl;
        if (parallax1 > 0.2)  // 视差要大于1°
        {
            R = R1;
            t = t1;
            vMapPoints = mapPoints1;
        }
        else
        {
            return false;
        }
    }
    else if(n2 == vNumMapPoints[3])
    {
        cout << "maxGood = " << n2 << "\nparallax = " << parallax2 << endl;
//        cout << "R is\n" << R1 << "\nt is " << -t1.transpose()<< endl;
        if (parallax2 > 0.2)  // 视差要大于1°
        {
            R = R1;
            t = -t1;
            vMapPoints = mapPoints2;
        }
        else
        {
            return false;
        }
    }
    else if(n3 == vNumMapPoints[3])
    {
        cout << "maxGood = " << n3 << "\nparallax = " << parallax3 << endl;
//        cout << "R is\n" << R2 << "\nt is " << t1.transpose()<< endl;
        if (parallax3 > 0.2)  // 视差要大于1°
        {
            R = R2;
            t = t1;
            vMapPoints = mapPoints3;
        }
        else
        {
            return false;
        }
    }
    else if(n4 == vNumMapPoints[3])
    {
        cout << "maxGood = " << n4 << "\nparallax = " << parallax4 << endl;
//        cout << "R is\n" << R2 << "\nt is " << -t1.transpose()<< endl;
        if (parallax4 > 0.2)  // 视差要大于1°
        {
            R = R2;
            t = -t1;
            vMapPoints = mapPoints4;
        }
        else
        {
            return false;
        }
    }

    return true;

}

void Initializer::SvdEssential(const Eigen::Matrix3d &E, Eigen::Matrix3d &R1, Eigen::Matrix3d &R2,
                               Eigen::Vector3d &t) const
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d sigmas = svd.singularValues();
    double sigma = (sigmas[0] + sigmas[1]) / 2;
//    cout << "sigmas is " << sigmas.transpose() << endl;
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    S(0,0) = S(1,1) = sigma; // 因为E的第三个奇异值一定为0，所以在这里强行设置

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    W(0,1) = -1; W(1,0) = 1; W(2,2) = 1;

    R1 = U*W*V.transpose();
    R1 /= R1.determinant();
    R2 = U*W.inverse()*V.transpose();
    R2 /= R2.determinant();
    t = U.col(2);
}

int Initializer::SelectRt(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Matrix3d &K,
                          std::vector<Eigen::Vector3d> &mapPoints, double &parallax_) const
{
    std::vector<double> vParallax;
    for (int i = 0; i < matches.size(); ++i)
    {
        // Step1: 根据RANSAC确定出来的正确的匹配点对，三角化出相应的mapPoint
        if (!bValidMatch[i])
        {
            continue;
        }

        Eigen::Vector2d p1 = keypoints1[matches[i].first];
        Eigen::Vector2d p2 = keypoints2[matches[i].second];
        Eigen::Vector3d mapPoint = triangulate(p1, p2, R, t, K);

        if (!std::isfinite(mapPoint.x()) || !std::isfinite(mapPoint.y()) || !std::isfinite(mapPoint.z()))
        {
            continue;
        }

        // Step2: 检查建出的mapPoint是否在两个相机的正前方
        // step2.1: 检查两个相机的视差，也就是两条视线的余弦cos
        Eigen::Vector3d Oc1(0,0,0);
        Eigen::Vector3d Oc2 = -R.transpose()*t;
        Eigen::Vector3d l1 = Oc1 - mapPoint; l1.normalize();
        Eigen::Vector3d l2 = Oc2 - mapPoint; l2.normalize();
        double parallax = l1.dot(l2);
        if (parallax > 0.9998)  // cos接近1，说明夹角几乎为0，视察太小，放弃三角化这个点
        {
//            continue;
        }

        // step2.2: 检查在初始帧的相机坐标系下，该点是否位于成像平面之前
//        if (mapPoint.z() < 0 && parallax < 0.9998)  // 如果视差不够大，可能计算机在算极大值或极小值时符号是随机的
        if (mapPoint.z() < 0)
        {
            continue;
        }

        // step2.3: 检查在第二帧的相机坐标系下，该点是否位于成像平面之前
        Eigen::Vector3d Xc = R*mapPoint + t;
//        if (Xc.z() < 0 && parallax < 0.9998)
        if (Xc.z() < 0)
        {
            continue;
        }

        // Step3: 检查建出的mapPoint在两帧中的重投影误差，因为我们是根据最小二乘法求出的mapPoint，所以它的重投影误差不会为0，就像这两条视线不相交，我们取它们的中点作为mapPoint
        Eigen::Vector3d reP1 = K*mapPoint;
        reP1 /= reP1.z();
        Eigen::Vector2d rep1 = reP1.topRows(2);
        Eigen::Vector3d reP2 = K*(R*mapPoint + t);
        reP2 /= reP2.z();
        Eigen::Vector2d rep2 = reP2.topRows(2);
        double reError1 = (rep1 - p1).norm();
        double reError2 = (rep2 - p2).norm();

        if (reError1 < 2 && reError2 < 2)   //　重投影误差小于两个像素
        {
            mapPoints.push_back(mapPoint);
            vParallax.push_back(parallax);
        }
    }

    // Step4: 看看是否有足够的mapPoint满足视差要求
    if (!mapPoints.empty())
    {
        std::sort(vParallax.begin(), vParallax.end());
        int idx = std::min(50, int(vParallax.size()-1));  // 选50是指如果有50个mapPoint的视差满足要求，则认为两帧之间的视差足够大了，之后的小视差的点也不care它们了
        parallax_ = std::acos(vParallax[idx]) * 180 / M_PI;
    }

    return mapPoints.size();
}

Eigen::Vector3d Initializer::triangulate(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Matrix3d &K) const
{
    // Step1: 建立系数矩阵A的后一项
    Eigen::Matrix<double, 3, 4> T1 = Eigen::Matrix<double, 3, 4>::Zero();
    T1.leftCols(3) = K;
    Eigen::Matrix<double, 3, 4> T2 = Eigen::Matrix<double, 3, 4>::Zero();
    T2.leftCols(3) = R; T2.col(3) = t;
    T2 = K * T2;
    Eigen::Matrix<double, 6, 4> T3 = Eigen::Matrix<double, 6, 4>::Zero();
    T3.topRows(3) = T1;
    T3.bottomRows(3) = T2;

    // Step2: 建立系数矩阵的前一项
    Eigen::Matrix<double, 4, 6> B = Eigen::Matrix<double, 4, 6>::Zero();
    Eigen::Matrix3d p1Hat = Sophus::SO3::hat(Eigen::Vector3d(p1.x(), p1.y(), 1));
    Eigen::Matrix3d p2Hat = Sophus::SO3::hat(Eigen::Vector3d(p2.x(), p2.y(), 1));
    B.topLeftCorner(2,3) = p1Hat.topRows(2);
    B.bottomRightCorner(2,3) = p2Hat.topRows(2);
    Eigen::Matrix<double, 4, 4> A = B * T3;

    // Step3: SVD分解系数矩阵，求出mapPoint坐标
    Eigen::JacobiSVD<Eigen::Matrix<double, 4, 4>> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix<double, 4, 4> V = svd.matrixV();
    Eigen::Matrix<double, 4, 1> pt = V.col(3);
    return (pt/pt[3]).topRows(3);   //　只取前三维，第四维是补充的齐次坐标，一定为1
}