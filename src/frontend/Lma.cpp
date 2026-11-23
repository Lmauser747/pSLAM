#include <cmath>
#include <algorithm>
#include <cstddef>

#include <Eigen/Dense>

#include "frontend/LfTracker.hpp"

// Use the Levenberg–Marquardt algorithm to find the optimal pose tranformation
namespace PoolSLAM {

namespace{
// Exponential map on se(3) -> SE(3)
inline Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1> &xi){
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();
    const double theta = phi.norm();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d V = Eigen::Matrix3d::Identity();

    if(theta > 1e-8){
        const Eigen::Vector3d axis = phi / theta;
        Eigen::Matrix3d K;
        K << 0.0, -axis.z(), axis.y(),
             axis.z(), 0.0, -axis.x(),
             -axis.y(), axis.x(), 0.0;
        const double sinT = std::sin(theta);
        const double cosT = std::cos(theta);

        R += sinT * K + (1.0 - cosT) * (K * K);
        V += ((1.0 - cosT) / theta) * K + ((theta - sinT) / theta) * (K * K);
    }

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = V * rho;
    return T;
}
}


// Refine the pose using the Levenberg–Marquardt algorithm
bool LfTracker::refinePoseLM(size_t numInliers, const Eigen::Matrix3d& K_cam, Eigen::Matrix<double, 3, 4>& Tcand, Eigen::Matrix<double, 6, 6>& outCovariance){
    if(numInliers < 4){
        return false;
    }

    constexpr int MAX_ITER = 8;
    constexpr double HUBER_THRESHOLD = 2.0;
    const double fx = K_cam(0, 0);
    const double fy = K_cam(1, 1);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 4>(0, 0) = Tcand;

    double lambda = 1e-3;

    auto computeError = [&](const Eigen::Matrix4d &currentT) -> double{
        double cost = 0.0;
        const Eigen::Matrix3d R = currentT.block<3, 3>(0, 0);
        const Eigen::Vector3d t = currentT.block<3, 1>(0, 3);

        for(int i = 0; static_cast<size_t>(i) < numInliers; i++){
            const Eigen::Vector3d PCam = R * matching.matchLandmarks.col(i) + t;
            if(PCam.z() <= 1e-4){
                return 1e12;
            }

            const Eigen::Vector2d pNorm = matching.curFeatures.col(i).hnormalized();
            const Eigen::Vector2d err(
                fx * (PCam.x() / PCam.z() - pNorm.x()),
                fy * (PCam.y() / PCam.z() - pNorm.y())
            );
            const double rNorm = err.norm();

            if(rNorm <= HUBER_THRESHOLD){
                cost += 0.5 * rNorm * rNorm;
            }else{
                cost += HUBER_THRESHOLD * (rNorm - 0.5 * HUBER_THRESHOLD);
            }
        }
        return cost;
    };

    double currentCost = computeError(T);

    for(int iter = 0; iter < MAX_ITER; iter++){
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();

        const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        const Eigen::Vector3d t = T.block<3, 1>(0, 3);

        for(int i = 0; static_cast<size_t>(i) < numInliers; i++){
            const Eigen::Vector3d PCam = R * matching.matchLandmarks.col(i) + t;
            if(PCam.z() <= 1e-4){
                continue;
            }
            const double invZ = 1.0 / PCam.z();
            const double invZ2 = invZ * invZ;
            const double x = PCam.x();
            const double y = PCam.y();

            const Eigen::Vector2d pNorm = matching.curFeatures.col(i).hnormalized();
            const Eigen::Vector2d err(
                fx * (x * invZ - pNorm.x()),
                fy * (y * invZ - pNorm.y())
            );
            const double errNorm = err.norm();

            const double w = (errNorm <= HUBER_THRESHOLD || errNorm < 1e-8) ? 1.0 : (HUBER_THRESHOLD / errNorm);

            Eigen::Matrix<double, 2, 6> J;
            J(0, 0) = fx * invZ;
            J(0, 1) = 0.0;
            J(0, 2) = -fx * x * invZ2;
            J(0, 3) = -fx * x * y * invZ2;
            J(0, 4) = fx * (1.0 + x * x * invZ2);
            J(0, 5) = -fx * y * invZ;

            J(1, 0) = 0.0;
            J(1, 1) = fy * invZ;
            J(1, 2) = -fy * y * invZ2;
            J(1, 3) = -fy * (1.0 + y * y * invZ2);
            J(1, 4) = fy * x * y * invZ2;
            J(1, 5) = fy * x * invZ;

            H.noalias() += w * (J.transpose() * J);
            g.noalias() -= w * (J.transpose() * err);
        }

        Eigen::Matrix<double, 6, 6> HDamped = H;
        for(int i = 0; i < 6; i++){
            HDamped(i, i) += lambda * std::max(1e-4, H(i, i));
        }

        const Eigen::Matrix<double, 6, 1> deltaXi = HDamped.ldlt().solve(g);
        if(deltaXi.squaredNorm() < 1e-12){
            break;
        }

        const Eigen::Matrix4d TTrial = expSE3(deltaXi) * T;
        const double trialCost = computeError(TTrial);

        if(trialCost < currentCost){
            lambda = std::max(lambda / 3.0, 1e-7);
            currentCost = trialCost;
            T = TTrial;
        }else{
            lambda = std::min(lambda * 10.0, 1e7);
        }
    }

    Tcand = T.block<3, 4>(0, 0);

    Eigen::Matrix<double, 6, 6> HFinal = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<3, 1>(0, 3);

    for(int i = 0; static_cast<size_t>(i) < numInliers; i++){
        const Eigen::Vector3d PCam = R * matching.matchLandmarks.col(i) + t;
        if(PCam.z() <= 1e-4){
            continue;
        }
        const double invZ = 1.0 / PCam.z();
        const double invZ2 = invZ * invZ;
        const double x = PCam.x();
        const double y = PCam.y();

        Eigen::Matrix<double, 2, 6> J;
        J << fx * invZ, 0.0, -fx * x * invZ2, -fx * x * y * invZ2, fx * (1.0 + x * x * invZ2), -fx * y * invZ,
             0.0, fy * invZ, -fy * y * invZ2, -fy * (1.0 + y * y * invZ2), fy * x * y * invZ2, fy * x * invZ;

        HFinal.noalias() += J.transpose() * J;
    }

    if(!HFinal.allFinite()){
        return false;
    }
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(HFinal);
    if(ldlt.info() != Eigen::Success || !ldlt.isPositive()){
        return false;
    }
    outCovariance = ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
    outCovariance = 0.5 * (outCovariance + outCovariance.transpose());
    if(!outCovariance.allFinite()){
        return false;
    }
    for(int i = 0; i < 6; ++i){
        if(outCovariance(i, i) <= 0.0){
            return false;
        }
    }
    return true;
}

}