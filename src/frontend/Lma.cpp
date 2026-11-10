#include <cmath>
#include <algorithm>
#include <cstddef>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtsam/geometry/Pose3.h>

#include "LfTracker.hpp"

// Use the Levenberg–Marquardt algorithm to find the optimal pose tranformation
namespace PoolSLAM {

namespace{
// Exponential map on se(3) -> SE(3)
inline Eigen::Matrix4d expSE3(const Eigen::Matrix<double, 6, 1>& xi){
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();
    const double theta = phi.norm();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d V = Eigen::Matrix3d::Identity();

    if (theta > 1e-8) {
        const Eigen::Vector3d axis = phi / theta;
        const Eigen::Matrix3d K = Eigen::Matrix3d{
            {0.0, -axis.z(), axis.y()},
            {axis.z(), 0.0, -axis.x()},
            {-axis.y(), axis.x(), 0.0}
        };
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



// Levenberg-Marquardt optimizer for refining Tcand over consensus inliers.
bool LfTracker::refinePoseLM(size_t numInliers, const Eigen::Matrix3d& K_cam, Eigen::Matrix<double, 3, 4>& Tcand, Eigen::Matrix<double, 6, 6>& outCovariance){
    constexpr int kMaxIter = 8;
    constexpr double kHuberThreshold = 2.0; // Inlier threshold in pixels
    const double fx = K_cam(0, 0), fy = K_cam(1, 1);
    const double cx = K_cam(0, 2), cy = K_cam(1, 2);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 4>(0, 0) = Tcand;

    double lambda = 1e-3;

    auto computeError = [&](const Eigen::Matrix4d& currentT) -> double{
        double cost = 0.0;
        const Eigen::Matrix3d R = currentT.block<3, 3>(0, 0);
        const Eigen::Vector3d t = currentT.block<3, 1>(0, 3);

        for (int inlierIdx = 0; static_cast<size_t>(inlierIdx) < numInliers; inlierIdx++){
            const Eigen::Vector3d P_cam = R * matching.matchLandmarks.col(inlierIdx) + t;
            if (P_cam.z() <= 1e-4) return 1e12; // Cheirality violation penalty

            const double u = fx * (P_cam.x() / P_cam.z()) + cx;
            const double v = fy * (P_cam.y() / P_cam.z()) + cy;
            const double r_norm = (Eigen::Vector2d(u, v) - matching.curFeatures.col(inlierIdx).hnormalized()).norm();

            // Huber robust loss
            if (r_norm <= kHuberThreshold) {
                cost += 0.5 * r_norm * r_norm;
            } else {
                cost += kHuberThreshold * (r_norm - 0.5 * kHuberThreshold);
            }
        }
        return cost;
    };

    double currentCost = computeError(T);

    for(int iter = 0; iter < kMaxIter; ++iter){
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();

        const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
        const Eigen::Vector3d t = T.block<3, 1>(0, 3);

        for(int inlierIdx = 0; static_cast<size_t>(inlierIdx) < numInliers; inlierIdx++) {
            const Eigen::Vector3d P_cam = R * matching.matchLandmarks.col(inlierIdx) + t;
            const double invZ = 1.0 / P_cam.z();
            const double invZ2 = invZ * invZ;
            const double x = P_cam.x(), y = P_cam.y();

            const double u = fx * x * invZ + cx;
            const double v = fy * y * invZ + cy;
            const Eigen::Vector2d err = Eigen::Vector2d(u, v) - matching.curFeatures.col(inlierIdx).hnormalized();
            const double errNorm = err.norm();

            // Huber weight
            const double w = (errNorm <= kHuberThreshold || errNorm < 1e-8) ? 1.0 : (kHuberThreshold / errNorm);

            // Compute 2x6 Jacobian
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

        // Apply Levenberg-Marquardt damping
        Eigen::Matrix<double, 6, 6> H_damped = H;
        for (int i = 0; i < 6; ++i) {
            H_damped(i, i) += lambda * std::max(1e-4, H(i, i));
        }

        // Solve damped normal equation
        const Eigen::Matrix<double, 6, 1> deltaXi = H_damped.ldlt().solve(g);

        if (deltaXi.squaredNorm() < 1e-12) {
            break; // Converged
        }

        const Eigen::Matrix4d T_trial = expSE3(deltaXi) * T;
        const double trialCost = computeError(T_trial);

        if (trialCost < currentCost){
            // Step accepted
            lambda = std::max(lambda / 3.0, 1e-7);
            currentCost = trialCost;
            T = T_trial;
        } else {
            // Step rejected
            lambda = std::min(lambda * 10.0, 1e7);
        }
    }

    Tcand = T.block<3, 4>(0, 0);

    // Compute final un-damped covariance for the factor graph
    Eigen::Matrix<double, 6, 6> H_final = Eigen::Matrix<double, 6, 6>::Zero();
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<3, 1>(0, 3);

    for(int inlierIdx = 0; static_cast<size_t>(inlierIdx) < numInliers; inlierIdx++){
        const Eigen::Vector3d P_cam = R * matching.matchLandmarks.col(inlierIdx) + t;
        const double invZ = 1.0 / P_cam.z(), invZ2 = invZ * invZ;
        const double x = P_cam.x(), y = P_cam.y();

        Eigen::Matrix<double, 2, 6> J;
        J << fx*invZ, 0.0, -fx*x*invZ2, -fx*x*y*invZ2, fx*(1.0 + x*x*invZ2), -fx*y*invZ,
             0.0, fy*invZ, -fy*y*invZ2, -fy*(1.0 + y*y*invZ2), fy*x*y*invZ2, fy*x*invZ;

        H_final.noalias() += J.transpose() * J;
    }

    outCovariance = H_final.inverse();
    return true;
}

}