#include <array>
#include <cmath>
#include <algorithm>
#include <optional>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "frontend/LfTracker.hpp"

namespace PoolSLAM{
namespace geometry{

constexpr double K_EPS = 1e-12;
constexpr double PI = 3.14159265358979323846;

struct DiagonalDecomp {
    double sigma1 = 0.0;
    double sigma2 = 0.0;
    Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
};

inline double solveLambdaTwistCubic(double a, double b, double c, double d){
    const double invA = 1.0 / a;
    const double a2 = b * invA;
    const double a1 = c * invA;
    const double a0 = d * invA;
    const double a2_3 = a2 / 3.0;
    
    const double p = a1 - (a2 * a2) / 3.0;
    const double q = a0 - (a1 * a2) / 3.0 + (2.0 * a2 * a2 * a2) / 27.0;
    const double delta = (q * q) * 0.25 + (p * p * p) / 27.0;

    double gamma = 0.0;
    if (delta <= 0.0) {
        const double sqrtNegP3 = std::sqrt(-p / 3.0);
        const double denom = 2.0 * sqrtNegP3 * (-p / 3.0);
        double w = (std::abs(denom) > K_EPS) ? (-q / denom) : 0.0;
        w = std::clamp(w, -1.0, 1.0);
        const double theta = std::acos(w);
        gamma = (2.0 * sqrtNegP3) * std::cos((theta - 2.0 * PI) / 3.0) - a2_3;
    } else {
        const double sqrtDelta = std::sqrt(delta);
        const double u = std::cbrt(-0.5 * q + sqrtDelta);
        const double v = std::cbrt(-0.5 * q - sqrtDelta);
        gamma = (u + v) - a2_3;
    }

    const double f = a0 + gamma * (a1 + gamma * (a2 + gamma));
    const double fPrime = a1 + gamma * (2.0 * a2 + 3.0 * gamma);
    if (std::abs(fPrime) > K_EPS) {
        gamma -= f / fPrime;
    }
    return gamma;
}

inline DiagonalDecomp diagonalizeD0(const Eigen::Matrix3d &D){
    DiagonalDecomp out;
    const double p1 = -(D(0,0) + D(1,1) + D(2,2));
    const double p0 = (D(0,0)*D(1,1) - D(0,1)*D(0,1)) + 
                      (D(0,0)*D(2,2) - D(0,2)*D(0,2)) + 
                      (D(1,1)*D(2,2) - D(1,2)*D(1,2));

    const double delta = std::max(0.0, p1 * p1 - 4.0 * p0);
    const double sqrtDelta = std::sqrt(delta);

    const double sigA = 0.5 * (-p1 + sqrtDelta);
    const double sigB = 0.5 * (-p1 - sqrtDelta);

    if (std::abs(sigA) >= std::abs(sigB)) {
        out.sigma1 = sigA; out.sigma2 = sigB;
    } else {
        out.sigma1 = sigB; out.sigma2 = sigA;
    }

    auto getEigenvector = [&](double r) -> Eigen::Vector3d{
        const double c = r * (r - (D(0,0) + D(1,1))) + (D(0,0)*D(1,1) - D(0,1)*D(0,1));
        if (std::abs(c) < K_EPS) {
            Eigen::Vector3d cross = Eigen::Vector3d(D(0,0)-r, D(0,1), D(0,2)).cross(Eigen::Vector3d(D(0,1), D(1,1)-r, D(1,2)));
            if (cross.norm() > K_EPS) return cross.normalized();
            return Eigen::Vector3d(D(0,0)-r, D(0,1), D(0,2)).cross(Eigen::Vector3d(D(0,2), D(1,2), D(2,2)-r)).normalized();
        }
        const double invC = 1.0 / c;
        const double a1 = (r * D(0,2) + D(0,1) * D(1,2) - D(0,2) * D(1,1)) * invC;
        const double a2 = (r * D(1,2) + D(0,1) * D(0,2) - D(0,0) * D(1,2)) * invC;
        const double scale = 1.0 / std::sqrt(a1 * a1 + a2 * a2 + 1.0);
        return Eigen::Vector3d(a1 * scale, a2 * scale, scale);
    };

    out.e1 = getEigenvector(out.sigma1);
    out.e2 = getEigenvector(out.sigma2);
    return out;
}

// Solves P3P with Lambda Twist + Cherality checks
std::optional<Eigen::Matrix<double, 3, 4>> solveP4PLambdaTwist(const std::array<Eigen::Vector3d, 4>& x, const std::array<Eigen::Vector3d, 4>& y){
    const double a01 = (x[0] - x[1]).norm();
    const double a02 = (x[0] - x[2]).norm();
    const double a12 = (x[1] - x[2]).norm();

    const double b01 = y[0].dot(y[1]);
    const double b02 = y[0].dot(y[2]);
    const double b12 = y[1].dot(y[2]);

    Eigen::Matrix3d M01 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d M02 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d M12 = Eigen::Matrix3d::Identity();
    M01(2, 2) = 0; M02(1, 1) = 0; M12(0, 0) = 0;
    
    M01(0, 1) = -b01; M01(1, 0) = -b01;
    M02(0, 2) = -b02; M02(2, 0) = -b02;
    M12(1, 2) = -b12; M12(2, 1) = -b12;

    const Eigen::Matrix3d D1 = M01 * a12 - M12 * a01;
    const Eigen::Matrix3d D2 = M02 * a12 - M12 * a02;

    const double a_coeff = D2.determinant();
    const double norm = (std::abs(a_coeff) > K_EPS) ? 1.0 / a_coeff : 1.0;
    const double a_norm = (std::abs(a_coeff) > K_EPS) ? 1.0 : a_coeff;

    const double b_coeff = norm * (
        D2.col(0).dot(D1.col(1).cross(D1.col(2))) +
        D2.col(1).dot(D1.col(2).cross(D1.col(0))) +
        D2.col(2).dot(D1.col(0).cross(D1.col(1)))
    );
    const double c_coeff = norm * (
        D1.col(0).dot(D2.col(1).cross(D2.col(2))) +
        D1.col(1).dot(D2.col(2).cross(D2.col(0))) +
        D1.col(2).dot(D2.col(0).cross(D2.col(1)))
    );
    const double d_norm = norm * D1.determinant();

    const double gamma = solveLambdaTwistCubic(a_norm, b_coeff, c_coeff, d_norm);
    const Eigen::Matrix3d D0 = D1 + gamma * D2;
    DiagonalDecomp decomp = diagonalizeD0(D0);

    const double s = std::sqrt(std::max(0.0, -decomp.sigma2 / decomp.sigma1));
    const std::array<double, 2> s_vals = {s, -s};
    const double a_ratio = (a01 * a01) / (a12 * a12);

    auto getTau = [&](double s_val) -> std::array<double, 4>{
        const Eigen::Vector3d h = decomp.e1 - s_val * decomp.e2;
        if (std::abs(h(0)) < K_EPS) return {-1.0, -1.0, 0.0, 0.0};

        const double invH0 = 1.0 / h(0);
        const double w0 = -h(1) * invH0;
        const double w1 = -h(2) * invH0;

        const double A = w1 * w1 - a_ratio;
        const double bPrime = w1 * (w0 - b01) + a_ratio * b12;
        const double C = (w0 * w0 - 2.0 * b01 * w0 + 1.0) - a_ratio;

        const double delta_prime = bPrime * bPrime - A * C;
        if (delta_prime < 0.0) return {-1.0, -1.0, w0, w1};

        const double sqrt_delta = std::sqrt(delta_prime);
        if (std::abs(A) < K_EPS) {
            if (std::abs(bPrime) > K_EPS) return {-0.5 * C / bPrime, -1.0, w0, w1};
            return {-1.0, -1.0, w0, w1};
        }

        const double sgn = (bPrime >= 0.0) ? 1.0 : -1.0;
        const double q = -(bPrime + sgn * sqrt_delta);
        return {q / A, (std::abs(q) > K_EPS) ? (C / q) : (q / A), w0, w1};
    };

    Eigen::Matrix3d X;
    X.col(0) = x[0] - x[1];
    X.col(1) = x[0] - x[2];
    X.col(2) = X.col(0).cross(X.col(1));
    if (std::abs(X.determinant()) < K_EPS) return std::nullopt;
    const Eigen::Matrix3d invX = X.inverse();

    auto getCandidate = [&](double tau, double w0, double w1) -> std::optional<Eigen::Matrix<double, 3, 4>>{
        const double denom = tau * (tau - 2.0 * b12) + 1.0;
        if (denom <= K_EPS) return std::nullopt;

        Eigen::Vector3d lambda;
        lambda(1) = a12 / std::sqrt(denom);
        lambda(2) = tau * lambda(1);
        lambda(0) = w0 * lambda(1) + w1 * lambda(2);

        if (lambda(0) <= K_EPS || lambda(1) <= K_EPS || lambda(2) <= K_EPS) return std::nullopt;

        for (int iter = 0; iter < 2; ++iter) {
            const double l0 = lambda(0), l1 = lambda(1), l2 = lambda(2);

            const double r0 = 0.5 * (l0 * l0 + l1 * l1 - a01 * a01) - b01 * l0 * l1;
            const double r1 = 0.5 * (l0 * l0 + l2 * l2 - a02 * a02) - b02 * l0 * l2;
            const double r2 = 0.5 * (l1 * l1 + l2 * l2 - a12 * a12) - b12 * l1 * l2;

            const double a_jac = l0 - b01 * l1;
            const double b_jac = l1 - b01 * l0;
            const double c_jac = l0 - b02 * l2;
            const double d_jac = l2 - b02 * l0;
            const double e_jac = l1 - b12 * l2;
            const double f_jac = l2 - b12 * l1;

            const double detJ = -(a_jac * d_jac * e_jac + b_jac * c_jac * f_jac);
            if (std::abs(detJ) < K_EPS) break;

            const double invDet = 1.0 / detJ;
            const double d_l0 = invDet * ( d_jac * e_jac * r0 + b_jac * f_jac * r1 - b_jac * d_jac * r2);
            const double d_l1 = invDet * ( c_jac * f_jac * r0 - a_jac * f_jac * r1 + a_jac * d_jac * r2);
            const double d_l2 = invDet * (-c_jac * e_jac * r0 + a_jac * e_jac * r1 + a_jac * b_jac * r2);

            lambda(0) += d_l0; lambda(1) += d_l1; lambda(2) += d_l2;

            if (lambda(0) <= K_EPS || lambda(1) <= K_EPS || lambda(2) <= K_EPS) return std::nullopt;
            if ((d_l0 * d_l0 + d_l1 * d_l1 + d_l2 * d_l2) < 1e-16) break;
        }

        const Eigen::Vector3d P0 = lambda(0) * y[0];
        const Eigen::Vector3d P1 = lambda(1) * y[1];
        const Eigen::Vector3d P2 = lambda(2) * y[2];

        Eigen::Matrix3d Y;
        Y.col(0) = P0 - P1;
        Y.col(1) = P0 - P2;
        Y.col(2) = Y.col(0).cross(Y.col(1));

        Eigen::Matrix<double, 3, 4> T;
        T.leftCols<3>() = Y * invX;
        T.rightCols<1>() = P0 - T.leftCols<3>() * x[0];
        return T;
    };

    std::optional<Eigen::Matrix<double, 3, 4>> bestT = std::nullopt;
    double bestError = std::numeric_limits<double>::max();
    const Eigen::Vector2d y3Norm = (y[3] / y[3].z()).head<2>();

    for (double s_val : s_vals) {
        auto [tau1, tau2, w0, w1] = getTau(s_val);

        for (double tau : {tau1, tau2}) {
            if (tau <= 0.0) continue;
            auto optT = getCandidate(tau, w0, w1);
            if (!optT) continue;

            const Eigen::Matrix<double, 3, 4>& T = *optT;
            const Eigen::Vector3d P3_cam = T.leftCols<3>() * x[3] + T.rightCols<1>();
            
            if (P3_cam.z() <= K_EPS) continue; 
            
            const double err = ((P3_cam / P3_cam.z()).head<2>() - y3Norm).squaredNorm();
            if (err < bestError) {
                bestError = err;
                bestT = T;
            }
        }
    }

    return bestT;
}

}
} 