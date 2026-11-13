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

struct DiagonalDecomp{
    double sigma1 = 0.0;
    double sigma2 = 0.0;
    Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
};

inline double computeEigenProductP0(const Eigen::Matrix3d &D){
    return (D(0, 0) * D(1, 1) - D(0, 1) * D(0, 1)) +
           (D(0, 0) * D(2, 2) - D(0, 2) * D(0, 2)) +
           (D(1, 1) * D(2, 2) - D(1, 2) * D(1, 2));
}

inline Eigen::Matrix3d findSingularD0(const Eigen::Matrix3d &D1, const Eigen::Matrix3d &D2){
    const double c3 = D2.determinant();
    if(std::abs(c3) <= K_EPS){
        return D2;
    }

    const double invC3 = 1.0 / c3;
    // Coefficient of gamma^2 (one column from D1, two from D2)
    const double a2 = invC3 * (
        D1.col(0).dot(D2.col(1).cross(D2.col(2))) +
        D1.col(1).dot(D2.col(2).cross(D2.col(0))) +
        D1.col(2).dot(D2.col(0).cross(D2.col(1)))
    );
    // Coefficient of gamma^1 (two columns from D1, one from D2)
    const double a1 = invC3 * (
        D2.col(0).dot(D1.col(1).cross(D1.col(2))) +
        D2.col(1).dot(D1.col(2).cross(D1.col(0))) +
        D2.col(2).dot(D1.col(0).cross(D1.col(1)))
    );
    // Coefficient of gamma^0
    const double a0 = invC3 * D1.determinant();

    const double a2_3 = a2 / 3.0;
    const double p = a1 - (a2 * a2) / 3.0;
    const double q = a0 - (a1 * a2) / 3.0 + (2.0 * a2 * a2 * a2) / 27.0;
    const double delta = (q * q) * 0.25 + (p * p * p) / 27.0;

    auto polishRoot = [&](double g) -> double{
        for(int i = 0; i < 2; i++){
            const double f = a0 + g * (a1 + g * (a2 + g));
            const double fPrime = a1 + g * (2.0 * a2 + 3.0 * g);
            if(std::abs(fPrime) > K_EPS){
                g -= f / fPrime;
            }
        }
        return g;
    };

    double bestGamma = 0.0;
    if(delta <= 0.0){
        const double sqrtNegP3 = std::sqrt(std::max(0.0, -p / 3.0));
        const double denom = 2.0 * sqrtNegP3 * (-p / 3.0);
        double w = (std::abs(denom) > K_EPS) ? (-q / denom) : 0.0;
        w = std::clamp(w, -1.0, 1.0);
        const double theta = std::acos(w);

        double minP0 = std::numeric_limits<double>::max();
        for(int k = 0; k < 3; k++){
            double g = polishRoot((2.0 * sqrtNegP3) * std::cos((theta + 2.0 * PI * k) / 3.0) - a2_3);
            const double p0 = computeEigenProductP0(D1 + g * D2);
            if(p0 < minP0){
                minP0 = p0;
                bestGamma = g;
            }
        }
    }else{
        const double sqrtDelta = std::sqrt(delta);
        const double u = std::cbrt(-0.5 * q + sqrtDelta);
        const double v = std::cbrt(-0.5 * q - sqrtDelta);
        bestGamma = polishRoot((u + v) - a2_3);
    }

    return D1 + bestGamma * D2;
}

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
    if(delta <= 0.0){
        const double sqrtNegP3 = std::sqrt(-p / 3.0);
        const double denom = 2.0 * sqrtNegP3 * (-p / 3.0);
        double w = (std::abs(denom) > K_EPS) ? (-q / denom) : 0.0;
        w = std::clamp(w, -1.0, 1.0);
        const double theta = std::acos(w);
        gamma = (2.0 * sqrtNegP3) * std::cos((theta - 2.0 * PI) / 3.0) - a2_3;
    }else{
        const double sqrtDelta = std::sqrt(delta);
        const double u = std::cbrt(-0.5 * q + sqrtDelta);
        const double v = std::cbrt(-0.5 * q - sqrtDelta);
        gamma = (u + v) - a2_3;
    }

    const double f = a0 + gamma * (a1 + gamma * (a2 + gamma));
    const double fPrime = a1 + gamma * (2.0 * a2 + 3.0 * gamma);
    if(std::abs(fPrime) > K_EPS){
        gamma -= f / fPrime;
    }
    return gamma;
}

inline DiagonalDecomp diagonalizeD0(const Eigen::Matrix3d &D){
    DiagonalDecomp out;
    const double p1 = -(D(0, 0) + D(1, 1) + D(2, 2));
    const double p0 = (D(0, 0) * D(1, 1) - D(0, 1) * D(0, 1)) +
                      (D(0, 0) * D(2, 2) - D(0, 2) * D(0, 2)) +
                      (D(1, 1) * D(2, 2) - D(1, 2) * D(1, 2));

    const double delta = std::max(0.0, p1 * p1 - 4.0 * p0);
    const double sqrtDelta = std::sqrt(delta);

    const double sigA = 0.5 * (-p1 + sqrtDelta);
    const double sigB = 0.5 * (-p1 - sqrtDelta);

    if(std::abs(sigA) >= std::abs(sigB)){
        out.sigma1 = sigA;
        out.sigma2 = sigB;
    }else{
        out.sigma1 = sigB;
        out.sigma2 = sigA;
    }

    auto getEigenvector = [&](double r) -> Eigen::Vector3d{
        const double c = r * (r - (D(0, 0) + D(1, 1))) + (D(0, 0) * D(1, 1) - D(0, 1) * D(0, 1));
        if(std::abs(c) < K_EPS){
            const Eigen::Vector3d r0(D(0, 0) - r, D(0, 1), D(0, 2));
            const Eigen::Vector3d r1(D(0, 1), D(1, 1) - r, D(1, 2));
            const Eigen::Vector3d r2(D(0, 2), D(1, 2), D(2, 2) - r);

            const Eigen::Vector3d c01 = r0.cross(r1);
            const Eigen::Vector3d c02 = r0.cross(r2);
            const Eigen::Vector3d c12 = r1.cross(r2);

            const double n01 = c01.squaredNorm();
            const double n02 = c02.squaredNorm();
            const double n12 = c12.squaredNorm();

            if(n01 >= n02 && n01 >= n12){
                return (n01 > K_EPS * K_EPS) ? c01.normalized() : Eigen::Vector3d::UnitX();
            }else if(n02 >= n12){
                return (n02 > K_EPS * K_EPS) ? c02.normalized() : Eigen::Vector3d::UnitY();
            }
            return (n12 > K_EPS * K_EPS) ? c12.normalized() : Eigen::Vector3d::UnitZ();
        }
        const double invC = 1.0 / c;
        const double a1 = (r * D(0, 2) + D(0, 1) * D(1, 2) - D(0, 2) * D(1, 1)) * invC;
        const double a2 = (r * D(1, 2) + D(0, 1) * D(0, 2) - D(0, 0) * D(1, 2)) * invC;
        const double scale = 1.0 / std::sqrt(a1 * a1 + a2 * a2 + 1.0);
        return Eigen::Vector3d(a1 * scale, a2 * scale, scale);
    };

    out.e1 = getEigenvector(out.sigma1);
    out.e2 = getEigenvector(out.sigma2);
    return out;
}

std::optional<Eigen::Matrix<double, 3, 4>> solveP4PLambdaTwist(const std::array<Eigen::Vector3d, 4>& x, const std::array<Eigen::Vector3d, 4>& y){
    const std::array<Eigen::Vector3d, 3> u = {
        y[0].normalized(),
        y[1].normalized(),
        y[2].normalized()
    };

    const double a01 = (x[0] - x[1]).norm();
    const double a02 = (x[0] - x[2]).norm();
    const double a12 = (x[1] - x[2]).norm();
    if(a01 < K_EPS || a02 < K_EPS || a12 < K_EPS){
        return std::nullopt;
    }

    const double b01 = u[0].dot(u[1]);
    const double b02 = u[0].dot(u[2]);
    const double b12 = u[1].dot(u[2]);

    Eigen::Matrix3d M01 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d M02 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d M12 = Eigen::Matrix3d::Identity();
    M01(2, 2) = 0.0;
    M02(1, 1) = 0.0;
    M12(0, 0) = 0.0;

    M01(0, 1) = -b01; M01(1, 0) = -b01;
    M02(0, 2) = -b02; M02(2, 0) = -b02;
    M12(1, 2) = -b12; M12(2, 1) = -b12;

    const double a01Sq = a01 * a01;
    const double a02Sq = a02 * a02;
    const double a12Sq = a12 * a12;

    const Eigen::Matrix3d D1 = M01 * a12Sq - M12 * a01Sq;
    const Eigen::Matrix3d D2 = M02 * a12Sq - M12 * a02Sq;

    const double c3 = D2.determinant();
    if(std::abs(c3) <= K_EPS){
        return std::nullopt;
    }
    const double c2 = (
        D1.col(0).dot(D2.col(1).cross(D2.col(2))) +
        D1.col(1).dot(D2.col(2).cross(D2.col(0))) +
        D1.col(2).dot(D2.col(0).cross(D2.col(1)))
    );
    const double c1 = (
        D2.col(0).dot(D1.col(1).cross(D1.col(2))) +
        D2.col(1).dot(D1.col(2).cross(D1.col(0))) +
        D2.col(2).dot(D1.col(0).cross(D1.col(1)))
    );
    const double c0 = D1.determinant();

    const double gamma = solveLambdaTwistCubic(c3, c2, c1, c0);
    const Eigen::Matrix3d D0 = D1 + gamma * D2;
    DiagonalDecomp decomp = diagonalizeD0(D0);

    if(std::abs(decomp.sigma1) < K_EPS){
        return std::nullopt;
    }

    const double s = std::sqrt(std::max(0.0, -decomp.sigma2 / decomp.sigma1));
    const std::array<double, 2> sVals = {s, -s};
    const double aRatio = a01Sq / a12Sq;

    auto getTau = [&](double sVal) -> std::array<double, 4>{
        const Eigen::Vector3d h = decomp.e1 - sVal * decomp.e2;
        if(std::abs(h(0)) < K_EPS){
            return {-1.0, -1.0, 0.0, 0.0};
        }

        const double invH0 = 1.0 / h(0);
        const double w0 = -h(1) * invH0;
        const double w1 = -h(2) * invH0;

        const double A = w1 * w1 - aRatio;
        const double bPrime = w1 * (w0 - b01) + aRatio * b12;
        const double C = (w0 * w0 - 2.0 * b01 * w0 + 1.0) - aRatio;

        const double deltaPrime = bPrime * bPrime - A * C;
        if(deltaPrime < 0.0){
            return {-1.0, -1.0, w0, w1};
        }

        const double sqrtDelta = std::sqrt(deltaPrime);
        if(std::abs(A) < K_EPS){
            if(std::abs(bPrime) > K_EPS){
                return {-0.5 * C / bPrime, -1.0, w0, w1};
            }
            return {-1.0, -1.0, w0, w1};
        }

        const double sgn = (bPrime >= 0.0) ? 1.0 : -1.0;
        const double q = -(bPrime + sgn * sqrtDelta);
        return {q / A, (std::abs(q) > K_EPS) ? (C / q) : (q / A), w0, w1};
    };

    Eigen::Matrix3d X;
    X.col(0) = x[0] - x[1];
    X.col(1) = x[0] - x[2];
    X.col(2) = X.col(0).cross(X.col(1));
    if(std::abs(X.determinant()) < K_EPS){
        return std::nullopt;
    }
    const Eigen::Matrix3d invX = X.inverse();

    auto getCandidate = [&](double tau, double w0, double w1) -> std::optional<Eigen::Matrix<double, 3, 4>>{
        const double denom = tau * (tau - 2.0 * b12) + 1.0;
        if(denom <= K_EPS){
            return std::nullopt;
        }

        Eigen::Vector3d lambda;
        lambda(1) = a12 / std::sqrt(denom);
        lambda(2) = tau * lambda(1);
        lambda(0) = w0 * lambda(1) + w1 * lambda(2);

        if(lambda(0) <= K_EPS || lambda(1) <= K_EPS || lambda(2) <= K_EPS){
            return std::nullopt;
        }

        for(int iter = 0; iter < 2; iter++){
            const double l0 = lambda(0);
            const double l1 = lambda(1);
            const double l2 = lambda(2);

            const double r0 = 0.5 * (l0 * l0 + l1 * l1 - a01Sq) - b01 * l0 * l1;
            const double r1 = 0.5 * (l0 * l0 + l2 * l2 - a02Sq) - b02 * l0 * l2;
            const double r2 = 0.5 * (l1 * l1 + l2 * l2 - a12Sq) - b12 * l1 * l2;

            const double aJac = l0 - b01 * l1;
            const double bJac = l1 - b01 * l0;
            const double cJac = l0 - b02 * l2;
            const double dJac = l2 - b02 * l0;
            const double eJac = l1 - b12 * l2;
            const double fJac = l2 - b12 * l1;

            const double detJ = -(aJac * dJac * eJac + bJac * cJac * fJac);
            if(std::abs(detJ) < K_EPS){
                break;
            }

            const double invDet = 1.0 / detJ;
            const double dL0 = invDet * ( dJac * eJac * r0 + bJac * fJac * r1 - bJac * dJac * r2);
            const double dL1 = invDet * ( cJac * fJac * r0 - aJac * fJac * r1 + aJac * dJac * r2);
            const double dL2 = invDet * (-cJac * eJac * r0 + aJac * eJac * r1 + bJac * cJac * r2);

            lambda(0) += dL0;
            lambda(1) += dL1;
            lambda(2) += dL2;

            if(lambda(0) <= K_EPS || lambda(1) <= K_EPS || lambda(2) <= K_EPS){
                return std::nullopt;
            }
            if((dL0 * dL0 + dL1 * dL1 + dL2 * dL2) < 1e-16){
                break;
            }
        }

        const Eigen::Vector3d P0 = lambda(0) * u[0];
        const Eigen::Vector3d P1 = lambda(1) * u[1];
        const Eigen::Vector3d P2 = lambda(2) * u[2];

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
    const Eigen::Vector2d y3Norm = y[3].hnormalized();

    for(size_t i = 0; i < sVals.size(); i++){
        auto [tau1, tau2, w0, w1] = getTau(sVals[i]);
        const std::array<double, 2> taus = {tau1, tau2};

        for(size_t j = 0; j < taus.size(); j++){
            const double tau = taus[j];
            if(tau <= 0.0){
                continue;
            }
            auto optT = getCandidate(tau, w0, w1);
            if(!optT.has_value()){
                continue;
            }

            const Eigen::Matrix<double, 3, 4> &T = *optT;
            const Eigen::Vector3d P3Cam = T.leftCols<3>() * x[3] + T.rightCols<1>();
            if(P3Cam.z() <= K_EPS){
                continue;
            }

            const double err = (P3Cam.hnormalized() - y3Norm).squaredNorm();
            if(err < bestError){
                bestError = err;
                bestT = T;
            }
        }
    }

    return bestT;
}

} // namespace geometry
} 