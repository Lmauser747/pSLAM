#include <cstddef>
#include <cstdint>
#include <optional>
#include <cmath>
#include <utility>

#include <eigen3/Eigen/Dense>

#include "frontend/HfTracker.hpp"
#include "CoreTypes.hpp"
#include "simd/simd_wrappers.hpp"

namespace PoolSLAM{

size_t HfTracker::trackAndExtract(const uint8_t *curBuffer, double timestamp, const std::optional<Eigen::Matrix3d> &R){
    if(curBuffer == nullptr){
        return false;
    }
    const bool usingPrior = R.has_value();
    if(!activeFeatures.empty()){
        Eigen::Matrix3d prior;
        if(usingPrior){
            prior = K * (*R) * KInv;
        }

        size_t survivorCount = 0;
        for(size_t i = 0; i < activeFeatures.size(); i++){
            auto feature = activeFeatures[i];
            auto ptEst = feature.ptPrev;
            if(usingPrior){
                ptEst = prior * ptEst.homogeneous();
            }
            if(alignFeatureICLK(feature, curBuffer, ptEst)){
                if(survivorCount != i){
                    activeFeatures[survivorCount] = std::move(activeFeatures[i]);
                }
                survivorCount++;
            }
        }

        activeFeatures.resize(survivorCount);
    }
}

FORCE_INLINE bool HfTracker::alignFeatureICLK(TrackedFeature<5> &feature, const uint8_t *curBuffer, Eigen::Vector2d &ptEst){
    const double ptLower = RADIUS + 1;
    const double ptUpperX = static_cast<double>(wImage) - RADIUS - 2;
    const double ptUpperY = static_cast<double>(hImage) - RADIUS - 2;

    bool converged = false;
    uint32_t madError = 0;

    for(int iter = 0; iter < ICLK_ITERS; iter++){
        if((ptEst.x() < ptLower) || (ptEst.y() < ptLower) || (ptEst.x() > ptUpperX) || (ptEst.y() > ptUpperY)){
            return false;
        }

        int32_t bX = 0;
        int32_t bY = 0;
        
        // Compute the accumulated error
        for(int dy = 0; dy <= RADIUS; dy++){
            for(int dx = 0; dx <= RADIUS; dx++){
                size_t idx = static_cast<size_t>((dy + RADIUS) * feature.PATCH_SIZE + (dx + 5));

                int16_t sampleI = bilinear(curBuffer, ptEst);
                int16_t diff = sampleI - static_cast<int16_t>(feature.patch[idx]);
                madError += static_cast<uint32_t>(std::abs(diff));

                bX += static_cast<int32_t>(feature.xGrad[idx]) * diff;
                bY += static_cast<int32_t>(feature.yGrad[idx]) * diff;
            }
        }

        Eigen::Vector2f b = {static_cast<float>(bX), static_cast<float>(bY)};
        
        Eigen::Vector2f delta = feature.HInv * b;

        // Update estimate based on the gradient
        ptEst -= delta;

        // Check for convergence
        float stepMagSqaured = delta.dot(delta);
        if(stepMagSqaured < ICLK_CONVERGENCE_THRESHOLD){
            converged =  true;
            break;
        }
    }

    if(!converged || (madError > ICLK_MAD_THRESHOLD)){
        return false;
    }

    if((ptEst.x() < ptLower) || (ptEst.y() < ptLower) || (ptEst.x() > ptUpperX) || (ptEst.y() > ptUpperY)){
        return false;
    }

    feature.ptPrev = feature.ptCurr;
    feature.ptCurr = ptEst;
    feature.age++;
    return true;
}

FORCE_INLINE int16_t HfTracker::bilinear(const uint8_t *curBuffer, Eigen::Vector2d pt){
    float x0 = static_cast<float>(std::floor(pt.x()));
    float y0 = static_cast<float>(std::floor(pt.y()));

    uint32_t xWeight = static_cast<uint32_t>((pt.x() - x0) * 256.0f);
    uint32_t yWeight = static_cast<uint32_t>((pt.y() - y0) * 256.0f);

    uint32_t xWeightC = 256u - xWeight;
    uint32_t yWeightC = 256u - yWeight;

    uint32_t w00, w01, w10, w11;
    w00 = xWeight * yWeight;
    w01 = xWeightC * yWeight;
    w10 = xWeight * yWeightC;
    w11 = xWeightC * yWeightC;

    size_t x, y;
    x = static_cast<size_t>(x0);
    y = static_cast<size_t>(y0);

    uint32_t pixel = 
    (w00 * curBuffer[y * wImage + x]) + (w01 * curBuffer[y * wImage + x + 1]) +
    (w10 * curBuffer[(y + 1) * wImage + x]) + (w11 * curBuffer[(y + 1) * wImage + x + 1]);

    return static_cast<int16_t>(pixel) >> 16;
}

}

