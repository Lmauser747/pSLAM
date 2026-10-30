#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

#include <eigen3/Eigen/Dense>

#include "frontend/HfTracker.hpp"
#include "CoreTypes.hpp"

namespace PoolSLAM{

size_t HfTracker::trackExistingFeatures(const uint8_t *curBuffer, const Eigen::Matrix3f &R){
    if(curBuffer == nullptr){
        handleTrackingError();
        return 0;
    }

    // Run IC-LK if we have previous features to track
    if(!activeFeatures.empty()){
        Eigen::Matrix3f prior = K * R * KInv;

        size_t survivorCount = 0;
        for(size_t i = 0; i < activeFeatures.size(); i++){
            auto &feature = activeFeatures[i];
            Eigen::Vector2f ptEst = (prior * feature.ptCurr.homogeneous()).hnormalized();
            if(alignFeatureICLK(feature, curBuffer, ptEst)){
                if(survivorCount != i){
                    activeFeatures[survivorCount] = std::move(activeFeatures[i]);
                }
                survivorCount++;
            }
        }

        activeFeatures.resize(survivorCount);
        
        // Run RANSAC to filter out outliers. 
        if(ransac(R) == 0){
            handleTrackingError();
            return 0;
        }

        // Remove features from buckets with too many features
        enforceBucketDensity();

        return activeFeatures.size();
    }else{
        return 0;
    }
}

bool HfTracker::alignFeatureICLK(TrackedFeature<5> &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst){
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
        // START VECTORIZE
        for(int dy = -RADIUS; dy <= RADIUS; dy++){
            for(int dx = -RADIUS; dx <= RADIUS; dx++){
                size_t idx = static_cast<size_t>((dy + RADIUS) * feature.PATCH_SIZE + (dx + RADIUS));
                Eigen::Vector2f offset{dy, dx};

                int16_t sampleI = bilinear(curBuffer, ptEst + offset);
                int16_t diff = sampleI - static_cast<int16_t>(feature.patch[idx]);
                madError = static_cast<uint32_t>(std::abs(diff));

                bX += static_cast<int32_t>(feature.xGrad[idx]) * diff;
                bY += static_cast<int32_t>(feature.yGrad[idx]) * diff;
            }
        }
        // END VECTORIZE

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

int16_t HfTracker::bilinear(const uint8_t *curBuffer, const Eigen::Vector2f &pt) const{
    float x0 = static_cast<float>(std::floor(pt.x()));
    float y0 = static_cast<float>(std::floor(pt.y()));

    uint32_t xWeight = static_cast<uint32_t>((pt.x() - x0) * 256.0f);
    uint32_t yWeight = static_cast<uint32_t>((pt.y() - y0) * 256.0f);

    uint32_t xWeightC = 256u - xWeight;
    uint32_t yWeightC = 256u - yWeight;

    uint32_t w00, w01, w10, w11;
    w00 = xWeightC * yWeightC;
    w01 = xWeightC * yWeight;
    w10 = xWeight * yWeightC;
    w11 = xWeight * yWeight;

    size_t x, y;
    x = static_cast<size_t>(x0);
    y = static_cast<size_t>(y0);

    uint32_t pixel = 
    (w00 * curBuffer[y * wImage + x]) + (w01 * curBuffer[y * wImage + x + 1]) +
    (w10 * curBuffer[(y + 1) * wImage + x]) + (w11 * curBuffer[(y + 1) * wImage + x + 1]);

    return static_cast<int16_t>(pixel >> 16);
}

int HfTracker::ransac(const Eigen::Matrix3f &R){
    const size_t numFeatures = activeFeatures.size();
    if(numFeatures < 2){
        return 0;
    }
    // Unpack AoS to SoA
    // START VECTORIZE
    for(size_t i = 0; i < numFeatures; i++){
        epi.xCurr[i] = activeFeatures[i].ptCurr.x();
        epi.yCurr[i] = activeFeatures[i].ptCurr.y();
        epi.xPrev[i] = activeFeatures[i].ptPrev.x();
        epi.yPrev[i] = activeFeatures[i].ptPrev.y();
    }
    // END VECTORIZE

    undistort(epi.xPrev, epi.yPrev, epi.prevRays, numFeatures);
    undistort(epi.xCurr, epi.yCurr, epi.currRays, numFeatures);

    // Rotate the rays from the previous frame to the current frame
    epi.scratch.noalias() = R * epi.prevRays.leftCols(numFeatures);

    // cross the rays from the current frame with the rays from the old frame
    for(size_t i = 0; i < numFeatures; i++){
        auto iInt = static_cast<Eigen::Index>(i); // Why does Eigen use int for indexing??
        Eigen::Vector3f qi = epi.scratch.col(iInt);
        epi.scratch.col(iInt) = qi.cross(epi.currRays.col(iInt));
    }
    // epi.scratch is the matrix M in the RASAC calcs
    int maxInliers = 0;
    Eigen::Vector3f t, tBest;
    for(int iter = 0; iter < RANSAC_ITERS; iter++){
        auto a = static_cast<Eigen::Index>((ransacRng() * numFeatures) >> 31);
        auto b = static_cast<Eigen::Index>((ransacRng() * numFeatures) >> 31);
        b = a == b ? (static_cast<size_t>(b) == numFeatures ? b - 1 : b + 1) : b;
        t = epi.scratch.col(a).cross(epi.scratch.col(b));
        if(t.norm() < RANSAC_TRANS_EP){
            continue;
        }
        t = t.normalized();
        epi.residual = epi.scratch.leftCols(numFeatures).transpose() * t;
        int inliers = 0;
        // START VECTORIZE
        for(size_t i = 0; i < numFeatures; i++){
            auto d = std::abs(epi.residual(static_cast<Eigen::Index>(i)));
            inliers = d < ransacTau ? inliers + 1 : inliers;
        }
        // END VECTORIZE
        if(inliers > maxInliers){
            maxInliers = inliers;
            tBest = t;
            if(maxInliers > static_cast<int>(RANSAC_EARLY_CONSENSUS * static_cast<float>(numFeatures))){
                break;
            }
        }
    }

    if(maxInliers > RANSAC_MIN_INLIERS){
        size_t writeIdx = 0;
        epi.residual = epi.scratch.leftCols(numFeatures).transpose() * tBest;
        for(size_t readIdx = 0; readIdx < numFeatures; readIdx++){
            auto d = epi.residual(static_cast<Eigen::Index>(readIdx));
            if(d < ransacTau){
                if(writeIdx != readIdx){
                    activeFeatures[writeIdx] = std::move(activeFeatures[readIdx]);
                    epi.currRays.col(static_cast<Eigen::Index>(writeIdx)) = epi.currRays.col(static_cast<Eigen::Index>(readIdx));
                }
                writeIdx++;
            }
        }
        activeFeatures.resize(writeIdx);
    }else{
        maxInliers = 0;
    }    

    return maxInliers;
}

void HfTracker::undistort(std::vector<float> &xDist, std::vector<float> &yDist, Eigen::Matrix3Xf &ptUndist, size_t count){
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / K(0, 0);
    float fyInv = 1.0f / K(1, 1);
    // START VECTORIZE
    for(size_t i = 0; i < count; i++){
        float xd = (xDist[i] - cx) * fxInv;
        float yd = (yDist[i] - cy) * fyInv;
        float x = xd;
        float y = yd;

        for(int iter = 0; iter < UNDISTORT_ITERS; iter++){
            float r2 = x * x + y * y;
            float r4 = r2 * r2;
            float xy = x * y;
            float radial = 1.0f + dist.k1 * r2 + dist.k2 * r4;

            float xEst = x * radial + 2.0f * dist.p1 * xy + dist.p2 * (r2 + 2 * x * x);
            float yEst = y * radial + 2.0f * dist.p2 * xy + dist.p1 * (r2 + 2 * y * y);

            x -= (xEst - xd);
            y -= (yEst - yd);
        }

        ptUndist(0, static_cast<Eigen::Index>(i)) = x;
        ptUndist(1, static_cast<Eigen::Index>(i)) = y;
        ptUndist(2, static_cast<Eigen::Index>(i)) = 1.0f;
    }
    // END VECTORIZE
}

void HfTracker::enforceBucketDensity(){
    bool featureRemoved = false;
    std::fill(gridOccupancy.begin(), gridOccupancy.end(), GridBucket{0, 0, 0});
    keep.assign(activeFeatures.size(), true);
    for(uint32_t i = 0; i < activeFeatures.size(); i++){
        auto feature = activeFeatures[i];
        auto &bucket = gridOccupancy[getBucketIdx(feature.ptCurr.x(), feature.ptCurr.y())];
        
        if(bucket.count == 0){
            bucket.olderIdx = i;
            bucket.count = 1;
        }else if(bucket.count == 1){
            // Remove features that are too close
            auto distL2 = (feature.ptCurr - activeFeatures[bucket.olderIdx].ptCurr).norm();
            if(distL2 < MIN_DIST_IN_BUCKET){
                if(activeFeatures[bucket.olderIdx].age < feature.age){
                    keep[bucket.olderIdx] = false;
                    bucket.olderIdx = i;
                    featureRemoved = true;
                }else{
                    keep[i] = false;
                    featureRemoved = true;
                }
            }else{
                bucket.count = 2;
                if(activeFeatures[bucket.olderIdx].age < feature.age){
                    bucket.youngerIdx = bucket.olderIdx;
                    bucket.olderIdx = i;
                }else{
                    bucket.youngerIdx = i;
                }
            }
        }else{
            featureRemoved = true;
            if(activeFeatures[bucket.olderIdx].age < feature.age){
                keep[bucket.youngerIdx] = false;
                bucket.youngerIdx = bucket.olderIdx;
                bucket.olderIdx = i;
            }else if(activeFeatures[bucket.youngerIdx].age < feature.age){
                keep[bucket.youngerIdx] = false;
                bucket.youngerIdx = i;
            }else{
                keep[i] = false;
            }
        }
    }

    // Compact active features
    // It is ok that the indexes in gridOccupancy become in invalidas they are no longer needed
    if(featureRemoved){
        size_t writeIdx = 0;
        for(size_t readIdx = 0; readIdx < activeFeatures.size(); readIdx++){
            if(keep[readIdx]){
                if(writeIdx != readIdx){
                    activeFeatures[writeIdx] = std::move(activeFeatures[readIdx]);
                }
                writeIdx++;
            }
        }
        activeFeatures.resize(writeIdx);
    }
}
}