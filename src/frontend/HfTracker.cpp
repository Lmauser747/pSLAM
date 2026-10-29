#include <cstddef>
#include <cstdint>
#include <optional>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <array>

#include <eigen3/Eigen/Dense>

#include "frontend/HfTracker.hpp"
#include "CoreTypes.hpp"
#include "simd/simd_wrappers.hpp"

namespace PoolSLAM{

size_t HfTracker::trackExistingFeatures(const uint8_t *curBuffer, const std::optional<Eigen::Matrix3f> &R){
    if(curBuffer == nullptr){
        handleTrackingError();
        return 0;
    }
    const bool usingPrior = R.has_value();
    // Run IC-LK if we have previous features to track
    if(!activeFeatures.empty()){
        Eigen::Matrix3f prior;
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
        
        // Run RANSAC to filter out outliers. 
        if(ransac(*R) != 0){
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

size_t HfTracker::replenishBuckets(const uint8_t *curBuffer){
    if(curBuffer == nullptr){
        return 0;
    }
    size_t bucketsOccipied = 0;
    
    for(size_t bucket = 0;  bucket < gridOccupancy.size(); bucket++){
        if(gridOccupancy[bucket].count == 0){
            auto offset = offsetFromBucket(bucket);
            fastPoints.clear();
            getFast(curBuffer, offset);
            auto kpCanidate = bestKp(curBuffer);
            if(!kpCanidate.has_value()){
                continue;
            }
            if(addKp(curBuffer, *kpCanidate)){
                gridOccupancy[bucket].count = 1;
                bucketsOccipied++;
            }
        }else{
            bucketsOccipied++;
        }
    }
    if(static_cast<float>(bucketsOccipied) / static_cast<float>(totalBuckets) < MIN_BUCKET_OCCUPANCY){
        handleTrackingError();
    }
    return bucketsOccipied;
}


FORCE_INLINE bool HfTracker::alignFeatureICLK(TrackedFeature<5> &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst){
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

FORCE_INLINE int16_t HfTracker::bilinear(const uint8_t *curBuffer, Eigen::Vector2f &pt){
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

int HfTracker::ransac(const Eigen::Matrix3f &R){
    const size_t numFeatures = activeFeatures.size();
    epi.xCurr.resize(numFeatures);
    epi.xPrev.resize(numFeatures);
    epi.yCurr.resize(numFeatures);
    epi.yPrev.resize(numFeatures);
    // Unpack AoS to SoA
    // START VECTORIZE
    for(size_t i = 0; i < numFeatures; i++){
        epi.xCurr[i] = activeFeatures[i].ptCurr.x();
        epi.yCurr[i] = activeFeatures[i].ptCurr.y();
        epi.xPrev[i] = activeFeatures[i].ptPrev.x();
        epi.yPrev[i] = activeFeatures[i].ptPrev.y();
    }
    // END VECTORIZE

    epi.currRays.resize(static_cast<Eigen::Index>(numFeatures));
    epi.prevRays.resize(static_cast<Eigen::Index>(numFeatures));
    epi.scratch.resize(static_cast<Eigen::Index>(numFeatures));
    epi.residual.resize(static_cast<Eigen::Index>(numFeatures));

    undistort(epi.xPrev, epi.yPrev, epi.prevRays, numFeatures);
    undistort(epi.xCurr, epi.yCurr, epi.prevRays, numFeatures);

    // Rotate the rays from the previous frame to the current frame
    epi.scratch.noalias() = R * epi.prevRays;

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
        auto a = static_cast<Eigen::Index>(ransacRng() % numFeatures);
        auto b = static_cast<Eigen::Index>(ransacRng() % numFeatures);
        b = a == b ? b + 1 : b;
        t = epi.scratch.col(a).cross(epi.scratch.col(b));
        if(t.norm() < RANSAC_TRANS_EP){
            continue;
        }
        t = t.normalized();
        epi.residual = epi.scratch.transpose() * t;
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
        epi.residual = epi.scratch.transpose() * tBest;
        for(size_t readIdx = 0; readIdx < numFeatures; readIdx++){
            auto d = epi.residual(static_cast<Eigen::Index>(readIdx));
            if(d < ransacTau){
                if(writeIdx != readIdx){
                    activeFeatures[writeIdx] = activeFeatures[readIdx];
                    epi.currRays.col(static_cast<Eigen::Index>(writeIdx)) = epi.currRays.col(static_cast<Eigen::Index>(readIdx));
                }
                writeIdx++;
            }
        }
        activeFeatures.resize(writeIdx);
        epi.currRays.conservativeResize(static_cast<Eigen::Index>(writeIdx));
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
        float yd = (yDist[i] = cy) * fyInv;
        float x = xd;
        float y = yd;

        for(int iter = 0; iter < UNDISTORT_ITERS; iter++){
            float r2 = x * x + y * x;
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
    keep.resize(activeFeatures.size(), true);
    for(size_t i = 0; i < activeFeatures.size(); i++){
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
            featureRemoved = 0;
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
                    activeFeatures[writeIdx] = activeFeatures[readIdx];
                }
                writeIdx++;
            }
        }
    }
}

size_t HfTracker::getFast(const uint8_t *curBuffer, size_t bucketOffset){
    std::array<uint8_t, 16> ring = {0};
    size_t nPts = 0;
    for(size_t y = 0; y < hBucket; y++){
        auto rowPtr = curBuffer + bucketOffset + y * wImage;
        for(size_t x = 0; x < wBucket; x++){
            auto centerPtr = rowPtr + x;
            auto center = *centerPtr;

            uint8_t upper = static_cast<uint8_t>(std::min(255, center + FAST_THRESHOLD));
            uint8_t lower = static_cast<uint8_t>(std::max(0, center - FAST_THRESHOLD));

            // Run the compass test to quickly tell if a keypoint is possible
            ring[0] = *(centerPtr + fastOffsets[0]);
            ring[4] = *(centerPtr + fastOffsets[4]);
            ring[8] = *(centerPtr + fastOffsets[8]);
            ring[12] = *(centerPtr + fastOffsets[12]);

            bool bCompass = ((ring[0] > upper) || (ring[8] > upper)) && ((ring[4] > upper) || (ring[12] > upper));
            bool dCompass = ((ring[0] < lower) || (ring[8] < lower)) && ((ring[4] < lower) || (ring[12] < lower));

            if(!(bCompass || dCompass)){
                continue;
            }


            // Extract masks for brighter and darker pixels
            uint32_t brighter = 0;
            uint32_t darker = 0;

            for(size_t i = 0; i < 16; i++){
                ring[i] = *(centerPtr + fastOffsets[i]);
                brighter |= (ring[i] > upper) ? 1u << i : 0u;
                darker |= (ring[i] < lower) ? 1u << i : 0u;
            }

            uint32_t brighter32 = (brighter << 16) | brighter;
            uint32_t darker32 = (darker << 16) | darker;

            // Check for runs of 9 consecutive bits using bit magic
            uint32_t brighterRun = ~0u;
            uint32_t darkerRun = ~0u;

            for(size_t i = 0; i < 9; i++){
                brighterRun &= brighter32 >> i;
                darkerRun &= darker32 >> i;
            }

            if(((brighterRun || darkerRun) & 0xFFFF)!= 0){
                fastPoints.push_back(Eigen::Vector2i(static_cast<int>(x), static_cast<int>(y)));
                nPts++;
            }
        }
    }

    return nPts;
}

std::optional<Eigen::Vector2i> HfTracker::bestKp(const uint8_t *curBuffer){
    // Initialize the gradients with one extra pixel on each size to make Sobel filter easier
    int Ixx, Iyy, Ixy;
    const size_t stride = TrackedFeature<RADIUS + 1>::NUM_PIXELS;
    float bestEigenvalue  = 0;
    Eigen::Vector2i bestPt;
    for(auto kp : fastPoints){
        size_t xPt = static_cast<size_t>(kp.x());
        size_t yPt = static_cast<size_t>(kp.y());

        auto topLeft = curBuffer + yPt * wImage + xPt; // Top left of patch

        // Find structure tensor
        for(Eigen::Index y = 0; y < ST_PATCH_SIZE; y++){
            auto rowPtr = topLeft + static_cast<size_t>(y) * wImage;
            for(Eigen::Index x = 0; x < ST_PATCH_SIZE; x++){
                int Ix = (2 * (*(rowPtr + x - 1) - *(rowPtr + x + 1)) +
                *(rowPtr + wImage + x - 1) + *(rowPtr - wImage + x - 1) - 
                *(rowPtr + wImage + x + 1) - *(rowPtr - wImage + x - 1));
                
                int Iy = (2 * *(rowPtr - wImage + x) - *(rowPtr + wImage + x) +
                *(rowPtr - wImage + x - 1) + *(rowPtr - wImage + x + 1) -
                *(rowPtr + wImage + x - 1) + *(rowPtr + wImage + x + 1));

                Ixx += Ix * Ix;
                Iyy += Iy * Iy;
                Ixy += Ix * Iy;
            }
        }

        // Compute Shi-Tomasi score
        float a = static_cast<float>(Ixx);
        float b = static_cast<float>(Ixy);
        float c = static_cast<float>(Iyy);

        float minEigenvalue = 0.5f * ((a + c) - std::sqrt((a - c) * (a - c) + + 4 * b * b));

        if(minEigenvalue > bestEigenvalue){
            bestEigenvalue = minEigenvalue;
            bestPt = kp;
        }
    }

    // Reject canidate points that are not good enough
    if(bestEigenvalue < ST_THRESHOLD){
        return std::nullopt;
    }else{
        return bestPt;
    }
}

}