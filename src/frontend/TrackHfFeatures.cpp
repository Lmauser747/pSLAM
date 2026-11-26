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

float HfTracker::trackExistingFeatures(const uint8_t *curBuffer, const Eigen::Matrix3f &R){
    if(curBuffer == nullptr){
        handleTrackingError();
        return 0.0f;
    }

    buildPyramid(curBuffer, pyramids[curPyrIdx]);

    float meanDisplacement = 0.0f;
    // Run IC-LK if we have previous features to track
    if(!activeFeatures.empty()){
        Eigen::Matrix3f prior = K * R.transpose() * KInv;

        size_t survivorCount = 0;
        for(size_t i = 0; i < activeFeatures.size(); i++){
            auto &feature = activeFeatures[i];
            Eigen::Vector2f ptEst = (prior * feature.ptCurr.homogeneous()).hnormalized();
            if(alignFeaturePyramidICLK(feature, curBuffer, ptEst)){
                if(survivorCount != i){
                    activeFeatures[survivorCount] = std::move(activeFeatures[i]);
                }
                survivorCount++;
            }
        }

        activeFeatures.resize(survivorCount);
        
        ransac(R);

        // Remove features from buckets with too many features
        enforceBucketDensity();

        float totalDisplacement = 0.0f;
        for (const auto &feat : activeFeatures) {
            totalDisplacement += (feat.ptCurr - feat.ptPrev).norm();
        }
        meanDisplacement = activeFeatures.empty() ? 0.0f : (totalDisplacement / static_cast<float>(activeFeatures.size()));
    }

    // Flip ring index after tracking so prevPyrIdx holds curBuffer even on empty or recovered sets
    curPyrIdx ^= 1u;
    hasPrevPyramid = true;
    return meanDisplacement;
}

void HfTracker::buildPyramid(const uint8_t *curBuffer, ImagePyramid &dst){
    std::copy(curBuffer, curBuffer + wImage * hImage, dst.levels[0].data());
    for(size_t lvl = 1; lvl < PYR_LEVELS; lvl++){
        const size_t dstW = dst.widths[lvl];
        const size_t dstH = dst.heights[lvl];
        if(dstW == 0 || dstH == 0){
            continue;
        }
        const uint8_t *srcPtr = (lvl == 1) ? curBuffer : dst.levels[lvl - 1].data();
        downsampleLevel(srcPtr, dst.widths[lvl - 1], dst.heights[lvl - 1], dst.levels[lvl].data(), dstW, dstH);
    }
}

void HfTracker::downsampleLevel(const uint8_t *src, size_t srcW, size_t srcH, uint8_t *dst, size_t dstW, size_t dstH){
    const int maxX = static_cast<int>(srcW) - 1;
    const int maxY = static_cast<int>(srcH) - 1;

    // Blur only even columns needed for 2x horizontal decimation
    for(size_t y = 0; y < srcH; y++){
        const uint8_t *rowPtr = src + y * srcW;
        uint16_t *scratchRow = pyrBlurScratch.data() + y * dstW;
        for(size_t x = 0; x < dstW; x++){
            const int cx = static_cast<int>(x * 2);
            if(cx >= 2 && cx + 2 <= maxX){
                scratchRow[x] = static_cast<uint16_t>(
                    rowPtr[cx - 2] + 4 * rowPtr[cx - 1] + 6 * rowPtr[cx] + 4 * rowPtr[cx + 1] + rowPtr[cx + 2]
                );
            }else{
                const int xm2 = std::clamp(cx - 2, 0, maxX);
                const int xm1 = std::clamp(cx - 1, 0, maxX);
                const int x0 = std::clamp(cx, 0, maxX);
                const int xp1 = std::clamp(cx + 1, 0, maxX);
                const int xp2 = std::clamp(cx + 2, 0, maxX);
                scratchRow[x] = static_cast<uint16_t>(
                    rowPtr[xm2] + 4 * rowPtr[xm1] + 6 * rowPtr[x0] + 4 * rowPtr[xp1] + rowPtr[xp2]
                );
            }
        }
    }

    for(size_t y = 0; y < dstH; y++){
        const int cy = static_cast<int>(y * 2);
        const uint16_t *rm2 = pyrBlurScratch.data() + static_cast<size_t>(std::clamp(cy - 2, 0, maxY)) * dstW;
        const uint16_t *rm1 = pyrBlurScratch.data() + static_cast<size_t>(std::clamp(cy - 1, 0, maxY)) * dstW;
        const uint16_t *r0 = pyrBlurScratch.data() + static_cast<size_t>(std::clamp(cy, 0, maxY)) * dstW;
        const uint16_t *rp1 = pyrBlurScratch.data() + static_cast<size_t>(std::clamp(cy + 1, 0, maxY)) * dstW;
        const uint16_t *rp2 = pyrBlurScratch.data() + static_cast<size_t>(std::clamp(cy + 2, 0, maxY)) * dstW;
        uint8_t *dstRow = dst + y * dstW;

        for(size_t x = 0; x < dstW; x++){
            const uint32_t sum = static_cast<uint32_t>(rm2[x]) + 4u * rm1[x] + 6u * r0[x] + 4u * rp1[x] + rp2[x];
            dstRow[x] = static_cast<uint8_t>((sum + 128u) >> 8);
        }
    }
}

bool HfTracker::extractTemplate(
    const uint8_t *buffer,
    size_t bufW,
    size_t bufH,
    const Eigen::Vector2f &pt,
    std::array<uint8_t, TrackedFeature<RADIUS>::NUM_PIXELS> &patch,
    std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &xGrad,
    std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &yGrad,
    Eigen::Matrix2f &HInv) const{
    // Leave one extra pixel margin for Sobel taps and floor plus one bilinear lookup
    const float ptLower = static_cast<float>(RADIUS + 1);
    const float ptUpperX = static_cast<float>(bufW) - static_cast<float>(RADIUS) - 3.0f;
    const float ptUpperY = static_cast<float>(bufH) - static_cast<float>(RADIUS) - 3.0f;
    if(pt.x() < ptLower || pt.y() < ptLower || pt.x() > ptUpperX || pt.y() > ptUpperY){
        return false;
    }

    int Txx = 0;
    int Tyy = 0;
    int Txy = 0;
    constexpr int PATCH_SIZE = TrackedFeature<RADIUS>::PATCH_SIZE;

    for(int dy = -RADIUS; dy <= RADIUS; dy++){
        const size_t rowOffset = static_cast<size_t>((dy + RADIUS) * PATCH_SIZE);
        for(int dx = -RADIUS; dx <= RADIUS; dx++){
            const size_t idx = rowOffset + static_cast<size_t>(dx + RADIUS);
            const Eigen::Vector2f c = pt + Eigen::Vector2f(static_cast<float>(dx), static_cast<float>(dy));

            const float center = bilinearFloatStride(buffer, bufW, c);
            patch[idx] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(center)), 0, 255));

            // Match addKp left minus right and top minus bottom 8x Sobel convention
            const float tl = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(-1.0f, -1.0f));
            const float tc = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(0.0f, -1.0f));
            const float tr = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(1.0f, -1.0f));
            const float ml = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(-1.0f, 0.0f));
            const float mr = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(1.0f, 0.0f));
            const float bl = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(-1.0f, 1.0f));
            const float bc = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(0.0f, 1.0f));
            const float br = bilinearFloatStride(buffer, bufW, c + Eigen::Vector2f(1.0f, 1.0f));

            const int16_t tx = static_cast<int16_t>(std::lround(2.0f * (ml - mr) + bl + tl - br - tr));
            const int16_t ty = static_cast<int16_t>(std::lround(2.0f * (tc - bc) + tl + tr - bl - br));

            xGrad[idx] = tx;
            yGrad[idx] = ty;

            Txx += tx * tx;
            Tyy += ty * ty;
            Txy += tx * ty;
        }
    }

    Eigen::Matrix2f H{
        {static_cast<float>(Txx), static_cast<float>(Txy)},
        {static_cast<float>(Txy), static_cast<float>(Tyy)}
    };

    if(std::abs(H.determinant()) < HESSIAN_STABILITY_THRESHOLD){
        return false;
    }

    HInv = H.inverse();
    return true;
}

bool HfTracker::alignPatchICLK(
    const std::array<uint8_t, TrackedFeature<RADIUS>::NUM_PIXELS> &patch,
    const std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &xGrad,
    const std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> &yGrad,
    const Eigen::Matrix2f &HInv,
    const uint8_t *curBuffer,
    size_t bufW,
    size_t bufH,
    int maxIters,
    Eigen::Vector2f &ptEst,
    float &madError) const{
    const double ptLower = RADIUS + 1;
    const double ptUpperX = static_cast<double>(bufW) - RADIUS - 2;
    const double ptUpperY = static_cast<double>(bufH) - RADIUS - 2;

    madError = 0.0f;

    for(int iter = 0; iter < maxIters; iter++){
        madError = 0.0f;
        if((ptEst.x() < ptLower) || (ptEst.y() < ptLower) || (ptEst.x() > ptUpperX) || (ptEst.y() > ptUpperY)){
            return false;
        }

        float bX = 0.0f;
        float bY = 0.0f;

        // START VECTORIZE
        for(int dy = -RADIUS; dy <= RADIUS; dy++){
            for(int dx = -RADIUS; dx <= RADIUS; dx++){
                size_t idx = static_cast<size_t>((dy + RADIUS) * TrackedFeature<RADIUS>::PATCH_SIZE + (dx + RADIUS));
                const Eigen::Vector2f offset{dx, dy};

                float sampleI = bilinearFloatStride(curBuffer, bufW, ptEst + offset);
                float diff = sampleI - static_cast<float>(patch[idx]);
                madError += std::abs(diff);

                bX += static_cast<float>(xGrad[idx]) * diff;
                bY += static_cast<float>(yGrad[idx]) * diff;
            }
        }
        // END VECTORIZE

        Eigen::Vector2f b = {bX, bY};

        // Compensate for the 8x Sobel scaling
        Eigen::Vector2f delta = (HInv * b) * 8.0f;

        ptEst += delta;

        if(delta.squaredNorm() < 1e-4f){
            break;
        }
    }

    if((ptEst.x() < ptLower) || (ptEst.y() < ptLower) || (ptEst.x() > ptUpperX) || (ptEst.y() > ptUpperY)){
        return false;
    }

    return true;
}

bool HfTracker::alignFeatureICLK(TrackedFeature<5> &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst){
    float madError = 0.0f;
    if(!alignPatchICLK(feature.patch, feature.xGrad, feature.yGrad, feature.HInv,
                       curBuffer, wImage, hImage, ICLK_ITERS, ptEst, madError)){
        return false;
    }

    if(madError > static_cast<float>(ICLK_MAD_THRESHOLD)){
        return false;
    }

    feature.ptPrev = feature.ptCurr;
    feature.ptCurr = ptEst;
    feature.age++;
    return true;
}

bool HfTracker::alignFeaturePyramidICLK(TrackedFeature<RADIUS> &feature, const uint8_t *curBuffer, Eigen::Vector2f &ptEst){
    if(hasPrevPyramid){
        const auto &prevPyr = pyramids[prevPyrIdx()];
        const auto &curPyr = pyramids[curPyrIdx];

        std::array<uint8_t, TrackedFeature<RADIUS>::NUM_PIXELS> coarsePatch;
        std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> coarseXGrad;
        std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> coarseYGrad;
        Eigen::Matrix2f coarseHInv;

        for(int lvl = static_cast<int>(PYR_LEVELS) - 1; lvl >= 1; lvl--){
            const size_t uLvl = static_cast<size_t>(lvl);
            const size_t lvlW = curPyr.widths[uLvl];
            const size_t lvlH = curPyr.heights[uLvl];
            const float scale = 1.0f / static_cast<float>(1u << lvl);

            const Eigen::Vector2f ptPrevL = feature.ptCurr * scale;
            Eigen::Vector2f ptEstL = ptEst * scale;
            const Eigen::Vector2f ptPreLevelL = ptEstL;

            // Skip unusable coarse levels instead of dropping border features early
            if(!extractTemplate(prevPyr.levels[uLvl].data(), lvlW, lvlH, ptPrevL,
                                coarsePatch, coarseXGrad, coarseYGrad, coarseHInv)){
                continue;
            }

            float coarseMad = 0.0f;
            if(!alignPatchICLK(coarsePatch, coarseXGrad, coarseYGrad, coarseHInv,
                               curPyr.levels[uLvl].data(), lvlW, lvlH, ICLK_COARSE_ITERS, ptEstL, coarseMad)){
                continue;
            }

            if((ptEstL - ptPreLevelL).norm() > COARSE_MAX_SHIFT){
                continue;
            }

            ptEst = ptEstL / scale;
        }
    }

    return alignFeatureICLK(feature, curBuffer, ptEst);
}

float HfTracker::bilinearFloat(const uint8_t *curBuffer, const Eigen::Vector2f &pt) const{
    return bilinearFloatStride(curBuffer, wImage, pt);
}

float HfTracker::bilinearFloatStride(const uint8_t *curBuffer, size_t stride, const Eigen::Vector2f &pt) const{
    float x0 = static_cast<float>(std::floor(pt.x()));
    float y0 = static_cast<float>(std::floor(pt.y()));

    float fx = pt.x() - x0;
    float fy = pt.y() - y0;
    float fxC = 1.0f - fx;
    float fyC = 1.0f - fy;

    size_t x = static_cast<size_t>(x0);
    size_t y = static_cast<size_t>(y0);

    return (fxC * fyC * static_cast<float>(curBuffer[y * stride + x])) +
           (fx * fyC * static_cast<float>(curBuffer[y * stride + x + 1])) +
           (fxC * fy * static_cast<float>(curBuffer[(y + 1) * stride + x])) +
           (fx * fy * static_cast<float>(curBuffer[(y + 1) * stride + x + 1]));
}

int16_t HfTracker::bilinear(const uint8_t *curBuffer, const Eigen::Vector2f &pt) const{
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
    (w11 * curBuffer[y * wImage + x]) + (w10 * curBuffer[y * wImage + x + 1]) +
    (w01 * curBuffer[(y + 1) * wImage + x]) + (w00 * curBuffer[(y + 1) * wImage + x + 1]);

    return static_cast<int16_t>(pixel >> 16);
}

int HfTracker::ransac(const Eigen::Matrix3f &R){
    const size_t numFeatures = activeFeatures.size();
    if(numFeatures < 2){
        return 0;
    }
    epi.xCurr.resize(numFeatures);
    epi.yCurr.resize(numFeatures);
    epi.xPrev.resize(numFeatures);
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

    undistort(epi.xPrev, epi.yPrev, epi.prevRays, numFeatures);
    undistort(epi.xCurr, epi.yCurr, epi.currRays, numFeatures);

    // Rotate the rays from the previous frame to the current frame
    epi.scratch.noalias() = R.transpose() * epi.prevRays.leftCols(numFeatures);
    

    // Check for pure rotation
    int pureRotations = 0;

    for(Eigen::Index i = 0; i < static_cast<Eigen::Index>(numFeatures); i++){
        auto p = epi.scratch.col(i).hnormalized();
        auto q = epi.currRays.col(i).hnormalized();
        if((p - q).norm() <= ransacTau){
            pureRotations++;
        }
    }

    if(pureRotations >= RANSAC_MIN_INLIERS && pureRotations >= static_cast<int>(numFeatures / 2)){
        size_t writeIdx = 0;
        for(size_t readIdx = 0; readIdx < numFeatures; readIdx++){
            auto i = static_cast<Eigen::Index>(readIdx);
            auto p = epi.scratch.col(i).hnormalized();
            auto q = epi.currRays.col(i).hnormalized();
            if((p - q).norm() <= ransacTau){
                if(writeIdx != readIdx){
                    activeFeatures[writeIdx] = activeFeatures[readIdx];
                }
                activeFeatures[writeIdx].ptUndis = epi.currRays.col(static_cast<Eigen::Index>(readIdx)).hnormalized();
                writeIdx++;  
            }
        }

        activeFeatures.resize(writeIdx);

        return static_cast<int>(writeIdx);
    }

   
    // epi scratch holds the previous rays rotated by R.T
    int maxInliers = 0;
    Eigen::Vector3f t, tBest;
    for(int iter = 0; iter < RANSAC_ITERS; iter++){
        // Select two distinct reandom numbers
        auto a = static_cast<Eigen::Index>((static_cast<uint64_t>(ransacRng()) * numFeatures) >> 31);
        auto b = static_cast<Eigen::Index>((static_cast<uint64_t>(ransacRng()) * numFeatures) >> 31);
        b = a == b ? (static_cast<size_t>(b) == (numFeatures - 1) ? b - 1 : b + 1) : b;

        Eigen::Vector3f ma = epi.scratch.col(a).cross(epi.currRays.col(a));
        Eigen::Vector3f mb = epi.scratch.col(b).cross(epi.currRays.col(b));

        t = ma.cross(mb);
        if(t.norm() < RANSAC_TRANS_EP){
            continue;
        }

        t = t.normalized();

        int inliers = 0;
        // START VECTORIZE
        for(int i = 0; static_cast<size_t>(i) < numFeatures; i++){
            // Use the Sampson Epipolar Distance
            Eigen::Vector3f lCurr = t.cross(epi.scratch.col(i));
            Eigen::Vector3f lPrev = t.cross(epi.currRays.col(i));

            float numerator = epi.currRays.col(i).dot(lCurr);
            float numerator2 = numerator * numerator;

            float denomenator2 = lCurr.head<2>().squaredNorm() + lPrev.head<2>().squaredNorm();

            float sed = denomenator2 > 1e-7 ? numerator2 / denomenator2 : 1e9;

            inliers = (sed < ransacTau * ransacTau) ? inliers + 1 : inliers;
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

    if(maxInliers >= RANSAC_MIN_INLIERS || (numFeatures >= RANSAC_MIN_INLIERS && maxInliers >= 8)){
        size_t writeIdx = 0;
        for(size_t readIdx = 0; readIdx < numFeatures; readIdx++){
            auto i = static_cast<Eigen::Index>(readIdx);
            // Use the Sampson Epipolar Distance
            Eigen::Vector3f lCurr = tBest.cross(epi.scratch.col(i));
            Eigen::Vector3f lPrev = tBest.cross(epi.currRays.col(i));

            float numerator = epi.currRays.col(i).dot(lCurr);
            float numerator2 = numerator * numerator;

            float denomenator2 = lCurr.head<2>().squaredNorm() + lPrev.head<2>().squaredNorm();

            float sed = denomenator2 > 1e-7 ? numerator2 / denomenator2 : 1e9;

            if(sed < ransacTau * ransacTau){
                if(writeIdx != readIdx){
                    activeFeatures[writeIdx] = std::move(activeFeatures[readIdx]);
                    epi.currRays.col(static_cast<Eigen::Index>(writeIdx)) = epi.currRays.col(static_cast<Eigen::Index>(readIdx));
                }
                activeFeatures[writeIdx].ptUndis = epi.currRays.col(static_cast<Eigen::Index>(readIdx)).hnormalized();
                writeIdx++;
            }
        }
        activeFeatures.resize(writeIdx);
    }else if(numFeatures >= RANSAC_MIN_INLIERS){
        handleTrackingError();
    }else{
        for(size_t i = 0; i < numFeatures; i++){
            activeFeatures[i].ptUndis = epi.currRays.col(static_cast<Eigen::Index>(i)).hnormalized();
        }
    }
        
    

    return maxInliers;
}

Eigen::Vector2f HfTracker::undistortPoint(float xDist, float yDist) const{
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / K(0, 0);
    float fyInv = 1.0f / K(1, 1);

    float xd = (xDist - cx) * fxInv;
    float yd = (yDist - cy) * fyInv;
    float x = xd;
    float y = yd;

    for(int iter = 0; iter < UNDISTORT_ITERS; iter++){
        float r2 = x * x + y * y;
        float r4 = r2 * r2;
        float xy = x * y;
        float radial = 1.0f + dist.k1 * r2 + dist.k2 * r4;

        float dxTangential = 2.0f * dist.p1 * xy + dist.p2 * (r2 + 2.0f * x * x);
        float dyTangential = 2.0f * dist.p2 * xy + dist.p1 * (r2 + 2.0f * y * y);

        if(std::abs(radial) > 1e-6f){
            x = (xd - dxTangential) / radial;
            y = (yd - dyTangential) / radial;
        }
    }

    return Eigen::Vector2f(x, y);
}

void HfTracker::undistort(std::vector<float> &xDist, std::vector<float> &yDist, Eigen::Matrix3Xf &ptUndist, size_t count){
    // START VECTORIZE
    for(size_t i = 0; i < count; i++){
        Eigen::Vector2f pt = undistortPoint(xDist[i], yDist[i]);

        ptUndist(0, static_cast<Eigen::Index>(i)) = pt.x();
        ptUndist(1, static_cast<Eigen::Index>(i)) = pt.y();
        ptUndist(2, static_cast<Eigen::Index>(i)) = 1.0f;
    }
    // END VECTORIZE
}

void HfTracker::enforceBucketDensity(){
    bool featureRemoved = false;
    std::fill(gridOccupancy.begin(), gridOccupancy.end(), GridBucket{0, UINT32_MAX, UINT32_MAX});
    keep.assign(activeFeatures.size(), true);
    for(uint32_t i = 0; i < activeFeatures.size(); i++){
        const auto &feature = activeFeatures[i];
        auto &bucket = gridOccupancy[getBucketIdx(feature.ptCurr.x(), feature.ptCurr.y())];
        
        if(bucket.count == 0){
            bucket.olderIdx = i;
            bucket.count = 1;
        }else if(bucket.count == 1){
            // Remove features that are too close
            auto distL2 = (feature.ptCurr - activeFeatures[bucket.olderIdx].ptCurr).norm();
            if(distL2 < MIN_DIST_IN_BUCKET){
                featureRemoved = true;
                if(activeFeatures[bucket.olderIdx].age < feature.age){
                    keep[bucket.olderIdx] = false;
                    bucket.olderIdx = i;
                }else{
                    keep[i] = false;
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
            // Remove features that are too close
            auto distOlderL2 = (feature.ptCurr - activeFeatures[bucket.olderIdx].ptCurr).norm();
            if(activeFeatures[bucket.olderIdx].age < feature.age){
                if(distOlderL2 < MIN_DIST_IN_BUCKET){
                    keep[bucket.olderIdx] = false;
                }else{
                    keep[bucket.youngerIdx] = false;
                    bucket.youngerIdx = bucket.olderIdx;
                }
                bucket.olderIdx = i;
            }else if(activeFeatures[bucket.youngerIdx].age < feature.age){
                if(distOlderL2 < MIN_DIST_IN_BUCKET){
                    keep[i] = false;
                }else{
                    keep[bucket.youngerIdx] = false;
                    bucket.youngerIdx = i;
                }
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

float HfTracker::computeKeyframeParallax(const Eigen::Matrix3f &R_kf_curr) const{
    if(activeFeatures.empty()){
        return 0.0f;
    }

    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const Eigen::Matrix3f R_curr_kf = R_kf_curr.transpose();

    std::vector<float> parallaxes;
    parallaxes.reserve(activeFeatures.size());

    for(const auto &feat : activeFeatures){
        const Eigen::Vector3f pRot = R_curr_kf * feat.ptKfUndis.homogeneous();
        if(pRot.z() <= 1e-6f){
            continue;
        }
        const Eigen::Vector2f pRotNorm = pRot.head<2>() / pRot.z();
        const Eigen::Vector2f diff = feat.ptUndis - pRotNorm;
        parallaxes.push_back(std::hypot(diff.x() * fx, diff.y() * fy));
    }

    if(parallaxes.empty()){
        return 0.0f;
    }

    const size_t mid = parallaxes.size() / 2;
    std::nth_element(parallaxes.begin(), parallaxes.begin() + static_cast<ptrdiff_t>(mid), parallaxes.end());
    return parallaxes[mid];
}

void HfTracker::markKeyframe(){
    const uint8_t *kfBuffer = hasPrevPyramid ? pyramids[prevPyrIdx()].levels[0].data() : nullptr;
    std::array<uint8_t, TrackedFeature<RADIUS>::NUM_PIXELS> newPatch;
    std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> newXGrad;
    std::array<int16_t, TrackedFeature<RADIUS>::NUM_PIXELS> newYGrad;
    Eigen::Matrix2f newHInv;

    for(auto &feat : activeFeatures){
        feat.ptKfUndis = feat.ptUndis;
        // Refresh level zero template at keyframes to prevent multi second drift
        if(kfBuffer != nullptr &&
           extractTemplate(kfBuffer, wImage, hImage, feat.ptCurr, newPatch, newXGrad, newYGrad, newHInv)){
            feat.patch = newPatch;
            feat.xGrad = newXGrad;
            feat.yGrad = newYGrad;
            feat.HInv = newHInv;
        }
    }
}
}