#include <cstddef>
#include <cstdint>
#include <optional>
#include <cmath>
#include <vector>
#include <array>

#include <eigen3/Eigen/Dense>

#include "frontend/HfTracker.hpp"
#include "CoreTypes.hpp"
#include "simd/Fast.hpp"

namespace PoolSLAM{

size_t HfTracker::replenishBuckets(const uint8_t *curBuffer){
    if(curBuffer == nullptr){
        return 0;
    }
    size_t bucketsOccipied = 0;
    
    for(size_t bucket = 0;  bucket < gridOccupancy.size(); bucket++){
        if(gridOccupancy[bucket].count == 0){
            fastPoints.clear();
            getFast(curBuffer, bucket);
            auto kpCanidate = bestKp(curBuffer);
            if(!kpCanidate.has_value()){
                continue;
            }
            if(addKp(curBuffer, *kpCanidate)){
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

size_t HfTracker::getFast(const uint8_t *curBuffer, size_t bucket){
    auto bucketOffset = offsetFromBucket(bucket);
    Eigen::Vector2i topLeft = topLeftBucket(bucket);
    size_t nPts = 0;
    #ifdef POOL_SLAM_SIMD
    size_t lineWidth = 16;
    #else
    size_t lineWidth = 1
    std::array<uint8_t, 16> ring = {0};
    #endif
    // START VECTORIZE
    for(size_t y = 0; y < hBucket; y++){
        auto rowPtr = curBuffer + bucketOffset + y * wImage;
        for(size_t x = 0; x < wBucket; x += lineWidth){
            auto centerPtr = rowPtr + x;
            #ifdef POOL_SLAM_SIMD
            nPts += simd::fastLine16(centerPtr, topLeft + Eigen::Vector2i(x, y), fastOffsets, FAST_THRESHOLD, fastPoints);
            #else
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

            if(((brighterRun | darkerRun) & 0xFFFF)!= 0){
                fastPoints.push_back(topLeft + Eigen::Vector2i(static_cast<int>(x), static_cast<int>(y)));
                nPts++;
            }
            #endif
        }
    }
    // END VECTORIZE
    
    return nPts;
}

std::optional<Eigen::Vector2i> HfTracker::bestKp(const uint8_t *curBuffer){
    // Initialize the gradients with one extra pixel on each size to make Sobel filter easier
    float bestEigenvalue  = 0;
    Eigen::Vector2i bestPt;
    for(auto kp : fastPoints){
        size_t xPt = static_cast<size_t>(kp.x());
        size_t yPt = static_cast<size_t>(kp.y());

        constexpr size_t PATCH_RADIUS = (ST_PATCH_SIZE - 1) / 2;
        auto topLeft = curBuffer + (yPt - PATCH_RADIUS)  * wImage + xPt - PATCH_RADIUS; // Top left of patch
        int Ixx = 0;
        int Iyy = 0;
        int Ixy = 0;

        // Find structure tensor
        for(Eigen::Index y = 0; y < ST_PATCH_SIZE; y++){
            auto rowPtr = topLeft + static_cast<size_t>(y) * wImage;
            for(Eigen::Index x = 0; x < ST_PATCH_SIZE; x++){
                int Ix = (2 * (*(rowPtr + x - 1) - *(rowPtr + x + 1)) +
                *(rowPtr + wImage + x - 1) + *(rowPtr - wImage + x - 1) - 
                *(rowPtr + wImage + x + 1) - *(rowPtr - wImage + x + 1));
                
                int Iy = (2 * (*(rowPtr - wImage + x) - *(rowPtr + wImage + x)) +
                *(rowPtr - wImage + x - 1) + *(rowPtr - wImage + x + 1) -
                *(rowPtr + wImage + x - 1) - *(rowPtr + wImage + x + 1));

                Ixx += Ix * Ix;
                Iyy += Iy * Iy;
                Ixy += Ix * Iy;
            }
        }

        // Compute Shi-Tomasi score
        float a = static_cast<float>(Ixx);
        float b = static_cast<float>(Ixy);
        float c = static_cast<float>(Iyy);

        float minEigenvalue = 0.5f * ((a + c) - std::sqrt((a - c) * (a - c) + 4 * b * b));

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

bool HfTracker::addKp(const uint8_t *curBuffer, Eigen::Vector2i &kp){
    TrackedFeature<RADIUS> feature;
    constexpr size_t PATCH_SIZE = static_cast<size_t>(TrackedFeature<RADIUS>::PATCH_SIZE);
    feature.id = nextId;
    nextId++;

    feature.ptCurr = kp.cast<float>();
    feature.ptPrev = kp.cast<float>();

    ptrdiff_t xPt = static_cast<ptrdiff_t>(kp.x());
    ptrdiff_t yPt = static_cast<ptrdiff_t>(kp.y());

    // Top left of the grid
    const uint8_t *topLeft = curBuffer + (yPt - RADIUS) * static_cast<ptrdiff_t>(wImage) + (xPt - RADIUS);

    int Tx, Ty;
    int Txx = 0;
    int Tyy = 0;
    int Txy = 0;

    // Find the patch around the keypoint and the gradient within the patch
    for(size_t y = 0; y < PATCH_SIZE; y++){
        const size_t patchRowOffset = PATCH_SIZE * y;
        const uint8_t *rowPtr = topLeft + y * wImage;
        for(size_t x = 0; x < PATCH_SIZE; x++){
            const size_t flatIdx = patchRowOffset + x;
            feature.patch[flatIdx] =*(rowPtr + x);

            Tx = (2 * (*(rowPtr + x - 1) - *(rowPtr + x + 1)) +
                *(rowPtr + wImage + x - 1) + *(rowPtr - wImage + x - 1) - 
                *(rowPtr + wImage + x + 1) - *(rowPtr - wImage + x + 1));
                
            Ty = (2 * (*(rowPtr - wImage + x) - *(rowPtr + wImage + x)) +
                *(rowPtr - wImage + x - 1) + *(rowPtr - wImage + x + 1) -
                *(rowPtr + wImage + x - 1) - *(rowPtr + wImage + x + 1));

            feature.xGrad[flatIdx] = Tx;
            feature.yGrad[flatIdx] = Ty;

            Txx += Tx * Tx;
            Tyy += Ty * Ty;
            Txy += Tx * Ty;
        }
    }

    Eigen::Matrix2f H{
        {static_cast<float>(Txx), static_cast<float>(Txy)},
        {static_cast<float>(Txy), static_cast<float>(Tyy)}
    };

    // Make sure Hessian is not unstable
    if(std::abs(H.determinant()) < HESSIAN_STABILITY_THRESHOLD){
        return false;
    }

    feature.HInv = H.inverse();

    activeFeatures.push_back(feature);

    return true;
}

}