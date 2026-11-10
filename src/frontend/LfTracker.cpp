#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>
#include <utility>
#include <cstddef>
#include <cmath>
#include <array>
#include <algorithm>
#include <optional>

#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/LossFunctions.h>

#include "frontend/LfTracker.hpp"
#include "descriptor/Qov.hpp"
#include "CoreTypes.hpp"
#include "descriptor/BinaryBow.hpp"

namespace PoolSLAM{

bool LfTracker::passKeyframe(uint32_t id, double timestamp, const uint8_t *curBuffer, const std::vector<TrackedFeature<FEATURE_RADIUS>> &features){
    // Check if tracker is busy
    if(busy.load(std::memory_order_relaxed)){
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_passKeyframe);
        for(size_t i = 0; i < config.wImage * config.hImage; i++){
            localBuffer[i] = curBuffer[i];
        }

        Keyframe newFrame(features.size(),  id, timestamp, nextFrameId);

        for(auto &feature : features){
            newFrame.addKp(feature.ptCurr.cast<Eigen::Vector2i>(), feature.id);
        }

        // Resizes features buffer and allocates words buffer
        newFrame.finalSize();

        keyframes.push_back(newFrame);
        nextFrameId++;
    }

    c_passKeyframe.notify_all();

    return true;
}

bool LfTracker::getMatch(BackendLfPacket &outPacket){
    if(!dataReady.load(std::memory_order_acquire)){
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_backendPacket);

        if(!stagedMatch.has_value()){
            dataReady.store(false, std::memory_order_relaxed);
            return false;
        }

        // Transfer staged match to outPacket and reset its state
        outPacket = std::move(*stagedMatch);
        stagedMatch.reset();

        dataReady.store(false, std::memory_order_release);
    }
    return true;
}

void LfTracker::processKeyframe(){
    while(runTracker.load(std::memory_order_relaxed)){
        size_t numMatches = 0;
        BackendLfPacket packet;
        {
            std::unique_lock<std::mutex> lock(m_passKeyframe);

            c_passKeyframe.wait(lock);
            busy.store(true, std::memory_order_acquire);

            // Blur the image
            gaussianBlur();
            
            // Find the QOV descriptors of the image
            calculateDescriptors();

            // Iterate over descriptors and add to database
            auto &kf = keyframes.back();
            float sumOfWeights = 0.0f;
            float freq = 1.0f / static_cast<float>(kf.features.size());
            for(auto &feat : kf.features){
                auto wordId = vocabTree.matchWord(feat.des, freq, kf.keyframeId, database);
                sumOfWeights += database.getEntries(wordId).back().weight;
                
                feat.wordId = wordId;
            }

            std::sort(kf.features.begin(), kf.features.end(), [](const auto &a, const auto& b){
                return a.wordId < b.wordId;
            });


            // Update the score of each frame
            uint32_t lastWordId = UINT32_MAX;
            auto weightNormFactor = 1.0f / sumOfWeights;
            for(auto &feat : kf.features){
                if(lastWordId != feat.wordId){
                    database.invertedIndex[feat.wordId].back().weight *= weightNormFactor;
                }
                auto &frames = database.getEntries(feat.wordId);
                for(uint32_t kfIdx = 0; kfIdx < frames.size(); kfIdx++){
                    keyframes[frames[kfIdx].keyframeId].score += std::min(frames[kfIdx].weight, frames.back().weight);
                }
            }

            // Find the best frame
            Keyframe &bestFrame = keyframes.front();
            float bestScore = 0.0;
            float lastScore = (keyframes.end() - 1)->score;
            for(auto &frame : keyframes){
                if(frame.score > bestScore && (frame.keyframeId + config.excludedFrames) < kf.keyframeId){
                    bestFrame = frame;
                    bestScore = frame.score;
                }
                frame.score = 0;
            }

            packet.currentFrameId = kf.id;
            packet.matchedFrameId = bestFrame.id;
            packet.timestamp = kf.timestamp;

            // Find the ratio between the score of the best frame and the score of the previous frame
            auto scoreRatio = bestScore / lastScore;

            packet.matchScore = bestScore;

            if(bestScore >= config.minTfIdfScore && scoreRatio >= MIN_SCORE_RATIO){
                numMatches = matchFeatures(kf.keyframeId, bestFrame.keyframeId);
            }else{
                busy.store(false, std::memory_order_release);
                continue;
            }
        }

        auto pair = ransac(numMatches);
        if(!pair.has_value()){
            busy.store(false, std::memory_order_release);
            continue;
        }

        packet.inlierCount = static_cast<uint32_t>(pair->second);
        packet.inlierRatio = static_cast<float>(pair->second) / static_cast<float>(numMatches);

        auto T = pair->first;

        // Find optimal pose and measurement covariance
        Eigen::Matrix<double, 6, 6> outCovariance;
        refinePoseLM(pair->second, config.K, T, outCovariance);

        gtsam::Rot3 rot(T.block<3, 3>(0, 0));
        gtsam::Point3 trans(T.block<3, 1>(0, 3));
        
        // Package noise model
        gtsam::Vector6 sigmas;
        sigmas << std::sqrt(outCovariance(3, 3)), // Rot X
            std::sqrt(outCovariance(4, 4)), // Rot Y
            std::sqrt(outCovariance(5, 5)), // Rot Z
            std::sqrt(outCovariance(0, 0)), // Trans X
            std::sqrt(outCovariance(1, 1)), // Trans Y
            std::sqrt(outCovariance(2, 2)); // Trans Z

        auto baseGaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

        {
            std::unique_lock<std::mutex> lock(m_backendPacket);
            stagedMatch = packet;
            stagedMatch->noiseModel = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(1.345), baseGaussian
            );
            stagedMatch->relativePose = gtsam::Pose3(rot, trans);
        }
        busy.store(false, std::memory_order_release);
    }
    
}

void LfTracker::calculateDescriptors(){
    auto &kf = keyframes.back();
    for(auto &feat : kf.features){
        uint16_t m10 = 0;
        uint16_t m01 = 0;
        // Find the orientation of the feature
        // BEGIN VECTORIZE
        for(int y = -QOV_RADIUS; y <= QOV_RADIUS; y++){
            auto rowPtr = localBuffer.get() + (feat.pt.y() + y) + feat.pt.x();
            for(int x = -QOV_RADIUS; x <= QOV_RADIUS; x++){
                m10 += *(rowPtr + x) * x;
                m01 += *(rowPtr + y) * y;
            }
        }
        // END VECTORIZE
        auto angle = std::atan2(m01, m10);
        auto featPtr = localBuffer.get() + config.wImage * static_cast<size_t>(feat.pt.y()) + feat.pt.x();
        auto offsets = qovPairs.offsetAtAngleRad(angle);
        feat.des = descriptor::generateDescriptor(featPtr, offsets);
    }
}

size_t LfTracker::matchFeatures(size_t curId, size_t matchId){
    auto &curKf = keyframes[curId];
    auto &matchKf = keyframes[matchId];

    // Match features
    Eigen::Index  matchingIdx = 0;
    for(size_t curFeatIdx = 0; curFeatIdx < curKf.features.size(); curFeatIdx++){
        int bestDist = INT32_MAX; // Distance of the best match
        int secondDist = INT32_MAX; // Distance of the 
        size_t bestIdx = 0;
        auto &curFeat = curKf.features[curFeatIdx];
        auto curParent = vocabTree.vocabulary[curFeat.wordId].parent;
        bool firstChild = false;
        bool exactMatch = false;
        // Iterate through features of matched frame until the best two features remain
        for(size_t matchFeatIdx = 0; matchFeatIdx < matchKf.features.size(); matchFeatIdx++){
            auto &matchFeat = matchKf.features[matchFeatIdx];
            if(matchFeat.featureId == curFeat.featureId){
                exactMatch = true;
                bestIdx = matchFeatIdx;
                break;
            }
            auto matchParent = vocabTree.vocabulary[matchFeat.wordId].parent;
            // Only look at features whose associated words share a parent
            if(matchParent != curParent){
                // Break if we passed the region where the parent's leaves reside
                if(firstChild){
                    break;
                }else{
                    continue;
                }
            }
            firstChild = true;
            auto hamDist = descriptor::fastHamming(curFeat.des, curFeat.des);
            if(hamDist < bestDist){
                secondDist = bestDist;
                bestDist = hamDist;
                bestIdx = matchFeatIdx;
            }else if(hamDist < secondDist){
                secondDist = hamDist;
            }
        }

        if(!exactMatch){
            // Reject features too similair to other features
            if(!(static_cast<float>(secondDist) / static_cast<float>(bestDist) >= MAX_FEATURE_DIST_RATIO)){
                continue;
            }

            auto &matchedFeat = matchKf.features[bestIdx];

            // Check features in the current keyframe near the current feature to see if they are more similar to the matched feature
            bool bidirectional = true;
            for(size_t i = curFeatIdx; i >= std::max<size_t>(0, BIDIRECTIONAL_SEARCH_INTERVAL); i--){
                auto hamDist = descriptor::fastHamming(matchedFeat.des, curKf.features[i].des);
                if(hamDist < bestDist){
                    bidirectional = false;
                    break;
                }
            }

            if(bidirectional){
                for(size_t i = curFeatIdx; i < std::min<size_t>(curKf.features.size(), BIDIRECTIONAL_SEARCH_INTERVAL); i++){
                    auto hamDist = descriptor::fastHamming(matchedFeat.des, curKf.features[i].des);
                    if(hamDist < bestDist){
                        bidirectional = false;
                        break;
                    }
                }
            }else{
                continue;
            }
        }

        matching.curFeatures.col(matchingIdx) = curFeat.pt.homogeneous().cast<double>();
        matching.matchLandmarks.col(matchingIdx) = matchKf.landmarks[static_cast<size_t>(bestIdx)].pt3d;
        matchingIdx++; 
    }

    return static_cast<size_t>(matchingIdx);
}

std::optional<std::pair<Eigen::Matrix<double, 3, 4>, size_t>> LfTracker::ransac(size_t numMatches){
    // Undistort the features
    undistort(matching.curFeatures);
    
    // Do a number of ransac iterations using P3P
    std::array<Eigen::Index, 4> m; // Match indicies
    std::array<Eigen::Vector3d, 4> x; // 3d points
    std::array<Eigen::Vector3d, 4> y; // homogenous points
    Eigen::Matrix<double, 3, 4> bestT; // Best transformation
    size_t bestInliers; // Best number of inliers
    for(int iter = 0; iter < config.ransacIters; iter++){
        // Find 4 random matches
        m[0] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);
        m[1] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);
        m[2] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);
        m[3] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);

        x[0] = matching.matchLandmarks.col(m[0]).cast<double>();
        x[1] = matching.matchLandmarks.col(m[1]).cast<double>();
        x[2] = matching.matchLandmarks.col(m[2]).cast<double>();
        x[3] = matching.matchLandmarks.col(m[3]).cast<double>();

        y[0] = matching.curFeatures.col(m[0]);
        y[1] = matching.curFeatures.col(m[1]);
        y[2] = matching.curFeatures.col(m[2]);
        y[3] = matching.curFeatures.col(m[3]);
        
        // Solve P3P for the transformation [R | t]
        auto T = geometry::solveP4PLambdaTwist(x, y);
        if(!T.has_value()){
            continue;
        }

        auto R = T->leftCols(3);
        auto t = T->rightCols(1);
        
        size_t inliers = 0;
        for(int match; match < static_cast<int>(numMatches); match++){
            Eigen::Vector3d P = matching.matchLandmarks.col(match);
            Eigen::Vector2d p = matching.curFeatures.col(match).hnormalized();

            Eigen::Vector2d proj = (R * P + t).hnormalized();
            
            auto e = (proj - p).norm();
            if(e <= config.maxReprojectionError){
                inliers++;
            }
        }

        if(inliers > bestInliers){
            bestInliers = inliers;
            bestT = *T;
        }
    }

    auto ratio = (static_cast<float>(bestInliers) / static_cast<float>(numMatches));
    if(bestInliers < config.ransacInliearThreshold || ratio < config.minInliearRatio){
        return std::nullopt;
    }

    // Repackage the inliers
    auto R = bestT.leftCols(3);
    auto t = bestT.rightCols(1);
    int writeIdx = 0;
    for(int readIdx = 0; static_cast<size_t>(readIdx) < numMatches; readIdx++){
        Eigen::Vector3d P = matching.matchLandmarks.col(readIdx);
        Eigen::Vector2d p = matching.curFeatures.col(readIdx).hnormalized();

        Eigen::Vector2d proj = (R * P + t).hnormalized();
        
        auto e = (proj - p).norm();
        if(e <= config.maxReprojectionError){
            matching.curFeatures.col(writeIdx) = matching.curFeatures.col(readIdx);
            matching.matchLandmarks.col(writeIdx) = matching.curFeatures.col(readIdx);
        }
    }

    return std::pair{bestT, ratio};


}

void LfTracker::undistort(Eigen::Matrix3Xd &pts){    
    auto count = pts.size();
    float cx = config.K(0, 2);
    float cy = config.K(1, 2);
    float fxInv = 1.0f / config.K(0, 0);
    float fyInv = 1.0f / config.K(1, 1);
    // START VECTORIZE
    for(int i = 0; i < count; i++){
        auto pt = pts.col(i).cast<float>();
        float xd = (pt.x() - cx) * fxInv;
        float yd = (pt.y() - cy) * fyInv;
        float x = xd;
        float y = yd;

        for(int iter = 0; iter < UNDISTORT_ITERS; iter++){
            float r2 = x * x + y * y;
            float r4 = r2 * r2;
            float xy = x * y;
            float radial = 1.0f + config.dist.k1 * r2 + config.dist.k2 * r4;

            float xEst = x * radial + 2.0f * config.dist.p1 * xy + config.dist.p2 * (r2 + 2 * x * x);
            float yEst = y * radial + 2.0f * config.dist.p2 * xy + config.dist.p1 * (r2 + 2 * y * y);

            x -= (xEst - xd);
            y -= (yEst - yd);
        }

        pts(0, i) = x;
        pts(1, i) = y;
        pts(2, i) = 1.0f;
    }
    // END VECTORIZE
}


void LfTracker::gaussianBlur(){
    // First pass using [1, 4, 6, 4, 1]
    for(size_t row = 0; row < config.hImage; row++){
        auto rowPtr = localBuffer.get() + row * config.wImage;
        blurData.rowBuffer[0] = static_cast<uint8_t>(
            *(rowPtr) * BlurData::KERNEL[0] + *(rowPtr) * BlurData::KERNEL[1] + 
            *(rowPtr) * BlurData::KERNEL[2] + *(rowPtr + 1) * BlurData::KERNEL[3] +
            *(rowPtr + 2) * BlurData::KERNEL[4]
        );
        blurData.rowBuffer[1] = static_cast<uint8_t>(
            *(rowPtr) * BlurData::KERNEL[0] + *(rowPtr) * BlurData::KERNEL[1] + 
            *(rowPtr + 1) * BlurData::KERNEL[2] + *(rowPtr + 2) * BlurData::KERNEL[3] +
            *(rowPtr + 3) * BlurData::KERNEL[4]
        );
        for(size_t col = 2; col < config.wImage - 2; col++){
            blurData.rowBuffer[col] = static_cast<uint8_t>(
                *(rowPtr + col - 2) * BlurData::KERNEL[0] + *(rowPtr + col - 1) * BlurData::KERNEL[1] + 
                *(rowPtr + col) * BlurData::KERNEL[2] + *(rowPtr + col + 1) * BlurData::KERNEL[3] +
                *(rowPtr + col + 2) * BlurData::KERNEL[4]
            );
        }
        const auto lastIndex = config.wImage - 1;
        blurData.rowBuffer[config.wImage - 2] = static_cast<uint8_t>(
            *(rowPtr + lastIndex - 3) * BlurData::KERNEL[0] + *(rowPtr + lastIndex - 2) * BlurData::KERNEL[1] + 
            *(rowPtr + lastIndex - 1) * BlurData::KERNEL[2] + *(rowPtr + lastIndex) * BlurData::KERNEL[3] +
            *(rowPtr + lastIndex) * BlurData::KERNEL[4]
        );
        blurData.rowBuffer[config.wImage - 2] = static_cast<uint8_t>(
            *(rowPtr + lastIndex - 2) * BlurData::KERNEL[0] + *(rowPtr + lastIndex - 1) * BlurData::KERNEL[1] + 
            *(rowPtr + lastIndex) * BlurData::KERNEL[2] + *(rowPtr + lastIndex) * BlurData::KERNEL[3] +
            *(rowPtr + lastIndex) * BlurData::KERNEL[4]
        );

        std::copy(blurData.rowBuffer.get(), blurData.rowBuffer.get() + lastIndex, rowPtr);
    }

    // Second pass using [1, 4, 6, 4, 1]T

    // First two rows because they are cut off
    auto firstRowPtr = localBuffer.get();
    auto secondRowPtr = firstRowPtr + config.wImage;
    for(size_t col = 0; col < config.wImage; col++){
        blurData.ringBuffer[0][col] = static_cast<uint8_t>(
            *(firstRowPtr) * BlurData::KERNEL[0] + *(firstRowPtr) * BlurData::KERNEL[1] + 
            *(firstRowPtr) * BlurData::KERNEL[2] + *(firstRowPtr + config.wImage) * BlurData::KERNEL[3] +
            *(firstRowPtr + 2 * config.wImage) * BlurData::KERNEL[4]
        );
        blurData.ringBuffer[1][col] = static_cast<uint8_t>(
            *(firstRowPtr) * BlurData::KERNEL[0] + *(firstRowPtr) * BlurData::KERNEL[1] + 
            *(secondRowPtr) * BlurData::KERNEL[2] + *(secondRowPtr + config.wImage) * BlurData::KERNEL[3] +
            *(secondRowPtr + 2 * config.wImage) * BlurData::KERNEL[4]
        );
    }

    // Main rows
    size_t ringWIdx = 2;
    size_t ringRIdx = 0;
    for(size_t row = 2; row < config.hImage - 2; row++){
        auto rowPtr = localBuffer.get() + row * config.wImage;
        for(size_t col = 0; col < config.wImage; col++){
            blurData.ringBuffer[ringWIdx][col] = static_cast<uint8_t>(
                *(rowPtr - 2 * config.wImage) * BlurData::KERNEL[0] + *(rowPtr - config.wImage) * BlurData::KERNEL[1] + 
                *(rowPtr) * BlurData::KERNEL[2] + *(rowPtr + config.wImage) * BlurData::KERNEL[3] +
                *(rowPtr + 2 * config.wImage) * BlurData::KERNEL[4]
            );
        }
        std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.wImage - 1, rowPtr -  2 * config.wImage);
        ringWIdx++;
        ringWIdx %= 3;
        ringRIdx++;
        ringRIdx %= 3;
    }


    // Last two rows because they are cut off
    auto penRowPtr = localBuffer.get() + config.wImage * (config.hImage - 2);
    for(size_t col = 0; col < config.wImage; col++){
        blurData.ringBuffer[ringWIdx][col] = static_cast<uint8_t>(
            *(penRowPtr - 2 * config.wImage) * BlurData::KERNEL[0] + *(penRowPtr - config.wImage) * BlurData::KERNEL[1] + 
            *(penRowPtr) * BlurData::KERNEL[2] + *(penRowPtr + config.wImage) * BlurData::KERNEL[3] +
            *(penRowPtr + config.wImage) * BlurData::KERNEL[4]
        );
    }

    std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.wImage - 1, penRowPtr -  2 * config.wImage);
    ringWIdx++;
    ringWIdx %= 3;
    ringRIdx++;
    ringRIdx %= 3;

    auto lastRowPtr = penRowPtr + config.wImage;
    for(size_t col = 0; col < config.wImage; col++){
        *(lastRowPtr) = static_cast<uint8_t>(
            *(lastRowPtr - 2 * config.wImage) * BlurData::KERNEL[0] + *(lastRowPtr - config.wImage) * BlurData::KERNEL[1] + 
            *(lastRowPtr) * BlurData::KERNEL[2] + *(lastRowPtr) * BlurData::KERNEL[3] +
            *(lastRowPtr) * BlurData::KERNEL[4]
        );
    }

    std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.wImage - 1, penRowPtr -  2 * config.wImage);
    ringWIdx++;
    ringWIdx %= 3;
    ringRIdx++;
    ringRIdx %= 3;
    std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.wImage - 1, penRowPtr -  2 * config.wImage);
}

};