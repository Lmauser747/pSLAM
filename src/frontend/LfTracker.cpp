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
#include <thread>
#include <numeric>

#include <eigen3/Eigen/Dense>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/LossFunctions.h>

#include "frontend/LfTracker.hpp"
#include "CoreTypes.hpp"
#include "descriptor/Qov.hpp"

namespace PoolSLAM{

void LfTracker::start(){
    if(runTracker.load(std::memory_order_relaxed)){
        return;
    }
    runTracker.store(true, std::memory_order_release);
    workerThread = std::thread(&LfTracker::processKeyframe, this);
}

void LfTracker::end(){
    if(!runTracker.load(std::memory_order_relaxed)){
        return;
    }
    runTracker.store(false, std::memory_order_release);
    c_passKeyframe.notify_all();
    if(workerThread.joinable()){
        workerThread.join();
    }
}

bool LfTracker::isRunning() const{
    return runTracker.load(std::memory_order_relaxed);
}

bool LfTracker::passKeyframe(uint32_t id, double timestamp, const uint8_t *curBuffer, const std::vector<TrackedFeature<FEATURE_RADIUS>> &features){
    // Check if tracker is busy or features are empty
    if(curBuffer == nullptr || busy.load(std::memory_order_acquire) || features.empty()){
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_passKeyframe);
        for(size_t i = 0; i < config.w * config.h; i++){
            localBuffer[i] = curBuffer[i];
        }

        Keyframe newFrame(features.size(), id, timestamp, nextFrameId);

        for(auto &feature : features){
            Eigen::Vector2i kp = feature.ptCurr.cast<int>();
            if(kp.x() >= QOV_RADIUS && kp.x() < static_cast<int>(config.w) - QOV_RADIUS &&
               kp.y() >= QOV_RADIUS && kp.y() < static_cast<int>(config.h) - QOV_RADIUS){
                newFrame.addKp(feature.ptCurr, feature.id);
            }
        }

        if(newFrame.features.empty()){
            return false;
        }

        // Resizes features buffer and allocates words buffer
        newFrame.finalSize();

        keyframes.push_back(std::move(newFrame));
        nextFrameId++;
        busy.store(true, std::memory_order_release);
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

bool LfTracker::updateKeyframe(BackendKeyframeUpdate &update){
    std::lock_guard<std::mutex> lock(m_passKeyframe);
    for(auto &kf : keyframes){
        if(kf.id == update.id){
            kf.pose = update.pose;
            kf.landmarks.assign(kf.features.size(), Landmark3d{});
            for(const auto &lm : update.landmarks){
                for(size_t i = 0; i < kf.features.size(); i++){
                    if(kf.features[i].featureId == lm.featureId){
                        kf.landmarks[i] = lm;
                        break;
                    }
                }
            }
            return true;
        }
    }
    return false;
}

void LfTracker::processKeyframe(){
    while(runTracker.load(std::memory_order_relaxed)){
        size_t numMatches = 0;
        BackendLfPacket packet;
        {
            std::unique_lock<std::mutex> lock(m_passKeyframe);

            c_passKeyframe.wait(lock, [this]{
                return busy.load(std::memory_order_acquire) || !runTracker.load(std::memory_order_relaxed);
            });

            if(!runTracker.load(std::memory_order_relaxed)){
                break;
            }

            lock.unlock();

            // Blur the image
            gaussianBlur();
            
            // Find the QOV descriptors of the image
            calculateDescriptors();

            lock.lock();

            // Iterate over descriptors and add to database
            auto &kf = keyframes.back();
            float freq = 1.0f / static_cast<float>(kf.features.size());
            for(auto &feat : kf.features){
                feat.wordId = vocabTree.matchWord(feat.des, freq, kf.keyframeId, database);
            }

            std::vector<size_t> perm(kf.features.size());
            std::iota(perm.begin(), perm.end(), 0);
            std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b){
                return kf.features[a].wordId < kf.features[b].wordId;
            });

            std::vector<descriptor::QovFeature> sortedFeatures;
            sortedFeatures.reserve(kf.features.size());
            for(size_t idx : perm){
                sortedFeatures.push_back(std::move(kf.features[idx]));
            }
            kf.features = std::move(sortedFeatures);

            if(kf.landmarks.size() == perm.size()){
                std::vector<Landmark3d> sortedLandmarks;
                sortedLandmarks.reserve(kf.landmarks.size());
                for(size_t idx : perm){
                    sortedLandmarks.push_back(std::move(kf.landmarks[idx]));
                }
                kf.landmarks = std::move(sortedLandmarks);
            }

            // Compute L1 norm across unique words in this keyframe
            float sumOfWeights = 0.0f;
            uint32_t lastWordId = UINT32_MAX;
            for(auto &feat : kf.features){
                if(lastWordId != feat.wordId){
                    lastWordId = feat.wordId;
                    if(feat.wordId < database.invertedIndex.size() && !database.getEntries(feat.wordId).empty()){
                        sumOfWeights += database.getEntries(feat.wordId).back().weight;
                    }
                }
            }

            // Normalize weights and update the score of each frame once per unique word
            std::vector<uint32_t> scoredKfIndices;
            scoredKfIndices.reserve(256);

            lastWordId = UINT32_MAX;
            auto weightNormFactor = (sumOfWeights > 0.0f) ? (1.0f / sumOfWeights) : 0.0f;
            for(auto &feat : kf.features){
                if(lastWordId != feat.wordId){
                    lastWordId = feat.wordId;
                    if(feat.wordId < database.invertedIndex.size() && !database.invertedIndex[feat.wordId].empty()){
                        database.invertedIndex[feat.wordId].back().weight *= weightNormFactor;
                        auto &frames = database.getEntries(feat.wordId);
                        for(uint32_t kfIdx = 0; kfIdx < frames.size(); kfIdx++){
                            uint32_t targetKfId = frames[kfIdx].keyframeId;
                            if(targetKfId < keyframes.size() && keyframes[targetKfId].score == 0.0f){
                                scoredKfIndices.push_back(targetKfId);
                            }
                            if(targetKfId < keyframes.size()){
                                keyframes[targetKfId].score += std::min(frames[kfIdx].weight, frames.back().weight);
                            }
                        }
                    }
                }
            }

            // Find the best frame
            Keyframe *bestFrame = nullptr;
            float bestScore = 0.0f;
            float lastScore = (keyframes.size() >= 2) ? keyframes[keyframes.size() - 2].score : 1.0f;
            // Search best frame only over keyframes with non-zero scores
            for(uint32_t kfId : scoredKfIndices){
                auto &frame = keyframes[kfId];
                if(frame.score > bestScore && (frame.keyframeId + config.excludedFrames) < kf.keyframeId && !frame.landmarks.empty()){
                    bestFrame = &frame;
                    bestScore = frame.score;
                }
                frame.score = 0.0f;
            }

            if(bestFrame == nullptr){
                busy.store(false, std::memory_order_release);
                continue;
            }

            packet.currentFrameId = kf.id;
            packet.matchedFrameId = bestFrame->id;
            packet.timestamp = kf.timestamp;

            // Find the ratio between the score of the best frame and the score of the previous frame
            auto refScore = (lastScore > 1e-6f) ? lastScore : 1.0f;
            auto scoreRatio = bestScore / refScore;

            packet.matchScore = bestScore;

            if(bestScore >= config.minTfIdfScore && scoreRatio >= MIN_SCORE_RATIO){
                numMatches = matchFeatures(kf.keyframeId, bestFrame->keyframeId);
            }else{
                busy.store(false, std::memory_order_release);
                continue;
            }
        }

        if(numMatches < std::max<size_t>(4, config.ransacInliearThreshold)){
            busy.store(false, std::memory_order_release);
            continue;
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
        if(!refinePoseLM(pair->second, config.K.cast<double>(), T, outCovariance)){
            busy.store(false, std::memory_order_release);
            continue;
        }

        if(!outCovariance.allFinite()){
            busy.store(false, std::memory_order_release);
            continue;
        }

        gtsam::Rot3 rot(T.block<3, 3>(0, 0));
        gtsam::Point3 trans(T.block<3, 1>(0, 3));
        
        // Package noise model
        gtsam::Vector6 sigmas;
        sigmas << std::sqrt(std::max(1e-12, outCovariance(3, 3))), // Rot X
            std::sqrt(std::max(1e-12, outCovariance(4, 4))), // Rot Y
            std::sqrt(std::max(1e-12, outCovariance(5, 5))), // Rot Z
            std::sqrt(std::max(1e-12, outCovariance(0, 0))), // Trans X
            std::sqrt(std::max(1e-12, outCovariance(1, 1))), // Trans Y
            std::sqrt(std::max(1e-12, outCovariance(2, 2))); // Trans Z

        if(!sigmas.allFinite()){
            busy.store(false, std::memory_order_release);
            continue;
        }

        auto baseGaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

        {
            std::unique_lock<std::mutex> lock(m_backendPacket);
            stagedMatch = packet;
            stagedMatch->noiseModel = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(1.345), baseGaussian
            );
            stagedMatch->relativePose = gtsam::Pose3(rot, trans);
            dataReady.store(true, std::memory_order_release);
        }
        busy.store(false, std::memory_order_release);
    }
    
}

void LfTracker::calculateDescriptors(){
    auto &kf = keyframes.back();
    for(auto &feat : kf.features){
        int32_t m10 = 0;
        int32_t m01 = 0;
        // Find the orientation of the feature
        // BEGIN VECTORIZE
        for(int y = -QOV_RADIUS; y <= QOV_RADIUS; y++){
            auto rowPtr = localBuffer.get() + static_cast<size_t>(feat.pt.y() + y) * config.w + feat.pt.x();
            for(int x = -QOV_RADIUS; x <= QOV_RADIUS; x++){
                m10 += *(rowPtr + x) * x;
                m01 += *(rowPtr + x) * y;
            }
        }
        // END VECTORIZE
        auto angle = std::atan2(static_cast<double>(m01), static_cast<double>(m10));
        auto featPtr = localBuffer.get() + config.w * static_cast<size_t>(feat.pt.y()) + feat.pt.x();
        const auto &offsets = qovPairs.offsetAtAngleRad(angle);
        feat.des = descriptor::generateDescriptor(featPtr, offsets);
    }
}

size_t LfTracker::matchFeatures(size_t curId, size_t matchId){
    auto &curKf = keyframes[curId];
    auto &matchKf = keyframes[matchId];

    if(curKf.features.empty() || matchKf.landmarks.empty() || matchKf.landmarks.size() < matchKf.features.size()){
        return 0;
    }

    if(matching.curFeatures.cols() < static_cast<Eigen::Index>(curKf.features.size())){
        matching.curFeatures.resize(3, static_cast<int>(curKf.features.size()));
        matching.matchLandmarks.resize(3, static_cast<int>(curKf.features.size()));
    }

    const size_t curSize = curKf.features.size();
    const size_t matchSize = matchKf.features.size();

    std::vector<int> bestDistForCur(curSize, INT32_MAX);
    std::vector<int> secondDistForCur(curSize, INT32_MAX);
    std::vector<size_t> bestMatchForCur(curSize, 0);
    std::vector<uint8_t> isExactForCur(curSize, 0);

    std::vector<int> bestDistForMatch(matchSize, INT32_MAX);
    std::vector<size_t> bestCurForMatch(matchSize, 0);
    std::vector<uint8_t> matchLockedByExact(matchSize, 0);

    // Match across all valid landmarks so tree quantization boundaries never drop descriptors
    for(size_t curFeatIdx = 0; curFeatIdx < curSize; curFeatIdx++){
        const auto &curFeat = curKf.features[curFeatIdx];
        for(size_t matchFeatIdx = 0; matchFeatIdx < matchSize; matchFeatIdx++){
            const auto &matchFeat = matchKf.features[matchFeatIdx];
            if(matchKf.landmarks[matchFeatIdx].featureId != matchFeat.featureId ||
               matchKf.landmarks[matchFeatIdx].pt3d.z() <= 0.0){
                continue;
            }
            if(matchFeat.featureId == curFeat.featureId){
                isExactForCur[curFeatIdx] = 1;
                bestMatchForCur[curFeatIdx] = matchFeatIdx;
                matchLockedByExact[matchFeatIdx] = 1;
                break;
            }
            const int hamDist = descriptor::fastHamming(curFeat.des, matchFeat.des);
            if(hamDist < bestDistForCur[curFeatIdx]){
                secondDistForCur[curFeatIdx] = bestDistForCur[curFeatIdx];
                bestDistForCur[curFeatIdx] = hamDist;
                bestMatchForCur[curFeatIdx] = matchFeatIdx;
            }else if(hamDist < secondDistForCur[curFeatIdx]){
                secondDistForCur[curFeatIdx] = hamDist;
            }

            if(hamDist < bestDistForMatch[matchFeatIdx]){
                bestDistForMatch[matchFeatIdx] = hamDist;
                bestCurForMatch[matchFeatIdx] = curFeatIdx;
            }
        }
    }

    Eigen::Index matchingIdx = 0;
    for(size_t curFeatIdx = 0; curFeatIdx < curSize; curFeatIdx++){
        const size_t bestIdx = bestMatchForCur[curFeatIdx];
        if(!isExactForCur[curFeatIdx]){
            const int bestDist = bestDistForCur[curFeatIdx];
            const int secondDist = secondDistForCur[curFeatIdx];
            if(bestDist == INT32_MAX || bestDist > MAX_HAMMING_DIST || secondDist == 0){
                continue;
            }
            if(secondDist < INT32_MAX &&
               static_cast<float>(bestDist) / static_cast<float>(secondDist) > MAX_FEATURE_DIST_RATIO){
                continue;
            }
            if(matchLockedByExact[bestIdx] || bestCurForMatch[bestIdx] != curFeatIdx){
                continue;
            }
        }

        if(matchingIdx < matching.curFeatures.cols() && bestIdx < matchKf.landmarks.size()){
            matching.curFeatures.col(matchingIdx) = curKf.features[curFeatIdx].ptSubpix.homogeneous().cast<double>();
            matching.matchLandmarks.col(matchingIdx) = matchKf.landmarks[bestIdx].pt3d;
            matchingIdx++;
        }
    }

    return static_cast<size_t>(matchingIdx);
}

std::optional<std::pair<Eigen::Matrix<double, 3, 4>, size_t>> LfTracker::ransac(size_t numMatches){
    if(numMatches < 4){
        return std::nullopt;
    }

    // Undistort the features
    undistort(matching.curFeatures, numMatches);
    
    const double fx = static_cast<double>(config.K(0, 0));
    const double fy = static_cast<double>(config.K(1, 1));

    // Do a number of ransac iterations using P3P
    std::array<Eigen::Index, 4> m; // Match indicies
    std::array<Eigen::Vector3d, 4> x; // 3d points
    std::array<Eigen::Vector3d, 4> y; // homogenous points
    Eigen::Matrix<double, 3, 4> bestT = Eigen::Matrix<double, 3, 4>::Zero(); // Best transformation
    size_t bestInliers = 0; // Best number of inliers
    const size_t earlyExitInliers = std::max<size_t>(
        config.ransacInliearThreshold,
        static_cast<size_t>(0.85f * static_cast<float>(numMatches))
    );
    for(int iter = 0; iter < config.ransacIters; iter++){
        // Find 4 distinct random matches
        for(size_t k = 0; k < 4; k++){
            bool collision;
            do{
                collision = false;
                m[k] = static_cast<Eigen::Index>((static_cast<uint64_t>(ransacRng()) * numMatches) >> 31);
                for(size_t prev = 0; prev < k; prev++){
                    if(m[k] == m[prev]){
                        collision = true;
                        break;
                    }
                }
            }while(collision);
        }

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
        for(size_t match = 0; match < numMatches; match++){
            Eigen::Vector3d P = matching.matchLandmarks.col(static_cast<int>(match));
            Eigen::Vector3d PCam = R * P + t;
            if(PCam.z() <= 1e-6){
                continue;
            }
            Eigen::Vector2d p = matching.curFeatures.col(static_cast<int>(match)).head<2>();
            Eigen::Vector2d proj = PCam.hnormalized();
            Eigen::Vector2d diff(fx * (proj.x() - p.x()), fy * (proj.y() - p.y()));
            
            auto e = diff.squaredNorm();
            if(e <= maxReprojectionErrorSq){
                inliers++;
            }
        }

        if(inliers > bestInliers){
            bestInliers = inliers;
            bestT = *T;
            if(bestInliers >= earlyExitInliers &&
               (static_cast<float>(bestInliers) / static_cast<float>(numMatches)) >= config.minInliearRatio){
                break;
            }
        }
    }

    auto ratio = (static_cast<float>(bestInliers) / static_cast<float>(numMatches));
    if(bestInliers < config.ransacInliearThreshold || ratio < config.minInliearRatio){
        return std::nullopt;
    }

    // Repackage the inliers
    auto R = bestT.leftCols(3);
    auto t = bestT.rightCols(1);
    Eigen::Index writeIdx = 0;
    for(Eigen::Index readIdx = 0; static_cast<size_t>(readIdx) < numMatches; readIdx++){
        Eigen::Vector3d P = matching.matchLandmarks.col(readIdx);
        Eigen::Vector3d PCam = R * P + t;
        if(PCam.z() <= 1e-6){
            continue;
        }
        Eigen::Vector2d p = matching.curFeatures.col(readIdx).head<2>();
        Eigen::Vector2d proj = PCam.hnormalized();
        Eigen::Vector2d diff(fx * (proj.x() - p.x()), fy * (proj.y() - p.y()));
        
        auto e = diff.squaredNorm();
        if(e <= maxReprojectionErrorSq){
            matching.curFeatures.col(writeIdx) = matching.curFeatures.col(readIdx);
            matching.matchLandmarks.col(writeIdx) = matching.matchLandmarks.col(readIdx);
            writeIdx++;
        }
    }

    return std::pair{bestT, static_cast<size_t>(writeIdx)};
}



void LfTracker::undistort(Eigen::Matrix3Xd &pts, size_t count){    
    if(count == 0 || count > static_cast<size_t>(pts.cols())){
        count = static_cast<size_t>(pts.cols());
    }
    float cx = config.K(0, 2);
    float cy = config.K(1, 2);
    float fxInv = 1.0f / config.K(0, 0);
    float fyInv = 1.0f / config.K(1, 1);
    // START VECTORIZE
    for(size_t i = 0; i < count; i++){
        auto pt = pts.col(static_cast<int>(i)).cast<float>();
        if(!std::isfinite(pt.x()) || !std::isfinite(pt.y())){
            continue;
        }
        float xd = (pt.x() - cx) * fxInv;
        float yd = (pt.y() - cy) * fyInv;
        float x = xd;
        float y = yd;

        for(int iter = 0; iter < UNDISTORT_ITERS; iter++){
            float r2 = x * x + y * y;
            float r4 = r2 * r2;
            float xy = x * y;
            float radial = 1.0f + config.dist.k1 * r2 + config.dist.k2 * r4;

            float dxTangential = 2.0f * config.dist.p1 * xy + config.dist.p2 * (r2 + 2.0f * x * x);
            float dyTangential = 2.0f * config.dist.p2 * xy + config.dist.p1 * (r2 + 2.0f * y * y);

            if(std::abs(radial) > 1e-6f){
                x = (xd - dxTangential) / radial;
                y = (yd - dyTangential) / radial;
            }
        }

        if(std::isfinite(x) && std::isfinite(y)){
            pts(0, static_cast<int>(i)) = static_cast<double>(x);
            pts(1, static_cast<int>(i)) = static_cast<double>(y);
            pts(2, static_cast<int>(i)) = 1.0;
        }
    }
    // END VECTORIZE
}


void LfTracker::gaussianBlur(){
    // First pass using [1, 4, 6, 4, 1]
    for(size_t row = 0; row < config.h; row++){
        auto rowPtr = localBuffer.get() + row * config.w;
        blurData.rowBuffer[0] = static_cast<uint8_t>((
            *(rowPtr) * BlurData::KERNEL[0] + *(rowPtr) * BlurData::KERNEL[1] + 
            *(rowPtr) * BlurData::KERNEL[2] + *(rowPtr + 1) * BlurData::KERNEL[3] +
            *(rowPtr + 2) * BlurData::KERNEL[4]
        ) >> 4);
        blurData.rowBuffer[1] = static_cast<uint8_t>((
            *(rowPtr) * BlurData::KERNEL[0] + *(rowPtr) * BlurData::KERNEL[1] + 
            *(rowPtr + 1) * BlurData::KERNEL[2] + *(rowPtr + 2) * BlurData::KERNEL[3] +
            *(rowPtr + 3) * BlurData::KERNEL[4]
        ) >> 4);
        for(size_t col = 2; col < config.w - 2; col++){
            blurData.rowBuffer[col] = static_cast<uint8_t>((
                *(rowPtr + col - 2) * BlurData::KERNEL[0] + *(rowPtr + col - 1) * BlurData::KERNEL[1] + 
                *(rowPtr + col) * BlurData::KERNEL[2] + *(rowPtr + col + 1) * BlurData::KERNEL[3] +
                *(rowPtr + col + 2) * BlurData::KERNEL[4]
            ) >> 4);
        }
        const auto lastIndex = config.w - 1;
        blurData.rowBuffer[config.w - 2] = static_cast<uint8_t>((
            *(rowPtr + lastIndex - 3) * BlurData::KERNEL[0] + *(rowPtr + lastIndex - 2) * BlurData::KERNEL[1] + 
            *(rowPtr + lastIndex - 1) * BlurData::KERNEL[2] + *(rowPtr + lastIndex) * BlurData::KERNEL[3] +
            *(rowPtr + lastIndex) * BlurData::KERNEL[4]
        ) >> 4);
        blurData.rowBuffer[lastIndex] = static_cast<uint8_t>((
            *(rowPtr + lastIndex - 2) * BlurData::KERNEL[0] + *(rowPtr + lastIndex - 1) * BlurData::KERNEL[1] + 
            *(rowPtr + lastIndex) * BlurData::KERNEL[2] + *(rowPtr + lastIndex) * BlurData::KERNEL[3] +
            *(rowPtr + lastIndex) * BlurData::KERNEL[4]
        ) >> 4);

        std::copy(blurData.rowBuffer.get(), blurData.rowBuffer.get() + config.w, rowPtr);
    }

    // Second pass using [1, 4, 6, 4, 1]T

    // First two rows because they are cut off
    auto firstRowPtr = localBuffer.get();
    auto secondRowPtr = firstRowPtr + config.w;
    for(size_t col = 0; col < config.w; col++){
        auto fCol = firstRowPtr + col;
        auto sCol = secondRowPtr + col;
        blurData.ringBuffer[0][col] = static_cast<uint8_t>((
            *(fCol) * BlurData::KERNEL[0] + *(fCol) * BlurData::KERNEL[1] + 
            *(fCol) * BlurData::KERNEL[2] + *(fCol + config.w) * BlurData::KERNEL[3] +
            *(fCol + 2 * config.w) * BlurData::KERNEL[4]
        ) >> 4);
        blurData.ringBuffer[1][col] = static_cast<uint8_t>((
            *(fCol) * BlurData::KERNEL[0] + *(fCol) * BlurData::KERNEL[1] + 
            *(sCol) * BlurData::KERNEL[2] + *(sCol + config.w) * BlurData::KERNEL[3] +
            *(sCol + 2 * config.w) * BlurData::KERNEL[4]
        ) >> 4);
    }

    // Main rows
    size_t ringWIdx = 2;
    size_t ringRIdx = 0;
    for(size_t row = 2; row < config.h - 2; row++){
        auto rowPtr = localBuffer.get() + row * config.w;
        for(size_t col = 0; col < config.w; col++){
            auto colPtr = rowPtr + col;
            blurData.ringBuffer[ringWIdx][col] = static_cast<uint8_t>((
                *(colPtr - 2 * config.w) * BlurData::KERNEL[0] + *(colPtr - config.w) * BlurData::KERNEL[1] + 
                *(colPtr) * BlurData::KERNEL[2] + *(colPtr + config.w) * BlurData::KERNEL[3] +
                *(colPtr + 2 * config.w) * BlurData::KERNEL[4]
            ) >> 4);
        }
        std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.w, rowPtr - 2 * config.w);
        ringWIdx++;
        ringWIdx %= 3;
        ringRIdx++;
        ringRIdx %= 3;
    }

    // Last two rows because they are cut off
    auto penRowPtr = localBuffer.get() + config.w * (config.h - 2);
    for(size_t col = 0; col < config.w; col++){
        auto colPtr = penRowPtr + col;
        blurData.ringBuffer[ringWIdx][col] = static_cast<uint8_t>((
            *(colPtr - 2 * config.w) * BlurData::KERNEL[0] + *(colPtr - config.w) * BlurData::KERNEL[1] + 
            *(colPtr) * BlurData::KERNEL[2] + *(colPtr + config.w) * BlurData::KERNEL[3] +
            *(colPtr + config.w) * BlurData::KERNEL[4]
        ) >> 4);
    }

    std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.w, penRowPtr - 2 * config.w);
    ringWIdx++;
    ringWIdx %= 3;
    ringRIdx++;
    ringRIdx %= 3;

    auto lastRowPtr = penRowPtr + config.w;
    for(size_t col = 0; col < config.w; col++){
        auto colPtr = lastRowPtr + col;
        *(colPtr) = static_cast<uint8_t>((
            *(colPtr - 2 * config.w) * BlurData::KERNEL[0] + *(colPtr - config.w) * BlurData::KERNEL[1] + 
            *(colPtr) * BlurData::KERNEL[2] + *(colPtr) * BlurData::KERNEL[3] +
            *(colPtr) * BlurData::KERNEL[4]
        ) >> 4);
    }

    std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.w, penRowPtr - config.w);
    ringWIdx++;
    ringWIdx %= 3;
    ringRIdx++;
    ringRIdx %= 3;
    std::copy(blurData.ringBuffer[ringRIdx].get(), blurData.ringBuffer[ringRIdx].get() + config.w, penRowPtr);
}

};