#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>
#include <utility>
#include <cstddef>
#include <cmath>
#include <array>
#include <algorithm>

#include "frontend/LfTracker.hpp"
#include "descriptor/Qov.hpp"
#include "CoreTypes.hpp"
#include "descriptor/BinaryBow.hpp"

namespace PoolSLAM{

namespace{
    struct DiagonalDecomp{
        double sigma1 = 0.0;
        double sigma2 = 0.0;
        Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
    };

    constexpr double K_EPS = 1e-12;
    constexpr double PI = 3.14159265358979323846;
    FORCE_INLINE double cubicSolver(const std::array<double, 4> &c);
    FORCE_INLINE DiagonalDecomp diagonalize(const Eigen::Matrix3d &D);
}

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

            // Find the ratio between the score of the best frame and the score of the previous frame
            auto scoreRatio = bestScore / lastScore;

            if(bestScore >= config.minTfIdfScore && scoreRatio >= MIN_SCORE_RATIO){
                matchFeatures(kf.keyframeId, bestFrame.keyframeId);
            }else{
                // reject frame
            }

  
        }
        

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

        matching.curFeatures.col(matchingIdx) = curFeat.pt.homogeneous().cast<float>();
        matching.matchLandmarks.col(matchingIdx) = matchKf.landmarks[static_cast<size_t>(matchingIdx)].pt3d;
        matchingIdx++; 
    }

    return static_cast<size_t>(matchingIdx);
}

size_t LfTracker::ransac(size_t numMatches){
    // Undistort the features
    undistort(matching.curFeatures);
    
    // Do a number of ransac iterations using P3P
    std::array<Eigen::Index, 4> m; // Match indicies
    std::array<Eigen::Vector3f, 4> x; // 3d points
    std::array<Eigen::Vector3f, 4> y; // homogenous points
    Eigen::Matrix3d M01 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d M02 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d M12 = Eigen::Matrix3d::Identity();
    M01(2, 2) = 0;
    M02(1, 1) = 0;
    M12(0, 0) = 0;
    Eigen::Matrix3d D0, D1, D2;
    std::array<double, 4> c;
    std::array<double, 4> tau;
    for(int iter = 0; iter < config.ransacIters; iter++){
        // Find 4 random matches
        m[0] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);
        m[1] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);
        m[2] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);
        m[3] = static_cast<Eigen::Index>((ransacRng() * numMatches) >> 31);

        x[0] = matching.matchLandmarks.col(m[0]);
        x[1] = matching.matchLandmarks.col(m[1]);
        x[2] = matching.matchLandmarks.col(m[2]);
        x[3] = matching.matchLandmarks.col(m[3]);

        y[0] = matching.curFeatures.col(m[0]);
        y[1] = matching.curFeatures.col(m[1]);
        y[2] = matching.curFeatures.col(m[2]);
        y[3] = matching.curFeatures.col(m[3]);
        

        // Solve P3P using Lambda twist
        double a01 = (x[0] - x[1]).norm();
        double a02 = (x[0] - x[2]).norm();
        double a12 = (x[1] - x[2]).norm();

        double b01 = y[0].dot(y[1]);
        double b02 = y[0].dot(y[2]);
        double b12 = y[1].dot(y[2]);

        M01(0, 1) = -b01;
        M01(1, 0) = -b01;
        M02(0, 2) = -b02;
        M02(0, 1) = -b02;
        M12(1, 2) = -b12;
        M12(2, 1) = -b12;
        
        D1 = M01 * a12 - M12 * a01;
        D2 = M02 * a12 - M12 * a02;

        c[3] = D1.determinant();
        auto norm = 1 / c[0];
        c[0] = norm * D2.determinant();
        c[1] = norm * (
            D2.col(0).dot(D1.col(1).cross(D1.col(2))) + 
            D2.col(1).dot(D1.col(2).cross(D1.col(0))) +
            D2.col(2).dot(D1.col(0).cross(D1.col(1)))
        );
        c[2] = norm * (
            D1.col(0).dot(D2.col(1).cross(D2.col(2))) + 
            D1.col(1).dot(D2.col(2).cross(D2.col(0))) +
            D1.col(2).dot(D2.col(0).cross(D2.col(1)))
        );

        // Solve the cubic
        auto gamma = cubicSolver(c);
        D0 = D1 + gamma * D2;

        DiagonalDecomp decomp = diagonalize(D0);
        auto s1 = std::sqrt(-decomp.sigma2 / decomp.sigma1);
        auto s2 = -s1;
        
    }
}

void LfTracker::undistort(Eigen::Matrix3Xf &pts){    
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

namespace {
    FORCE_INLINE double cubicSolver(const std::array<double, 4> &c){
        auto c2 = c[2] * c[2];
        auto p = c[1] - (c[2] * c[2]) * (1.0 / 3.0);
        auto q = c[0] - (c[1] * c[2]) * (1.0 / 3.0) + (c2 * c[2]) * (2.0 / 27.0);
        auto p327 = (p * p * p) * (1.0 / 27.0);
        auto dis = (q * q) * (1.0 / 4.0) + p327;
        auto gamma = 0.0;
        auto offset = c[2] * (1.0f / 3.0f);
        if(dis <= 0){
            // Use the Viete method
            auto w = (-q * 0.5) / std::sqrt(-p327);
            w = std::clamp(w, -1.0, 1.0);
            auto theta = std::acos(w) * (1.0 / 3.0);
            auto coeff = 2.0 * (std::sqrt(-p * (1.0 / 3.0)));
            constexpr double PHASE = -2.0 * PI / 3.0;
            gamma = coeff * std::cos(theta - PHASE) - offset;
        }else{
            // Use Cardano Method
            auto sqrtdis = std::sqrt(dis);
            auto qd2 = -q * 0.5f;
            auto u = std::cbrt(qd2 + sqrtdis);
            auto v = std::cbrt(qd2 - sqrtdis);
            gamma = u + v - offset;
        }

        // Polish the root using N-R
        auto f = c[0] + gamma * (c[1] + gamma * (c[2] + gamma));
        auto fPrime = c[1] + gamma * (2.0 * c[2] + 3.0 * gamma);
        if (std::abs(fPrime) > K_EPS) {
            gamma -= f / fPrime;
        }

        return gamma;
    }

    // Gemeni ftw
    FORCE_INLINE DiagonalDecomp diagonalize(const Eigen::Matrix3d &D){
        DiagonalDecomp out;

        const double d00 = D(0, 0), d01 = D(0, 1), d02 = D(0, 2);
        const double d11 = D(1, 1), d12 = D(1, 2);
        const double d22 = D(2, 2);

        const double d00pd11 = d00 + d11;
        const double d00md11sd01sq = d00 * d11 - d01 * d01;

        // Characteristic quadratic: lambda^2 + p1 * lambda + p0 = 0
        const double p1 = -(d00pd11 + d22);
        const double p0 = d00md11sd01sq + (d00 * d22 - d02 * d02) + (d11 * d22 - d12 * d12);

        const double delta = std::max(0.0, p1 * p1 - 4.0 * p0);
        const double sqrtDelta = std::sqrt(delta);

        const double sigA = 0.5 * (-p1 + sqrtDelta);
        const double sigB = 0.5 * (-p1 - sqrtDelta);

        // Enforce |sigma1| >= |sigma2| for downstream numerical stability
        if (std::abs(sigA) >= std::abs(sigB)) {
            out.sigma1 = sigA;
            out.sigma2 = sigB;
        } else {
            out.sigma1 = sigB;
            out.sigma2 = sigA;
        }

        // Helper lambda to compute an eigenvector via 2x2 Cramer's rule
        auto getEigenvector = [&](double r) -> Eigen::Vector3d {
            const double c = r * (r - d00pd11) + d00md11sd01sq;
            
            // If the 2x2 minor is degenerate, fall back to cross product of rows
            if (std::abs(c) < K_EPS) {
                Eigen::Vector3d row0(d00 - r, d01, d02);
                Eigen::Vector3d row1(d01, d11 - r, d12);
                Eigen::Vector3d cross = row0.cross(row1);
                double norm = cross.norm();
                if (norm > K_EPS) {
                    return cross / norm;
                }
                Eigen::Vector3d row2(d02, d12, d22 - r);
                cross = row0.cross(row2);
                return cross.normalized();
            }

            const double invC = 1.0 / c;
            const double a1 = (r * d02 + d01 * d12 - d02 * d11) * invC;
            const double a2 = (r * d12 + d01 * d02 - d00 * d12) * invC;

            const double scale = 1.0 / std::sqrt(a1 * a1 + a2 * a2 + 1.0);
            return Eigen::Vector3d(a1 * scale, a2 * scale, scale);
        };

        out.e1 = getEigenvector(out.sigma1);
        out.e2 = getEigenvector(out.sigma2);

        return out;
    }
}


};