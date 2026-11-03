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

namespace PoolSLAM{

bool LfTracker::passKeyframe(uint64_t id, double timestamp, const uint8_t *curBuffer, const std::vector<TrackedFeature<FEATURE_RADIUS>> &features){
    // Check if tracker is busy
    if(busy.load(std::memory_order_relaxed)){
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_passKeyframe);
        for(size_t i = 0; i < config.wImage * config.hImage; i++){
            localBuffer[i] = curBuffer[i];
        }

        frame.id = id;
        frame.timestamp = timestamp;
        frame.features.clear();

        for(auto &feature : features){
            frame.addDes(descriptor::QovDescriptor{0}, feature.ptCurr.cast<Eigen::Vector2i>());
        }
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