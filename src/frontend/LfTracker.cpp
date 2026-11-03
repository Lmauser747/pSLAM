#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>
#include <utility>
#include <cstddef>
#include <cmath>

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
            frame.addDes(descriptor::QovDescriptor{0}, feature.ptCurr);
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
};