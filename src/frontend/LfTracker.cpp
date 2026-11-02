#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <utility>

#include "frontend/LfTracker.hpp"
#include "CoreTypes.hpp"

namespace PoolSLAM{

bool LfTracker::passKeyframe(uint64_t id, double timestamp, const uint8_t *curBuffer){
    // Check if tracker is busy
    if(busy.load(std::memory_order_relaxed)){
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_passKeyframe);

        curFrame.id = id;
        curFrame.timestamp = timestamp;
        // Copy frame to a local buffer
        std::memcpy(localBuffer.get(), curBuffer, config.wImage * config.hImage);
        busy.store(true, std::memory_order_relaxed);
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