#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>
#include <utility>
#include <cstddef>
#include <cmath>

#include "frontend/LfTracker.hpp"
#include "descriptor/Orb.hpp"
#include "CoreTypes.hpp"

namespace PoolSLAM{

bool LfTracker::calculateDescriptors(uint64_t id, double timestamp, const uint8_t *curBuffer, const std::vector<TrackedFeature<ST_RADIUS>> &features){
    // Check if tracker is busy
    if(busy.load(std::memory_order_relaxed)){
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_passKeyframe);
        frame.features.clear();
        frame.id = id;
        frame.timestamp = timestamp;
        for(auto &feature : features){
            uint32_t m01 = 0; // Vectical center of mass
            uint32_t m10 = 0; // Horizontal center of mass
            for(ptrdiff_t y = -ORB_RADIUS; y <= ORB_RADIUS; y++){
                auto rowPtr  = curBuffer + static_cast<size_t>(y) * config.wImage;
                for(ptrdiff_t x = -ORB_RADIUS; x <= ORB_RADIUS; x++){
                    auto ptPtr = rowPtr + x;
                    m10 += *(ptPtr) * x;
                    m01 += *(ptPtr) * y;
                }
            }

            // get the angle of the feature
            auto theta = std::atan2(static_cast<double>(m01), static_cast<double>(m10));

            Eigen::Vector2i pt = feature.ptCurr.cast<Eigen::Vector2i>();
            auto centerPtr = curBuffer + static_cast<size_t>(pt.y()) * config.wImage + pt.x();

            auto des = descriptor::generateDescriptor(centerPtr, orbPairs.offsetAtAngleRad(theta));

            frame.addDes(des, pt);
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