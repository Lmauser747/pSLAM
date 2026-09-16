#include "HAL.hpp"
#include <algorithm>
#include <mutex>
#include <optional>

constexpr double MAX_GAP = 0.1;

namespace PoolSLAM{
bool DepthStream::fastPush(const Depth &depth){
    // If the incoming data is in order, add it to the buffer
    if(depthBuffer.empty() || depth.timestamp >= depthBuffer.back().timestamp){
        if(depthBuffer.size() >= maxSize){
            depthBuffer.pop_front();
        }

        depthBuffer.push_back(depth);
        return true;
    }

    // Don't add data that is older than the oldest data in the queue
    if(depthBuffer.front().timestamp > depth.timestamp){
        return false;
    }

    // If the data is out-of-order, add it to the correct in-order position
    auto insertPosition = std::upper_bound(
        depthBuffer.begin(), depthBuffer.end(), depth.timestamp, 
        [](double t, const Depth &d){return t < d.timestamp;}
    );

    depthBuffer.insert(insertPosition, depth);

    if(depthBuffer.size() > maxSize){
        depthBuffer.pop_front();
    }

    return true;
}

bool DepthStream::pushDepth(const Depth &depth){
    {
        std::lock_guard<std::mutex> lock(m_depthBuffer);
        if(!fastPush(depth)){
            return false;
        }
    }

    c_depthBuffer.notify_one();
    return true;
}

std::optional<Depth> DepthStream::getInterDepth(double targetTime){

    std::lock_guard<std::mutex> lock(m_depthBuffer);

    // Check if the buffer is empty
    if(depthBuffer.empty()){
        return std::nullopt;
    }

    // Check if the buffer containts data suitible for interpolating at targetTime
    if((depthBuffer.front().timestamp < targetTime) || (depthBuffer.back().timestamp > targetTime)){
        return std::nullopt;
    }

    // Get iterator for the depth newer than the target time
    auto iterator = std::lower_bound(
        depthBuffer.begin(), depthBuffer.end(), targetTime,
        [](const Depth &s, double t){return s.timestamp < t;}
    );
    
    // If there is no older data, return the newer data
    if(iterator == depthBuffer.begin()){
        return *iterator;
    }

    // Make sure the difference between the two readings is not too large
    const Depth& right = *iterator;
    const Depth& left = *(iterator - 1);
    double diff = right.timestamp - left.timestamp;
    if(diff > MAX_GAP){
        return std::nullopt;
    }
    
    // Interpolate between the two time points
    double alpha = (targetTime -  left.timestamp) / diff; 
    double z = left.z + alpha * (right.z - left.z);

    Depth d;
    d.z = z;
    d.timestamp = targetTime;

    return d;
}

void DepthStream::pruneUntil(double time){
    {
        std::lock_guard<std::mutex> lock(m_depthBuffer);
        while(!depthBuffer.empty() && (depthBuffer.front().timestamp < time)){
            depthBuffer.pop_front();
        }
    }
}

void DepthStream::flush(){
    {
        std::lock_guard<std::mutex> lock(m_depthBuffer);
        depthBuffer.clear();
    }
}

}