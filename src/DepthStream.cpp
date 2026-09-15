#include "HAL.hpp"
#include <algorithm>
#include <mutex>

namespace PoolSLAM{
    void DepthStream::pushDepth(const Depth &depth){
        // Take the mutex and pushes it into the queue
        {
            std::lock_guard<std::mutex> lock(m_depthQueue);
            depthQueue.push(depth);
        }
        
        // Notifies other threads data has been added to the queue
        c_depthQueue.notify_one();
    }

    Depth DepthStream::popDepth(){
        // Wait until notified and then lock the mutex
        std::unique_lock<std::mutex> lock(m_depthQueue);
        c_depthQueue.wait(lock, depthQueue.size() > 0);
        
        // Pop the depth from the front of the queue
        Depth depth = depthQueue.front();
        depthQueue.pop();

        return depth;
    }

    Depth DepthStream::peekDepth(){
        // Wait until notified and then lock the mutex
        std::unique_lock<std::mutex> lock(m_depthQueue);
        c_depthQueue.wait(lock, depthQueue.size() > 0);

        // Return the depth at the front of the queue without removing it
        return depthQueue.front();
    }
}