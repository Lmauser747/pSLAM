#include <atomic>

#include "Synchronizer.hpp"
#include "HALTypes.hpp"
#include "SystemTypes.hpp"


namespace PoolSLAM{

void Synchronizer::threadTask(){
    while(isRunning.load(std::memory_order_relaxed)){
        PixelArray *nextFrame = nullptr;
        SensorHealth camHealth = imageStream.waitForFrame(nextFrame);

        // Check if stopped mid loop
        if(!isRunning.load(std::memory_order_relaxed)){
            if(nextFrame){
                imageStream.recycleFrame(nextFrame);
            }
            break;
        }

        // Deal with degraded/disabled camera
        if(camHealth == SensorHealth::DISABLED){
            syncStatus.store(SyncStatus::FATALCAM, std::memory_order_relaxed);
            syncFatalShutdown(syncStatus, nextFrame);
            break;
        }else if(camHealth == SensorHealth::DEGRADED || nextFrame == nullptr){
            if(nextFrame){
                imageStream.recycleFrame(nextFrame);
            }
            break;
        }

    }
        

}

}