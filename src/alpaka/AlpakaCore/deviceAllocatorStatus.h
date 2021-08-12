#ifndef HeterogeneousCore_AlpakaUtilities_deviceAllocatorStatus_h
#define HeterogeneousCore_AlpakaUtilities_deviceAllocatorStatus_h

#include <map>

namespace cms {
  namespace alpakatools {
    namespace allocator {
      struct TotalBytes {
        size_t free;
        size_t live;
        size_t liveRequested;  // CMS: monitor also requested amount
        TotalBytes() { free = live = liveRequested = 0; }
      };
      /// Map type of device ordinals to the number of cached bytes cached by each device
      using GpuCachedBytes = std::map<int, TotalBytes>;
    }  // namespace allocator

    allocator::GpuCachedBytes deviceAllocatorStatus();
  }  // namespace cuda
}  // namespace cms

#endif
