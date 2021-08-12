#include "AlpakaCore/deviceAllocatorStatus.h"

#include "getCachingDeviceAllocator.h"

namespace cms::alpakatools {
  allocator::GpuCachedBytes deviceAllocatorStatus() { return allocator::getCachingDeviceAllocator().CacheStatus(); }
}  // namespace cms::alpakatools
