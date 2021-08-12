#include <cassert>
#include <limits>

#include <cuda_runtime.h>

#include "AlpakaCore/ScopedSetDevice.h"
#include "AlpakaCore/allocate_device.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::alpakatools::allocator::binGrowth, cms::alpakatools::allocator::maxBin);
}

namespace cms::alpakatools {
  void *allocate_device(int dev, size_t nbytes, Queue stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
    //   cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, &stream);
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(dev);
    //   cudaCheck(cudaMallocAsync(&ptr, nbytes, stream));
        cudaMallocAsync(&ptr, nbytes, stream);
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaMalloc(&ptr, nbytes);
    }
    return ptr;
  }

  void free_device(int device, void *ptr, Queue stream) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingDeviceAllocator().DeviceFree(device, ptr);
#if CUDA_VERSION >= 11020
    } else if constexpr (allocator::policy == allocator::Policy::Asynchronous) {
      ScopedSetDevice setDeviceForThisScope(device);
    //   cudaCheck(cudaFreeAsync(ptr, stream));
#endif
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
    //   cudaCheck(cudaFree(ptr));
    }
  }

}  // namespace cms::alpakatools
