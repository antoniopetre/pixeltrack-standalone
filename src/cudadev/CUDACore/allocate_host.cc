#include <limits>

#include "CUDACore/allocate_host.h"
#include "CUDACore/cudaCheck.h"

#include "getCachingHostAllocator.h"

namespace {
  const size_t maxAllocationSize = allocator::intPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_host(size_t nbytes, cudaStream_t stream) {
    void *ptr = nullptr;
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      ptr = allocator::getCachingHostAllocator().allocate(allocator::HostTraits::kHostDevice, nbytes, stream);
    } else {
      cudaCheck(cudaMallocHost(&ptr, nbytes));
    }
    return ptr;
  }

  void free_host(void *ptr) {
    if constexpr (allocator::policy == allocator::Policy::Caching) {
      allocator::getCachingHostAllocator().free(allocator::HostTraits::kHostDevice, ptr);
    } else {
      cudaCheck(cudaFreeHost(ptr));
    }
  }

}  // namespace cms::cuda
