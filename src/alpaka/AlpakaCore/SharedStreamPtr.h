#ifndef HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h
#define HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h

#include <memory>
#include <type_traits>
#include "AlpakaCore/alpakaConfigCommon.h"

namespace cms {
  namespace alpakatools {
    // cudaStream_t itself is a typedef for a pointer, for the use with
    // edm::ReusableObjectHolder the pointed-to type is more interesting
    // to avoid extra layer of indirection
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    using SharedStreamPtr = std::shared_ptr<std::remove_pointer_t<alpaka::QueueCudaRtNonBlocking>>;
#else
    using SharedStreamPtr = std::shared_ptr<std::remove_pointer_t<alpaka::QueueCpuNonBlocking>>;
#endif
  }  // namespace alpakatools
}  // namespace cms

#endif
