#ifndef HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h
#define HeterogeneousCore_AlpakaUtilities_SharedStreamPtr_h

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {
    // cudaStream_t itself is a typedef for a pointer, for the use with
    // edm::ReusableObjectHolder the pointed-to type is more interesting
    // to avoid extra layer of indirection
    using SharedStreamPtr = std::shared_ptr<std::remove_pointer_t<cudaStream_t>>;
  }  // namespace alpakatools
}  // namespace cms

#endif
