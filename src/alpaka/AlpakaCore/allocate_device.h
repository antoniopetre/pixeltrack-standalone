#ifndef HeterogeneousCore_AlpakaUtilities_allocate_device_h
#define HeterogeneousCore_AlpakaUtilities_allocate_device_h

#include <cuda_runtime.h>
#include "SharedStreamPtr.h"

namespace cms {
  namespace alpakatools {
    // Allocate device memory
    void *allocate_device(int device, size_t nbytes, Queue stream);

    // Free device memory (to be called from unique_ptr)
    void free_device(int device, void *ptr, Queue stream);
  }  // namespace alpakatools
}  // namespace cms

#endif
