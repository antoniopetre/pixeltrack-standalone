#ifndef HeterogeneousCore_AlpakaUtilities_ScopedSetDevice_h
#define HeterogeneousCore_AlpakaUtilities_ScopedSetDevice_h

#include "AlpakaCore/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {
    class ScopedSetDevice {
    public:
      explicit ScopedSetDevice(int newDevice) {
        cudaCheck(cudaGetDevice(&prevDevice_));
        cudaCheck(cudaSetDevice(newDevice));
      }

      ~ScopedSetDevice() {
        // Intentionally don't check the return value to avoid
        // exceptions to be thrown. If this call fails, the process is
        // doomed anyway.
        cudaSetDevice(prevDevice_);
      }

    private:
      int prevDevice_;
    };
  }  // namespace alpakatools
}  // namespace cms

#endif
