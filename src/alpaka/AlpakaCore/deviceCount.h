#ifndef HeterogenousCore_AlpakaUtilities_deviceCount_h
#define HeterogenousCore_AlpakaUtilities_deviceCount_h

#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {
    inline int deviceCount() {
      int ndevices;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      cudaGetDeviceCount(&ndevices);
#endif
      return ndevices;
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
