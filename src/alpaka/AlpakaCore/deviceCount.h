#ifndef HeterogenousCore_AlpakaUtilities_deviceCount_h
#define HeterogenousCore_AlpakaUtilities_deviceCount_h

#include <cuda_runtime.h>
// #include "AlpakaCore/alpakaConfigAcc.h"

namespace cms {
  namespace alpakatools {
    int deviceCount() {
      int ndevices;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      // cudaGetDeviceCount(&ndevices);
      ndevices = alpaka::getDevCount<ALPAKA_ACCELERATOR_NAMESPACE::PltfAcc1>();
#else
      ndevices = 1;
#endif      
      return ndevices;
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
