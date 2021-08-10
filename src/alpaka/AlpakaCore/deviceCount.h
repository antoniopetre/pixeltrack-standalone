#ifndef HeterogenousCore_AlpakaUtilities_deviceCount_h
#define HeterogenousCore_AlpakaUtilities_deviceCount_h

#include <cuda_runtime.h>
// #include "AlpakaCore/alpakaConfigAcc.h"

namespace cms {
  namespace alpakatools {
    int deviceCount() {
      int ndevices;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      cudaGetDeviceCount(&ndevices);
    //   ndevices = alpaka::getDevCount<alpaka::Pltf<DevAcc1>>();
#endif
      // TODO ANTONIOå
      
      return ndevices;
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
