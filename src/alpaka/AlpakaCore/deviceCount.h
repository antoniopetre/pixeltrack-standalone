#ifndef HeterogenousCore_AlpakaUtilities_deviceCount_h
#define HeterogenousCore_AlpakaUtilities_deviceCount_h

#include "AlpakaCore/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {
    inline int deviceCount() {
      int ndevices;
      cudaCheck(cudaGetDeviceCount(&ndevices));
      return ndevices;
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
