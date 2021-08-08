#ifndef HeterogenousCore_AlpakaUtilities_currentDevice_h
#define HeterogenousCore_AlpakaUtilities_currentDevice_h

#include "AlpakaCore/cudaCheck.h"

#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {
    inline int currentDevice() {
      int dev;
      cudaCheck(cudaGetDevice(&dev));
      return dev;
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
