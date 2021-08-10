#include "deviceCount.h"

namespace cms::alpakatools {
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
}  // namespace cms::alpakatools