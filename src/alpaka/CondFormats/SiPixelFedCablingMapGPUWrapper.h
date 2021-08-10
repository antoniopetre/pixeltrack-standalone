#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/ESProduct.h"
#include <cuda_runtime.h>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelFedCablingMapGPUWrapper {
  public:
    using CablingMapDeviceBuf = AlpakaDeviceBuf<SiPixelFedCablingMapGPU>;

    explicit SiPixelFedCablingMapGPUWrapper(CablingMapDeviceBuf cablingMap, bool quality)
        : cablingMapDevice_{std::move(cablingMap)}, hasQuality_{quality} {}
    ~SiPixelFedCablingMapGPUWrapper() = default;

    bool hasQuality() const { return hasQuality_; }

    const SiPixelFedCablingMapGPU* cablingMap() const { return alpaka::getPtrNative(cablingMapDevice_); }

    // returns pointer to GPU memory
    const SiPixelFedCablingMapGPU *getGPUProductAsync(cudaStream_t cudaStream) const;

    // returns pointer to GPU memory
    const unsigned char *getModToUnpAllAsync(cudaStream_t cudaStream) const;

  private:
    CablingMapDeviceBuf cablingMapDevice_;
    bool hasQuality_;

    //std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnpDefault;
    // AlpakaDeviceBuf<unsigned char> modToUnpDefault;

    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    struct GPUData {
      CablingMapDeviceBuf cablingMapDevice;
       ~GPUData() {

       }
    };
    // cms::alpakatools::ESProduct<GPUData> gpuData_;

    struct ModulesToUnpack {
      
      unsigned char *modToUnpDefault = nullptr;  // pointer to GPU
      ~ModulesToUnpack() {

      }
    };
    // cms::alpakatools::ESProduct<ModulesToUnpack> modToUnp_;
    #endif
    
    };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
