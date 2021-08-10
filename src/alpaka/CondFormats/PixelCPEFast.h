#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/pixelCPEforGPU.h"

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaMemoryHelper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelCPEFast {
  public:
    PixelCPEFast(AlpakaDeviceBuf<pixelCPEforGPU::CommonParams> commonParams,
                 AlpakaDeviceBuf<pixelCPEforGPU::DetParams> detParams,
                 AlpakaDeviceBuf<pixelCPEforGPU::LayerGeometry> layerGeometry,
                 AlpakaDeviceBuf<pixelCPEforGPU::AverageGeometry> averageGeometry,
                 AlpakaDeviceBuf<pixelCPEforGPU::ParamsOnGPU> params)
        : m_commonParams(std::move(commonParams)),
          m_detParams(std::move(detParams)),
          m_layerGeometry(std::move(layerGeometry)),
          m_averageGeometry(std::move(averageGeometry)),
          m_params(std::move(params)) {}

    ~PixelCPEFast() = default;

    pixelCPEforGPU::ParamsOnGPU const* params() const { return alpaka::getPtrNative(m_params); }

  // The return value can only be used safely in kernels launched on
  // the same cudaStream, or after cudaStreamSynchronize.
  const pixelCPEforGPU::ParamsOnGPU getGPUProductAsync(cudaStream_t cudaStream) const {
    
    const auto &data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData &data, cudaStream_t stream) {
    // and now copy to device...
    auto cParams = cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::CommonParams>(1u);
    data.h_paramsOnGPU.m_commonParams = alpaka::getPtrNative(cParams);
    
    uint32_t size_detParams = alpaka::extent::getExtentVec(this->m_detParams)[0u];
    auto detParams = cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::DetParams>(size_detParams);
    data.h_paramsOnGPU.m_detParams = alpaka::getPtrNative(detParams);

    auto avgGeom = cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::AverageGeometry>(1u);
    data.h_paramsOnGPU.m_averageGeometry = alpaka::getPtrNative(avgGeom);
    
    auto layerGeom = cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::LayerGeometry>(1u);
    data.h_paramsOnGPU.m_layerGeometry = alpaka::getPtrNative(layerGeom);
    
    auto parGPU = cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::ParamsOnGPU>(1u);
    data.d_paramsOnGPU = alpaka::getPtrNative(parGPU);
    
    alpaka::prepareForAsyncCopy(cParams);
    alpaka::prepareForAsyncCopy(detParams);
    alpaka::prepareForAsyncCopy(avgGeom);
    alpaka::prepareForAsyncCopy(layerGeom);
    alpaka::prepareForAsyncCopy(parGPU);

    // alpaka::memcpy(queue, data.d_paramsOnGPU, data.h_paramsOnGPU, 1u);
    // alpaka::memcpy(queue, data.h_paramsOnGPU.m_commonParams, this->m_commonParamsGPU, 1u);
    // alpaka::memcpy(queue, data.h_paramsOnGPU.m_averageGeometry, this->m_averageGeometry, 1u);
    // alpaka::memcpy(queue, data.h_paramsOnGPU.m_layerGeometry, this->m_layerGeometry, 1u);
    // alpaka::memcpy(queue, data.h_paramsOnGPU.m_detParams, this->m_detParamsGPU.data(), size_detParams);
  });
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  return *data.d_paramsOnGPU;
#endif
  return data.h_paramsOnGPU;
}

  private:
    AlpakaDeviceBuf<pixelCPEforGPU::CommonParams> m_commonParams;
    AlpakaDeviceBuf<pixelCPEforGPU::DetParams> m_detParams;
    AlpakaDeviceBuf<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
    AlpakaDeviceBuf<pixelCPEforGPU::AverageGeometry> m_averageGeometry;
    AlpakaDeviceBuf<pixelCPEforGPU::ParamsOnGPU> m_params;

    struct GPUData {
   
    // not needed if not used on CPU...
    pixelCPEforGPU::ParamsOnGPU h_paramsOnGPU;
    pixelCPEforGPU::ParamsOnGPU *d_paramsOnGPU = nullptr;  // copy of the above on the Device
     ~GPUData() {
       if (d_paramsOnGPU != nullptr) {
        //cudafree
       }
     }
  };
  cms::alpakatools::ESProduct<GPUData> gpuData_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
