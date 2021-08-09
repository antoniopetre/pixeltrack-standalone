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
    #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // data.h_paramsOnGPU.m_commonParams = cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::CommonParams>(1u);
    // data.h_paramsOnGPU.m_detParams = cms::alpakatools::allocDeviceBuf(this->m_detParams.size());
    // data.h_paramsOnGPU.m_averageGeometry = cms::alpakatools::allocDeviceBuf(1);
    // data.h_paramsOnGPU.m_layerGeometry = cms::alpakatools::allocDeviceBuf(1);
    // data.d_paramsOnGPU = cms::alpakatools::allocDeviceBuf(1);


    cudaMalloc((void **)&data.h_paramsOnGPU.m_commonParams, sizeof(pixelCPEforGPU::CommonParams));
    // cudaMalloc((void **)&data.h_paramsOnGPU.m_detParams,
    //                     this->m_detParams.size() * sizeof(pixelCPEforGPU::DetParams)); //size doesn't exist
    cudaMalloc((void **)&data.h_paramsOnGPU.m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry));
    cudaMalloc((void **)&data.h_paramsOnGPU.m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry));
    cudaMalloc((void **)&data.d_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU));

    cudaMemcpyAsync(
        data.d_paramsOnGPU, &data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU), cudaMemcpyDefault, stream);
    cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_commonParams,
                              &this->m_commonParams,
                              sizeof(pixelCPEforGPU::CommonParams),
                              cudaMemcpyDefault,
                              stream);
    cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_averageGeometry,
                              &this->m_averageGeometry,
                              sizeof(pixelCPEforGPU::AverageGeometry),
                              cudaMemcpyDefault,
                              stream);
    cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_layerGeometry,
                              &this->m_layerGeometry,
                              sizeof(pixelCPEforGPU::LayerGeometry),
                              cudaMemcpyDefault,
                              stream);
    cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_detParams,
                              alpaka::getPtrNative(this->m_detParams),
                              sizeof(alpaka::getPtrNative(this->m_detParams)),
                              cudaMemcpyDefault,
                              stream); //size + data dont exist
    
    #endif
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
