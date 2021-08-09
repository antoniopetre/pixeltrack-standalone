// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "AlpakaCore/cudaCheck.h"
/*
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"


const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    // allocate
    cudaCheck(cudaMalloc(&data.cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));

    // transfer
    cudaCheck(cudaMemcpyAsync(
        data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyDefault, stream));
  });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, cudaStream_t stream) {
        cudaCheck(cudaMalloc((void**)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
        cudaCheck(cudaMemcpyAsync(data.modToUnpDefault,
                                  this->modToUnpDefault.data(),
                                  this->modToUnpDefault.size() * sizeof(unsigned char),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.modToUnpDefault;
}
*/
