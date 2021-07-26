#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#include <tbb/task_scheduler_init.h>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDivHelper.h"
#include "CUDACore/CMSUnrollLoop.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct ball {
  template <typename T_Acc>
  ALPAKA_FN_ACC void operator()(const T_Acc& acc, unsigned int size) const {

    auto blockIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
    //auto mask = __ballot_sync(0xffffffff, blockIdxLocal < size);
  }
};

int main() {
    unsigned int size = 1024;
    const Vec1 threadsPerBlockOrElementsPerThread1(Vec1::all(32));
    const Vec1 blocksPerGrid1(Vec1::all(32));
    auto workDivMultiBlockInit1 =
      cms::alpakatools::make_workdiv(blocksPerGrid1, threadsPerBlockOrElementsPerThread1);

    const DevAcc1 device_1(alpaka::getDevByIdx<PltfAcc1>(0u));
    alpaka::Queue<DevAcc1, alpaka::Blocking> queue_1_0(device_1);

    alpaka::enqueue(queue_1_0, alpaka::createTaskKernel<Acc1>(workDivMultiBlockInit1, 
                  ball(), size));
    alpaka::wait(queue_1_0);

}