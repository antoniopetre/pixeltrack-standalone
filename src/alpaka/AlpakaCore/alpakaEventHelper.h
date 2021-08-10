#ifndef ALPAKAEVENT_H
#define ALPAKAEVENT_H

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaDevices.h"

using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    template <typename Queue, typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto createEvent(T_Acc const& acc) {
      return alpaka::Event<Queue>(acc);
    }

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    // device-wide memory fence
    // CPU serial implementation: no fence needed
    template <typename Queue, typename TDim, typename TIdx>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto createEvent(alpaka::AccCpuSerial<TDim, TIdx> const& acc) {
      return alpaka::Event<Queue>(acc);
    }
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    // device-wide memory fence
    // CPU parallel implementation using TBB tasks: std::atomic_thread_fence()
    template <typename Queue, typename TDim, typename TIdx>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto createEvent(alpaka::AccCpuTbbBlocks<TDim, TIdx> const& acc) {
      return alpaka::Event<Queue>(acc);
    }
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // device-wide memory fence
    // GPU parallel implementation using CUDA: __threadfence()
    template <typename Queue, typename TDim, typename TIdx>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto createEvent(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc) {
      return alpaka::Event<Queue>(acc);
    }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAEVENT_H
