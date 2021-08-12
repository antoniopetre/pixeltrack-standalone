#ifndef ALPAKAQUEUE_H
#define ALPAKAQUEUE_H

namespace cms {
  namespace alpakatools {

    template <typename T_Acc>
    ALPAKA_FN_INLINE auto createQueueNonBlocking(T_Acc const& acc) {
    // #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED || 
      // using AccQueueProperty = alpaka::NonBlocking;
      // alpaka::Queue<T_Acc, AccQueueProperty> x(acc);
      cms::alpakatools::Queue x(acc);
      return x;
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAQUEUE_H
