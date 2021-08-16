#ifndef ALPAKAQUEUE_H
#define ALPAKAQUEUE_H

namespace cms {
  namespace alpakatools {

    template <typename T_Acc>
    cms::alpakatools::Queue * createQueueNonBlocking(T_Acc const& acc) {
      using BufHost = alpaka::Buf<T_Acc, cms::alpakatools::Queue, alpaka::DimInt<1u>, uint32_t>;
      BufHost y(alpaka::allocBuf<cms::alpakatools::Queue, uint32_t>(acc, 1u));
      cms::alpakatools::Queue *x = alpaka::getPtrNative(y);
      printf("after alloc\n");
      return x;
    }

    template <typename T_Acc>
    cms::alpakatools::Queue createQueueNonBlockingNon(T_Acc const& acc) {
      cms::alpakatools::Queue x(acc);
      return x;
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAQUEUE_H
