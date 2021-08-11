#ifndef ALPAKAEVENT_H
#define ALPAKAEVENT_H

namespace cms {
  namespace alpakatools {

    template <typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto createQueueNonBlocking(T_Acc const& acc) {
      using AccQueueProperty = alpaka::NonBlocking;
      alpaka::Queue<T_Acc, AccQueueProperty> x(acc);
      return x;
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAEVENT_H
