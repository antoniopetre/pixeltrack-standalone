#ifndef ALPAKAEVENT_H
#define ALPAKAEVENT_H

// #include "AlpakaCore/alpakaConfig.h"
// #include "AlpakaCore/alpakaDevices.h"

// using namespace alpaka_common;

namespace cms {
  namespace alpakatools {

    template <typename Queue, typename T_Acc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE auto createEvent(T_Acc const& acc) {
      return alpaka::Event<Queue>(acc);
    }

  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAEVENT_H
