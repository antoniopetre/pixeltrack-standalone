
#ifndef HeterogeneousCore_AlpakaUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_AlpakaUtilities_eventWorkHasCompleted_h

#include "AlpakaCore/alpakaConfigCommon.h"

namespace cms {
  namespace alpakatools {
    /**
   * Returns true if the work captured by the event (=queued to the
   * CUDA stream at the point of cudaEventRecord()) has completed.
   *
   * Returns false if any captured work is incomplete.
   *
   * In case of errors, throws an exception.
   */

    inline bool eventWorkHasCompleted(alpaka::Event<Queue> event) {
      const auto ret = alpaka::isComplete(event);
      if (ret == true) {
        return true;
      } else if (ret == false) {
        return false;
      }
      return false;  // to keep compiler happy
      // TODO ANTONIO
    }
  }  // namespace alpakatools
}  // namespace cms

#endif
