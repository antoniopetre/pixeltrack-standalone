#ifndef HeterogeneousCore_AlpakaUtilities_EventCache_h
#define HeterogeneousCore_AlpakaUtilities_EventCache_h

#include <vector>

#include <cuda_runtime.h>

#include "Framework/ReusableObjectHolder.h"
#include "AlpakaCore/SharedEventPtr.h"
#include "AlpakaCore/alpakaConfigCommon.h"
#include "AlpakaCore/deviceCount.h"

class CUDAService;

namespace cms {
  namespace alpakatools {
    class EventCache {
    public:

      using BareEvent = SharedEventPtr::element_type;

      EventCache();

      // Gets a (cached) CUDA event for the current device. The event
      // will be returned to the cache by the shared_ptr destructor. The
      // returned event is guaranteed to be in the state where all
      // captured work has completed, i.e. cudaEventQuery() == cudaSuccess.
      //
      // This function is thread safe
      template <typename T_Acc>
      SharedEventPtr get(T_Acc acc);

    private:
      friend class ::CUDAService;

      // thread safe
      template <typename T_Acc>
      SharedEventPtr makeOrGet(int dev, T_Acc acc);

      // not thread safe, intended to be called only from CUDAService destructor
      void clear();

      class Deleter {
      public:
        Deleter() = default;
        Deleter(int d) : device_{d} {}
        void operator()(alpaka::Event<Queue> *event) const;

      private:
        int device_ = -1;
      };

      std::vector<edm::ReusableObjectHolder<BareEvent, Deleter>> cache_;
    };

    // Gets the global instance of a EventCache
    // This function is thread safe
    EventCache& getEventCache();
  }  // namespace alpakatools
}  // namespace cms

#endif
