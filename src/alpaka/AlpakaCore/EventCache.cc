#include "AlpakaCore/EventCache.h"
#include "AlpakaCore/currentDevice.h"
#include "AlpakaCore/deviceCount.h"
#include "AlpakaCore/eventWorkHasCompleted.h"
#include "AlpakaCore/ScopedSetDevice.h"
#include "alpakaEventHelper.h"

namespace cms::alpakatools {
  void EventCache::Deleter::operator()(alpaka::Event<Queue> *event) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      cudaEventDestroy(event);
#endif
    }
  }
//TODO ANTONIO

  // EventCache should be constructed by the first call to
  // getEventCache() only if we have CUDA devices present
  EventCache::EventCache() : cache_(deviceCount()) {}

  template <typename T_Acc>
  SharedEventPtr EventCache::get(T_Acc acc) {
    const auto dev = currentDevice();
    auto event = makeOrGet(dev, acc);
    // captured work has completed, or a just-created event
    if (eventWorkHasCompleted(*(event.get()))) {
      return event;
    }

    // Got an event with incomplete captured work. Try again until we
    // get a completed (or a just-created) event. Need to keep all
    // incomplete events until a completed event is found in order to
    // avoid ping-pong with an incomplete event.
    std::vector<SharedEventPtr> ptrs{std::move(event)};
    bool completed;
    do {
      event = makeOrGet(dev, acc);
      completed = eventWorkHasCompleted(*(event.get()));
      if (not completed) {
        ptrs.emplace_back(std::move(event));
      }
    } while (not completed);
    return event;
  }


  template <typename T_Acc>
  SharedEventPtr EventCache::makeOrGet(int dev, T_Acc acc) {
    return cache_[dev].makeOrGet([dev, acc]() {
      // alpaka::Event<Queue> event(dev);
      auto event = cms::alpakatools::createEvent<Queue>(acc);
      // it should be a bit faster to ignore timings
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      cudaEventCreateWithFlags(event, cudaEventDisableTiming);
#endif
      return std::unique_ptr<BareEvent, Deleter>(event, Deleter{dev});
// TODO ANTONIO
    });
  }

  void EventCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // EventCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(deviceCount());
  }

  EventCache& getEventCache() {
    // the public interface is thread safe
    static EventCache cache;
    return cache;
  }
}  // namespace cms::alpakatools
