#ifndef HeterogeneousCore_AlpakaUtilities_StreamCache_h
#define HeterogeneousCore_AlpakaUtilities_StreamCache_h

#include <vector>

#include <cuda_runtime.h>

#include "Framework/ReusableObjectHolder.h"
#include "AlpakaCore/SharedStreamPtr.h"
#include "alpakaQueueHelper.h"
#include "AlpakaCore/currentDevice.h"
#include "AlpakaCore/deviceCount.h"
#include "AlpakaCore/ScopedSetDevice.h"

class CUDAService;

namespace cms {
  namespace alpakatools {
    class StreamCache {
    public:
      using BareStream = SharedStreamPtr::element_type;

      StreamCache() : cache_(deviceCount()) {}

      // Gets a (cached) CUDA stream for the current device. The stream
      // will be returned to the cache by the shared_ptr destructor.
      // This function is thread safe
      // template <typename T_Acc>
      // SharedStreamPtr get(T_Acc acc);

      template <typename T_Acc>
      SharedStreamPtr get(T_Acc acc) {
        const auto dev = currentDevice();
        cms::alpakatools::Queue *stream = cms::alpakatools::createQueueNonBlocking<T_Acc>(acc);
        cms::alpakatools::Queue stream2 = cms::alpakatools::createQueueNonBlockingNon<T_Acc>(acc);
        printf("cache1\n");
        *stream = stream2;
        printf("cache2\n");
        SharedStreamPtr x = std::make_shared<cms::alpakatools::Queue*>(stream);
        printf("cache3\n");
        // SharedStreamPtr x = cache_[dev].makeOrGet([dev, acc]() {
        // cms::alpakatools::Queue *stream = cms::alpakatools::createQueueNonBlocking<T_Acc>(acc);
        //Todo antonio
        //     return std::unique_ptr<BareStream, Deleter>(&stream,  Deleter{dev});
        // });
        // alpaka::wait(*(x.get()));
        return x;
      }

    private:
      friend class ::CUDAService;
      // not thread safe, intended to be called only from CUDAService destructor
      void clear() {
        // Reset the contents of the caches, but leave an
        // edm::ReusableObjectHolder alive for each device. This is needed
        // mostly for the unit tests, where the function-static
        // StreamCache lives through multiple tests (and go through
        // multiple shutdowns of the framework).
        cache_.clear();
        cache_.resize(deviceCount());
      }

      class Deleter {
      public:
        Deleter() = default;
        Deleter(int d) : device_{d} {}
        // void operator()(Queue *stream) const;
        void operator()(Queue **stream) const {
          if (device_ != -1) {
            ScopedSetDevice deviceGuard{device_};
      // #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      //       cudaStreamDestroy(*stream);
      // #endif
            // delete stream;
            (*stream)->~Queue();
          }
        }

      private:
        int device_ = -1;
      };

      std::vector<edm::ReusableObjectHolder<BareStream, Deleter>> cache_;
      // AlpakaDeviceBuf<edm::ReusableObjectHolder<BareStream, Deleter>> cache_2;
    };

    // Gets the global instance of a StreamCache
    // This function is thread safe
    StreamCache& getStreamCache();
    // StreamCache& getStreamCache() {
    //   // the public interface is thread safe
    //   static StreamCache cache;
    //   return cache;
    // }
  }  // namespace alpakatools
}  // namespace cms

#endif
