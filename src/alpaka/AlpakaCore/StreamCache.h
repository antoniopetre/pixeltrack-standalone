#ifndef HeterogeneousCore_AlpakaUtilities_StreamCache_h
#define HeterogeneousCore_AlpakaUtilities_StreamCache_h

#include <vector>

#include <cuda_runtime.h>

#include "Framework/ReusableObjectHolder.h"
#include "AlpakaCore/SharedStreamPtr.h"
#include "alpakaQueueHelper.h"
#include "AlpakaCore/currentDevice.h"
#include "AlpakaCore/deviceCount.h"

class CUDAService;

namespace cms {
  namespace alpakatools {
    class StreamCache {
    public:
      using BareStream = SharedStreamPtr::element_type;

      StreamCache();

      // Gets a (cached) CUDA stream for the current device. The stream
      // will be returned to the cache by the shared_ptr destructor.
      // This function is thread safe
      // template <typename T_Acc>
      // SharedStreamPtr get(T_Acc acc);

      template <typename T_Acc>
      ALPAKA_FN_HOST SharedStreamPtr get(T_Acc acc) {

        const auto dev = currentDevice();
        using AccQueueProperty = alpaka::NonBlocking;
        // using QueueNon = alpaka::Queue<T_Acc, AccQueueProperty>;
        cms::alpakatools::Queue stream = cms::alpakatools::createQueueNonBlocking<T_Acc>(acc);
        // SharedStreamPtr x(new cms::alpakatools::Queue(stream));
        SharedStreamPtr x = std::make_shared<cms::alpakatools::Queue>(stream);
        // SharedStreamPtr x(cms::alpakatools::Queue(stream));
        // SharedStreamPtr x = std::make_shared<cms::alpakatools::Queue>();
        // return cache_[dev].makeOrGet([stream, dev, acc]() {
        //   Queue stream;
            // auto stream = cms::alpakatools::createQueueNonBlocking<T_Acc>(acc);
          //cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            // return std::unique_ptr<BareStream, Deleter>(&stream, Deleter{dev});
            //Todo antonio
            // return std::unique_ptr<BareStream>(&stream);
        // });
        return x;
      }

    private:
      friend class ::CUDAService;
      // not thread safe, intended to be called only from CUDAService destructor
      void clear();

      class Deleter {
      public:
        Deleter() = default;
        Deleter(int d) : device_{d} {}
        void operator()(Queue *stream) const;

      private:
        int device_ = -1;
      };

      std::vector<edm::ReusableObjectHolder<BareStream, Deleter>> cache_;
    };

    // Gets the global instance of a StreamCache
    // This function is thread safe
    StreamCache& getStreamCache();
  }  // namespace alpakatools
}  // namespace cms

#endif
