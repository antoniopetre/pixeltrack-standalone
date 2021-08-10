// #include "AlpakaCore/StreamCache.h"
// #include "AlpakaCore/currentDevice.h"
// #include "AlpakaCore/deviceCount.h"
// #include "AlpakaCore/ScopedSetDevice.h"
// #include "alpakaQueueHelper.h"

// namespace cms::alpakatools {
//   void StreamCache::Deleter::operator()(Queue *stream) const {
//     if (device_ != -1) {
//       ScopedSetDevice deviceGuard{device_};
// #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
//       cudaStreamDestroy(*stream);
// #endif
//     }
//   }

//   // StreamCache should be constructed by the first call to
//   // getStreamCache() only if we have CUDA devices present
//   StreamCache::StreamCache() : cache_(cms::alpakatools::deviceCount()) {}

//   template <typename T_Acc>
//   SharedStreamPtr StreamCache::get(T_Acc acc) {

//     const auto dev = currentDevice();
//     return cache_[dev].makeOrGet([dev, acc]() {
//     //   Queue stream;
//         auto stream = createQueueNonBlocking<T_Acc>(acc);
//       //cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
//       return std::unique_ptr<BareStream, Deleter>(stream, Deleter{dev});
//     });
// }

//   void StreamCache::clear() {
//     // Reset the contents of the caches, but leave an
//     // edm::ReusableObjectHolder alive for each device. This is needed
//     // mostly for the unit tests, where the function-static
//     // StreamCache lives through multiple tests (and go through
//     // multiple shutdowns of the framework).
//     cache_.clear();
//     cache_.resize(cms::alpakatools::deviceCount());
//   }

//   StreamCache& getStreamCache() {
//     // the public interface is thread safe
//     static StreamCache cache;
//     return cache;
//   }
// }  // namespace cms::alpakatools
