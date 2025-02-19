#ifndef HeterogeneousCore_CUDACore_ESProduct_h
#define HeterogeneousCore_CUDACore_ESProduct_h

#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>

#include "CUDACore/EventCache.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/eventWorkHasCompleted.h"

namespace cms {
  namespace hip {
    template <typename T>
    class ESProduct {
    public:
      ESProduct() : gpuDataPerDevice_(deviceCount()) {
        for (size_t i = 0; i < gpuDataPerDevice_.size(); ++i) {
          gpuDataPerDevice_[i].m_event = getEventCache().get();
        }
      }
      ~ESProduct() = default;

      // transferAsync should be a function of (T&, hipStream_t)
      // which enqueues asynchronous transfers (possibly kernels as well)
      // to the CUDA stream
      template <typename F>
      const T& dataForCurrentDeviceAsync(hipStream_t cudaStream, F transferAsync) const {
        auto device = currentDevice();

        auto& data = gpuDataPerDevice_[device];

        // If GPU data has already been filled, we can return it
        // immediately
        if (not data.m_filled.load()) {
          // It wasn't, so need to fill it
          std::scoped_lock<std::mutex> lk{data.m_mutex};

          if (data.m_filled.load()) {
            // Other thread marked it filled while we were locking the mutex, so we're free to return it
            return data.m_data;
          }

          if (data.m_fillingStream != nullptr) {
            // Someone else is filling

            // Check first if the recorded event has occurred
            if (eventWorkHasCompleted(data.m_event.get())) {
              // It was, so data is accessible from all CUDA streams on
              // the device. Set the 'filled' for all subsequent calls and
              // return the value
              auto should_be_false = data.m_filled.exchange(true);
              assert(not should_be_false);
              data.m_fillingStream = nullptr;
            } else if (data.m_fillingStream != cudaStream) {
              // Filling is still going on. For other CUDA stream, add
              // wait on the CUDA stream and return the value. Subsequent
              // work queued on the stream will wait for the event to
              // occur (i.e. transfer to finish).
              cudaCheck(hipStreamWaitEvent(cudaStream, data.m_event.get(), 0),
                        "Failed to make a stream to wait for an event");
            }
            // else: filling is still going on. But for the same CUDA
            // stream (which would be a bit strange but fine), we can just
            // return as all subsequent work should be enqueued to the
            // same CUDA stream (or stream to be explicitly synchronized
            // by the caller)
          } else {
            // Now we can be sure that the data is not yet on the GPU, and
            // this thread is the first to try that.
            transferAsync(data.m_data, cudaStream);
            assert(data.m_fillingStream == nullptr);
            data.m_fillingStream = cudaStream;
            // Record in the cudaStream an event to mark the readiness of the
            // EventSetup data on the GPU, so other streams can check for it
            cudaCheck(hipEventRecord(data.m_event.get(), cudaStream));
            // Now the filling has been enqueued to the cudaStream, so we
            // can return the GPU data immediately, since all subsequent
            // work must be either enqueued to the cudaStream, or the cudaStream
            // must be synchronized by the caller
          }
        }

        return data.m_data;
      }

    private:
      struct Item {
        mutable std::mutex m_mutex;
        mutable SharedEventPtr m_event;  // guarded by m_mutex
        // non-null if some thread is already filling (hipStream_t is just a pointer)
        mutable hipStream_t m_fillingStream = nullptr;  // guarded by m_mutex
        mutable std::atomic<bool> m_filled = false;     // easy check if data has been filled already or not
        mutable T m_data;                               // guarded by m_mutex
      };

      std::vector<Item> gpuDataPerDevice_;
    };
  }  // namespace hip
}  // namespace cms

#endif
