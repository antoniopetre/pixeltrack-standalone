#ifndef HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_AlpakaUtilities_interface_device_unique_ptr_h

#include <functional>
#include <memory>

#include <cuda_runtime.h>

#include "AlpakaCore/allocate_device.h"
#include "AlpakaCore/currentDevice.h"

namespace cms {
  namespace alpakatools {
    namespace device {
      namespace impl {
        // Additional layer of types to distinguish from host::unique_ptr
        class DeviceDeleter {
        public:
          DeviceDeleter() = default;  // for edm::Wrapper
          DeviceDeleter(int device, Queue stream) : device_{device}, stream_{stream} {}

          void operator()(void *ptr) {
            if (device_ >= 0) {
              free_device(device_, ptr, stream_);
            }
          }

        private:
          int device_ = -1;
          Queue stream_;
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;

      namespace impl {
        template <typename T>
        struct make_device_unique_selector {
          using non_array = cms::alpakatools::device::unique_ptr<T>;
        };
        template <typename T>
        struct make_device_unique_selector<T[]> {
          using unbounded_array = cms::alpakatools::device::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_device_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace device

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique(Queue stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      int dev = currentDevice();
      void *mem = allocate_device(dev, sizeof(T), stream);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{dev, stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(size_t n,
                                                                                              Queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the device memory is not supported");
      int dev = currentDevice();
      void *mem = allocate_device(dev, n * sizeof(element_type), stream);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{dev, stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename device::impl::make_device_unique_selector<T>::non_array make_device_unique_uninitialized(
        Queue stream) {
      int dev = currentDevice();
      void *mem = allocate_device(dev, sizeof(T), stream);
      return typename device::impl::make_device_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                              device::impl::DeviceDeleter{dev, stream}};
    }

    template <typename T>
    typename device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique_uninitialized(
        size_t n, Queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      int dev = currentDevice();
      void *mem = allocate_device(dev, n * sizeof(element_type), stream);
      return typename device::impl::make_device_unique_selector<T>::unbounded_array{
          reinterpret_cast<element_type *>(mem), device::impl::DeviceDeleter{dev, stream}};
    }

    template <typename T, typename... Args>
    typename device::impl::make_device_unique_selector<T>::bounded_array make_device_unique_uninitialized(Args &&...) =
        delete;
  }  // namespace alpakatools
}  // namespace cms

#endif
