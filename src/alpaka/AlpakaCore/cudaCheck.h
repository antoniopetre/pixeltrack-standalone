
/*
#ifndef HeterogeneousCore_AlpakaUtilities_cudaCheck_h
#define HeterogeneousCore_AlpakaUtilities_cudaCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// Boost headers
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {

    [[noreturn]] inline void abortOnCudaError(const char* file,
                                              int line,
                                              const char* cmd,
                                              const char* error,
                                              const char* message,
                                              const char* description = nullptr) {
      std::ostringstream out;
      out << "\n";
      out << file << ", line " << line << ":\n";
      out << "cudaCheck(" << cmd << ");\n";
      out << error << ": " << message << "\n";
      if (description)
        out << description << "\n";

      out << "\nCurrent stack trace:\n";
      out << boost::stacktrace::stacktrace();
      out << "\n";

      throw std::runtime_error(out.str());
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    inline bool cudaCheck_(
        const char* file, int line, const char* cmd, CUresult result, const char* description = nullptr) {
      if (result == CUDA_SUCCESS)
        return true;

      const char* error;
      const char* message;
      cuGetErrorName(result, &error);
      cuGetErrorString(result, &message);
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }

    inline bool cudaCheck_(
        const char* file, int line, const char* cmd, cudaError_t result, const char* description = nullptr) {
      if (result == cudaSuccess)
        return true;

      const char* error = cudaGetErrorName(result);
      const char* message = cudaGetErrorString(result);
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }
#endif
  }  // namespace alpakatools
}  // namespace cms

#define cudaCheck(ARG, ...) (cms::alpakatools::cudaCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_AlpakaUtilities_cudaCheck_h
*/
