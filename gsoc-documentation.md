# Portability for the Patatrack Pixel Track Reconstruction with Alpaka


## 1. Motivation

The Compact Muon Solenoid (CMS) experiment is one of the largest experiments at Large Hadron Collider (LHC) that has been built to search for new physics. CMS Software (CMSSW) is the framework utilized by the CMS experiment for Data Acquisition, Trigger and Event Reconstruction. The future upgrade of the LHC will add new challenges for CMS detector due to the larger amount of data that will be produced. 

One solution is to look towards a heterogeneous High-Level Trigger (HLT) Computing Farm, where the computing load can be distributed among different hardware (CPU, GPU, FPGAS). This project aims to use Portability Libraries like Alpaka to write code for the CMS Software (CMSSW), in order to have a single implementation which will be compiled on different architectures. 

In the last years, a new pixel track reconstruction that can run on GPUs (CUDA) has been implemented. The goal is to compare the timing performances between Alpaka implementation and native CUDA.


## 2. Task

The primary goal was to profile and optimize the Alpaka code for the GPU backend. The code was refactored, changing the main workflow to become similar with the Native CUDA implementation.


## 3. Accomplishments

### 3.1  add elements_with_stride class and test
<br> You can find the commit here: [https://github.com/antoniopetre/pixeltrack-standalone/commit/e7d4ec9efcd3163f3cf58853e541331035633b13)</br> <br>
This class simplifies "for" loops over all the elements. Indices are local to the grid. The class can iterate over 1-dimensional, 2-dimensional or 3-dimensional grids. The objective was to make the code easier to read. The performance is the same as the legacy version.

A test was added in order to understand how to use this class. 

### 3.2 Refactor CAHitNtupletGeneratorKernelsImpl using elements_with_stride class
<br> You can find the commit here: [https://github.com/antoniopetre/pixeltrack-standalone/commit/4f825fe6e66e32ac6cb6d5b35c96aea0622d40d2)</br> <br>
One kernel was refactored using the new ``` elements_with_stride``` class in order to make the code easier to read.

### 3.3 Profiling Alpaka
The times for the first Alpaka version can be found in the next table:

| Performances (events/s) | CUDA (NVidia V100) | CUDA (NVidia T4) | Serial | TBB |
| -------- | -------------- | -------------- |  -------------- | -------------- |
| Alpaka | 167.136 +- 3.064 | 137.721 +- 2.502 |  16.494 +- 0.126 | 6.560 +- 0.036 |

An extensive profiling was done using the ```nvprof``` command. Major time differences were found between Alpaka CUDA and Native CUDA using NVidia V100. These are shown in the next figure:

![alt text](https://github.com/antoniopetre/pixeltrack-standalone/blob/gsoc/plots/kernels_alpakaVsCUDA.png)

One possible cause for these differences could be the performance of atomics or barriers in Alpaka. Thus, tests for Alpaka and CUDA were developed in order to compare the two implementations.

### 3.4 Tests for atomic and barriers
<br> You can find the commit here: [https://github.com/antoniopetre/pixeltrack-standalone/commit/af2edd11a30d60fa2cf6cabbfbf778afea3d71d2)</br> <br>

Atomics and barriers tests were added in order to compare the performances between Native CUDA and Alpaka CUDA.

I tested the performance for 4 types of atomics:
+ atomics in global memory, block wide
+ atomics in shared memory, block wide
+ atomics in global memory, grid wide
+ atomics in shared memory, grid wide

I used two different types of variables: 
+ with single-precision (SP)
+ with double-precision (DP)

The results for the atomics are the following:

| Atomic Type (using NVidia V100) | Global Block | Shared Block | Global Grid | Shared Grid |
| -------- | -------------- | -------------- |  -------------- | -------------- |
| Native CUDA SP (s) | 7.342 +- 0.086 | 3.233 +- 0.008 | 29.964 +- 0.008 | 3.277 +- 0.018 |
| Alpaka CUDA SP (s) | 7.307 +- 0.057 | 3.332 +- 0.010 | 29.959 +- 0.001 | 3.343 +- 0.017 |
| Native CUDA DP (s) | 6.139 +- 0.045 | 3.959 +- 0.011 | 29.964 +- 0.005 | 3.941 +- 0.004 |
| Alpaka CUDA DP (s) | 6.173 +- 0.027 | 3.939 +- 0.003 | 29.960 +- 0.004 | 3.938 +- 0.001 |

| Atomic Type (using NVidia T4) | Global Block | Shared Block | Global Grid | Shared Grid |
| -------- | -------------- | -------------- |  -------------- | -------------- |
| Native CUDA SP (s) | 7.222 +- 0.033 | 6.126  +- 0.026 | 25.955 +- 1.7E-5 | 5.813  +- 0.015 |
| Alpaka CUDA SP (s) | 7.236 +- 0.028 | 5.983  +- 0.043 | 25.955 +- 4.0E-5 | 5.833  +- 0.044 |
| Native CUDA DP (s) | 5.776 +- 0.014 | 32.522 +- 0.121 | 25.955 +- 1.6E-5 | 32.340 +- 0.115 |
| Alpaka CUDA DP (s) | 5.787 +- 0.012 | 32.500 +- 0.143 | 25.955 +- 4.6E-5 | 32.409 +- 0.118 |

For barriers, I tested the performance of three types:
+ syncThreads - synchronize threads from a particular block
+ global threadfence - halt the current thread until all previous threads writes to global memory
+ shared threadfence - halt the current thread until all previous threads writes to shared memory

Again, I used two different types of variables:
+ with single-precision (SP)
+ with double-precision (DP)

The results for the barriers are the following:

| Barrier type (using NVidia V100) | syncThreads | Global threadFence | Shared threadFence
| -------- | -------------- | -------------- |  --------------
| Alpaka CUDA (s) | 0.072 +- 1.37E-5 | 0.0326 +- 3.6E-5 | 0.1056 +- 1.4E-4
| Native CUDA (s) | 0.073 +- 7.85E-6 | 0.0327 +- 5.4E-5 | 0.1057 +- 1.17E-5

| Barrier type (using NVidia T4) | syncThreads | Global threadFence | Shared threadFence
| -------- | -------------- | -------------- |  --------------
| Alpaka CUDA (s) | 0.1553 +- 6.2E-6 | 0.0436 +- 0.0019 | 0.3617 +- 0.0165
| Native CUDA (s) | 0.1534 +- 0.0039 | 0.0426 +- 0.0021 | 0.3545 +- 0.013

The conclusion is that the atomics and barriers performances in Alpaka is similar to the one in Native CUDA.

### 3.5 Optimize prefixScan implementation
<br> You can find the commit here: [https://github.com/antoniopetre/pixeltrack-standalone/commit/f414fbc6078cd6fde1cf8809551c89439553de9d)</br> <br>

The prefixScan algorithm is implemented in Alpaka using two kernels, while a single kernel is used for Native CUDA. The first one is executed by all the blocks and the second one is executed only by one block.

I refactored the prefixScan implementation in order to use a single kernel. In this way, the ```cudaLaunchKernel``` time will be reduced.

### 3.6 change assert in Alpaka application
<br> You can find the commit here: [https://github.com/antoniopetre/pixeltrack-standalone/commit/e0005b437ab5ed9f7278fa4ee833f1a2ab900970)</br> <br>

I profiled the kernels between Alpaka CUDA and Native CUDA using NVidia Nsight Compute. Some Alpaka kernels have the occupancy lower than the Native CUDA implementation. An example is shown in the next figure:

![alt text](https://github.com/antoniopetre/pixeltrack-standalone/blob/gsoc/plots/histo_alpakaVsCUDA.png)

I found out that the problem was related to the ```assert``` use in Alpaka. In Native CUDA, the assert is disabled by default, becoming enabled only if the macro ```GPU_DEBUG``` is set (this is not the case in Alpaka). Taking into consideration that ```assert``` in device code has a considerable impact on the performance, I changed the ```assert``` from the Alpaka implementation with the macro ```ALPAKA_ASSERT_OFFLOAD``` from the Alpaka library (compatible from Alpaka 0.6.1). This macro enables ```assert``` only if ```ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST``` is set. This change increased the occupancy for some kernels in Alpaka.

### 3.7 add ScopedContext
<br> You can find the commit here: [https://github.com/antoniopetre/pixeltrack-standalone/commit/cfbf694face559efad725684e7c81ae74a63e23e)</br> <br>

After I profiled the Alpaka CUDA and Native CUDA pixeltrack implementation again, I realised that the difference between the kernels time is only around 0.07 seconds, while the difference between the API calls for the two implementations is around 3.6 seconds. For example, in the Alpaka version 8004 streams are created (only two streams are created for the Native CUDA version). Therefore, the stream/event logic from the Alpaka version must be changed in order to obtain better results.

Several files were ported from the Native CUDA implementation and adapted for the Alpaka version. A clear problem is that some helper methods must be templated by the accelerator. In addition, asynchronous copy was added in the Alpaka implementation.

The stream/event logic was adapted succesfully in the Alpaka version (even though it's a more demanding task and some methods should be refactored in the future).

Only two "significant" problems exist within the current version of the Alpaka stream/event workflow:
+ the TBB version for Alpaka doesn't work. Even if the workflow is portable for Alpaka CUDA and Alpaka Serial, runtime errors exist for the TBB version. There are two possible causes:
  + the stream/event logic must become specialized for the TBB version (after I have added only the new workflow, there were no runtime errors, but the validation test doesn't work with the TBB version).
  + the asynchronous copy doesn't work for the TBB version (```Segmentation Fault``` - possible inconsistencies in the Alpaka library).
+ the new workflow uses a reusable object in order to instantiate only one stream which will be reused by the others events. I couldn't solve this even if I have spent my last two weeks on this error. Thus, the new workflow creates a new stream for every event (like the legacy version), but it uses the same workflow as the Native CUDA implementation. It will be easy to adapt the code to use the reusable object after this error will be caught, thus obtaining significant speedup for the Alpaka CUDA version. One possible problem could be that the "Framework" classes have been identical copied from the Native CUDA version (maybe in the future these classes must be adapted for the Alpaka library calls).

The times for the final Alpaka version are the following:

| Performances (events/s) | CUDA (NVidia V100) | CUDA (NVidia T4) | Serial | TBB |
| -------- | -------------- | -------------- |  -------------- | -------------- |
| Alpaka | 175.5402 +- 2.7107 | 139.5 +- 2.6614 |  17.41814 +- 0.284 | to be adapted |

### 3.8. Other things I have done

Among my tasks, I compared atomicMin operation between Native CUDA and Alpaka CUDA. I added a page on the Patatrack Wiki where I explained how to use compile and run the Alpaka version.
<br> You can find this page here: [https://patatrack.web.cern.ch/patatrack/wiki/cuda2alpaka/)</br> <br>

## 4. Bonus

Encouragement: always read the documentation!!