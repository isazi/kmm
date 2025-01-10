#pragma once

#include <cstddef>

#include "kmm/core/macros.hpp"

#ifdef KMM_USE_CUDA
    #include <cublas_v2.h>
    #include <cuda.h>
    #include <cuda_bf16.h>
    #include <cuda_fp16.h>
    #include <cuda_runtime_api.h>
#elif KMM_USE_HIP
    #include <hip_bf16.h>
    #include <hip_runtime_api.h>
#endif

namespace kmm {

#ifdef KMM_USE_CUDA
using half_type = __half;
using bfloat16_type = __nv_bfloat16;

    #define GPU_DEVICE_ATTRIBUTE_MAX                   CU_DEVICE_ATTRIBUTE_MAX
    #define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR \
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR \
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    #define GPU_MEMHOSTALLOC_PORTABLE            CU_MEMHOSTALLOC_PORTABLE
    #define GPU_MEMHOSTALLOC_DEVICEMAP           CU_MEMHOSTALLOC_DEVICEMAP
    #define GPU_SUCCESS                          CUDA_SUCCESS
    #define GPU_ERROR_OUT_OF_MEMORY              CUDA_ERROR_OUT_OF_MEMORY
    #define GPU_MEM_ALLOCATION_TYPE_PINNED       CU_MEM_ALLOCATION_TYPE_PINNED
    #define GPU_MEM_HANDLE_TYPE_NONE             CU_MEM_HANDLE_TYPE_NONE
    #define GPU_MEM_LOCATION_TYPE_DEVICE         CU_MEM_LOCATION_TYPE_DEVICE
    #define GPU_ERROR_UNKNOWN                    CUDA_ERROR_UNKNOWN
    #define GPU_STREAM_NON_BLOCKING              CU_STREAM_NON_BLOCKING
    #define GPU_EVENT_WAIT_DEFAULT               CU_EVENT_WAIT_DEFAULT
    #define GPU_ERROR_NOT_READY                  CUDA_ERROR_NOT_READY
    #define GPU_EVENT_DISABLE_TIMING             CU_EVENT_DISABLE_TIMING
    #define GPU_MEMORYTYPE_HOST                  CU_MEMORYTYPE_HOST
    #define GPU_MEMORYTYPE_DEVICE                CU_MEMORYTYPE_DEVICE
    #define GPU_ERROR_NO_DEVICE                  CUDA_ERROR_NO_DEVICE
    #define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE    CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    #define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    #define GPU_CTX_MAP_HOST                     CU_CTX_MAP_HOST
    #define gpuCtxGetDevice                      cuCtxGetDevice
    #define gpuDeviceGetName                     cuDeviceGetName
    #define gpuDeviceGetAttribute                cuDeviceGetAttribute
    #define gpuMemGetInfo                        cuMemGetInfo
    #define gpuMemcpyHtoDAsync                   cuMemcpyHtoDAsync
    #define gpuMemcpyDtoHAsync                   cuMemcpyDtoHAsync
    #define gpuStreamSynchronize                 cuStreamSynchronize
    #define gpuMemsetD8Async                     cuMemsetD8Async
    #define gpuMemsetD16Async                    cuMemsetD16Async
    #define gpuMemsetD32Async                    cuMemsetD32Async
    #define gpuMemcpyAsync                       cuMemcpyAsync
    #define gpuMemHostAlloc                      cuMemHostAlloc
    #define gpuMemFreeHost                       cuMemFreeHost
    #define gpuMemAlloc                          cuMemAlloc
    #define gpuMemFree                           cuMemFree
    #define gpuMemPoolCreate                     cuMemPoolCreate
    #define gpuMemPoolDestroy                    cuMemPoolDestroy
    #define gpuMemAllocFromPoolAsync             cuMemAllocFromPoolAsync
    #define gpuMemFreeAsync                      cuMemFreeAsync
    #define gpuCtxGetStreamPriorityRange         cuCtxGetStreamPriorityRange
    #define gpuStreamCreateWithPriority          cuStreamCreateWithPriority
    #define gpuStreamQuery                       cuStreamQuery
    #define gpuStreamDestroy                     cuStreamDestroy
    #define gpuEventSynchronize                  cuEventSynchronize
    #define gpuEventRecord                       cuEventRecord
    #define gpuStreamWaitEvent                   cuStreamWaitEvent
    #define gpuEventQuery                        cuEventQuery
    #define gpuEventDestroy                      cuEventDestroy
    #define gpuEventCreate                       cuEventCreate
    #define gpuMemcpyHtoD                        cuMemcpyHtoD
    #define gpuMemcpy2DAsync                     cuMemcpy2DAsync
    #define gpuMemcpy2D                          cuMemcpy2D
    #define gpuMemcpyDtoH                        cuMemcpyDtoH
    #define gpuMemcpyDtoDAsync                   cuMemcpyDtoDAsync
    #define gpuMemcpyDtoD                        cuMemcpyDtoD
    #define gpuGetErrorName                      cuGetErrorName
    #define gpuGetErrorString                    cuGetErrorString
    #define GPUrtGetErrorName                    cudaGetErrorName
    #define GPUrtGetErrorString                  cudaGetErrorString
    #define gpuGetLastError                      cudaGetLastError
    #define gpuInit                              cuInit
    #define gpuDeviceGetCount                    cuDeviceGetCount
    #define gpuDeviceGet                         cuDeviceGet
    #define gpuCtxCreate                         cuCtxCreate
    #define gpuCtxDestroy                        cuCtxDestroy
    #define gpuDevicePrimaryCtxRetain            cuDevicePrimaryCtxRetain
    #define gpuDevicePrimaryCtxRelease           cuDevicePrimaryCtxRelease
    #define gpuCtxPushCurrent                    cuCtxPushCurrent
    #define gpuCtxPopCurrent                     cuCtxPopCurrent
    #define GPUrtLaunchKernel                    cudaLaunchKernel
    #define gpuMemPoolTrimTo                     cuMemPoolTrimTo
    #define gpuDeviceGetDefaultMemPool           cuDeviceGetDefaultMemPool
    #define gpuPointerGetAttribute               cuPointerGetAttribute

using GPUresult = CUresult;
using gpuError_t = cudaError_t;
using GPUdevice = CUdevice;
using GPUdevice_attribute = CUdevice_attribute;
using GPUcontext = CUcontext;
using GPUmemorytype = CUmemorytype;
using GPUstream_t = CUstream;
using GPUdeviceptr = CUdeviceptr;
using GPUmemoryPool = CUmemoryPool;
using GPUmemPoolProps = CUmemPoolProps;
using GPUmemAllocationType = CUmemAllocationType;
using GPUmemAllocationHandleType = CUmemAllocationHandleType;
using GPUmemLocationType = CUmemLocationType;
using GPUevent_t = CUevent;
using GPU_MEMCPY2D = CUDA_MEMCPY2D;

    // cuBLAS
    #define blasCreate          cublasCreate
    #define blasSetStream       cublasSetStream
    #define blasDestroy         cublasDestroy
    #define blasGetStatusName   cublasGetStatusName
    #define blasGetStatusString cublasGetStatusString

using blasStatus_t = cublasStatus_t;
using blasHandle_t = cublasHandle_t;

#elif KMM_USE_HIP

// HIP backend
// Experimental draft, not working
using half_type = __half;
using bfloat16_type = __hip_bfloat16;

    #define GPU_DEVICE_ATTRIBUTE_MAX                   HIP_DEVICE_ATTRIBUTE_MAX
    #define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X       HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y       HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z       HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X        HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y        HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z        HIP_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR \
        HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR \
        HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    #define GPU_MEMHOSTALLOC_PORTABLE            HIP_MEMHOSTALLOC_PORTABLE
    #define GPU_MEMHOSTALLOC_DEVICEMAP           HIP_MEMHOSTALLOC_DEVICEMAP
    #define GPU_SUCCESS                          HIP_SUCCESS
    #define GPU_ERROR_OUT_OF_MEMORY              HIP_ERROR_OUT_OF_MEMORY
    #define GPU_MEM_ALLOCATION_TYPE_PINNED       HIP_MEM_ALLOCATION_TYPE_PINNED
    #define GPU_MEM_HANDLE_TYPE_NONE             HIP_MEM_HANDLE_TYPE_NONE
    #define GPU_MEM_LOCATION_TYPE_DEVICE         HIP_MEM_LOCATION_TYPE_DEVICE
    #define GPU_ERROR_UNKNOWN                    HIP_ERROR_UNKNOWN
    #define GPU_STREAM_NON_BLOCKING              HIP_STREAM_NON_BLOCKING
    #define GPU_EVENT_WAIT_DEFAULT               HIP_EVENT_WAIT_DEFAULT
    #define GPU_ERROR_NOT_READY                  HIP_ERROR_NOT_READY
    #define GPU_EVENT_DISABLE_TIMING             HIP_EVENT_DISABLE_TIMING
    #define GPU_MEMORYTYPE_HOST                  HIP_MEMORYTYPE_HOST
    #define GPU_MEMORYTYPE_DEVICE                HIP_MEMORYTYPE_DEVICE
    #define GPU_ERROR_NO_DEVICE                  HIP_ERROR_NO_DEVICE
    #define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
    #define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    #define GPU_CTX_MAP_HOST                     HIP_CTX_MAP_HOST
    #define gpuCtxGetDevice                      hipCtxGetDevice
    #define gpuDeviceGetName                     hipDeviceGetName
    #define gpuDeviceGetAttribute                hipDeviceGetAttribute
    #define gpuMemGetInfo                        hipMemGetInfo
    #define gpuMemcpyHtoDAsync                   hipMemcpyHtoDAsync
    #define gpuMemcpyDtoHAsync                   hipMemcpyDtoHAsync
    #define gpuStreamSynchronize                 hipStreamSynchronize
    #define gpuMemsetD8Async                     hipMemsetD8Async
    #define gpuMemsetD16Async                    hipMemsetD16Async
    #define gpuMemsetD32Async                    hipMemsetD32Async
    #define gpuMemcpyAsync                       hipMemcpyAsync
    #define gpuMemHostAlloc                      hipMemHostAlloc
    #define gpuMemFreeHost                       hipMemFreeHost
    #define gpuMemAlloc                          hipMemAlloc
    #define gpuMemFree                           hipMemFree
    #define gpuMemPoolCreate                     hipMemPoolCreate
    #define gpuMemPoolDestroy                    hipMemPoolDestroy
    #define gpuMemAllocFromPoolAsync             hipMemAllocFromPoolAsync
    #define gpuMemFreeAsync                      hipMemFreeAsync
    #define gpuCtxGetStreamPriorityRange         hipCtxGetStreamPriorityRange
    #define gpuStreamCreateWithPriority          hipStreamCreateWithPriority
    #define gpuStreamQuery                       hipStreamQuery
    #define gpuStreamDestroy                     hipStreamDestroy
    #define gpuEventSynchronize                  hipEventSynchronize
    #define gpuEventRecord                       hipEventRecord
    #define gpuStreamWaitEvent                   hipStreamWaitEvent
    #define gpuEventQuery                        hipEventQuery
    #define gpuEventDestroy                      hipEventDestroy
    #define gpuEventCreate                       hipEventCreate
    #define gpuMemcpyHtoD                        hipMemcpyHtoD
    #define gpuMemcpy2DAsync                     hipMemcpy2DAsync
    #define gpuMemcpy2D                          hipMemcpy2D
    #define gpuMemcpyDtoH                        hipMemcpyDtoH
    #define gpuMemcpyDtoDAsync                   hipMemcpyDtoDAsync
    #define gpuMemcpyDtoD                        hipMemcpyDtoD
    #define gpuGetErrorName                      hipGetErrorName
    #define gpuGetErrorString                    hipGetErrorString
    #define GPUrtGetErrorName                    hipGetErrorName
    #define GPUrtGetErrorString                  hipGetErrorString
    #define gpuGetLastError                      hipGetLastError
    #define gpuInit                              hipInit
    #define gpuDeviceGetCount                    hipDeviceGetCount
    #define gpuDeviceGet                         hipDeviceGet
    #define gpuCtxCreate                         hipCtxCreate
    #define gpuCtxDestroy                        hipCtxDestroy
    #define gpuDevicePrimaryCtxRetain            hipDevicePrimaryCtxRetain
    #define gpuDevicePrimaryCtxRelease           hipDevicePrimaryCtxRelease
    #define gpuCtxPushCurrent                    hipCtxPushCurrent
    #define gpuCtxPopCurrent                     hipCtxPopCurrent
    #define GPUrtLaunchKernel                    hipLaunchKernel
    #define gpuMemPoolTrimTo                     hipMemPoolTrimTo
    #define gpuDeviceGetDefaultMemPool           hipDeviceGetDefaultMemPool
    #define gpuPointerGetAttribute               hipPointerGetAttribute

using GPUresult = hipError_t;
using gpuError_t = hipError_t;
using GPUdevice = hipError_t;
using GPUdevice_attribute = hipDeviceAttribute_t;
using GPUcontext = HIPcontext;
using GPUmemorytype = HIPmemorytype;
using GPUstream_t = HIPstream;
using GPUdeviceptr = HIPdeviceptr;
using GPUmemoryPool = HIPmemoryPool;
using GPUmemPoolProps = HIPmemPoolProps;
using GPUmemAllocationType = HIPmemAllocationType;
using GPUmemAllocationHandleType = HIPmemAllocationHandleType;
using GPUmemLocationType = HIPmemLocationType;
using GPUevent_t = HIPevent;
using GPU_MEMCPY2D = HIP_MEMCPY2D;

    // cuBLAS
    #define blasCreate                           rocblasCreate
    #define blasSetStream                        rocblasSetStream
    #define blasDestroy                          rocblasDestroy
    #define blasGetStatusName                    rocblasGetStatusName
    #define blasGetStatusString                  rocblasGetStatusString

using blasStatus_t = rocblasStatus_t;
using blasHandle_t = rocblasHandle_t;

#else
    #define GPU_DEVICE_ATTRIBUTE_MAX   1
    #define GPU_MEMHOSTALLOC_PORTABLE  0
    #define GPU_MEMHOSTALLOC_DEVICEMAP 0
    #define GPU_SUCCESS                0
    #define GPU_ERROR_OUT_OF_MEMORY    0
    #define GPU_ERROR_UNKNOWN          0
    #define GPU_ERROR_NOT_READY        0
    #define GPU_EVENT_DISABLE_TIMING   2
    #define GPU_ERROR_NO_DEVICE        100
    #define GPU_CTX_MAP_HOST           0x08

using half_type = unsigned char;
using bfloat16_type = char;
using size_t = std::size_t;

using GPUdevice = int;
class dim3 {
  public:
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}

    unsigned int x;
    unsigned int y;
    unsigned int z;
};
enum GPUmemAllocationType { GPU_MEM_ALLOCATION_TYPE_PINNED = 1 };
enum GPUmemAllocationHandleType { GPU_MEM_HANDLE_TYPE_NONE = 0 };
enum GPUmemLocationType { GPU_MEM_LOCATION_TYPE_DEVICE = 1 };
struct GPUmemLocation {
    int id;
    GPUmemLocationType type;
};
struct GPUmemPoolProps {
    GPUmemAllocationType allocType;
    GPUmemAllocationHandleType handleTypes;
    GPUmemLocation location;
    size_t maxSize;
    unsigned char reserved[54];
    unsigned short usage;
    void* win32SecurityAttributes;
};
using GPUcontext = int*;
using GPUstream_t = int*;
using GPUdeviceptr = size_t;
using GPUmemoryPool = int*;
using GPUevent_t = void*;
enum GPUmemorytype { GPU_MEMORYTYPE_HOST, GPU_MEMORYTYPE_DEVICE };
struct GPU_MEMCPY2D {
    size_t Height;
    size_t WidthInBytes;
    GPUdeviceptr dstDevice;
    void* dstHost;
    GPUmemorytype dstMemoryType;
    size_t dstPitch;
    size_t dstXInBytes;
    size_t dstY;
    GPUdeviceptr srcDevice;
    const void* srcHost;
    GPUmemorytype srcMemoryType;
    size_t srcPitch;
    size_t srcXInBytes;
    size_t srcY;
};
enum GPUresult {};
enum gpuError_t {};
using GPUdevice_attribute = int;
using GPUstream_flags = int;
using GPUevent_wait_flags = int;

    #define GPU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    GPUdevice_attribute(1)
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          GPUdevice_attribute(2)
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          GPUdevice_attribute(3)
    #define GPU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          GPUdevice_attribute(4)
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           GPUdevice_attribute(5)
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           GPUdevice_attribute(6)
    #define GPU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           GPUdevice_attribute(7)
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR GPUdevice_attribute(75)
    #define GPU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR GPUdevice_attribute(76)
    #define GPU_STREAM_NON_BLOCKING                       GPUstream_flags(1)
    #define GPU_EVENT_WAIT_DEFAULT                        GPUevent_wait_flags(0)

enum GPUpointer_attribute {};
    #define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE             GPUpointer_attribute(2)
    #define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL          GPUpointer_attribute(9)

// Dummy functions
GPUresult gpuCtxGetDevice(GPUdevice*);
GPUresult gpuDeviceGetName(char*, int, GPUdevice);
GPUresult gpuDeviceGetAttribute(int*, GPUdevice_attribute, GPUdevice);
GPUresult gpuMemGetInfo(size_t*, size_t*);
GPUresult gpuMemcpyHtoDAsync(GPUdeviceptr, const void*, size_t, GPUstream_t);
GPUresult gpuMemcpyDtoHAsync(void*, GPUdeviceptr, size_t, GPUstream_t);
GPUresult gpuStreamSynchronize(GPUstream_t);
GPUresult gpuMemsetD8Async(GPUdeviceptr, unsigned char, size_t, GPUstream_t);
GPUresult gpuMemsetD16Async(GPUdeviceptr, unsigned short, size_t, GPUstream_t);
GPUresult gpuMemsetD32Async(GPUdeviceptr, unsigned int, size_t, GPUstream_t);
GPUresult gpuMemcpyAsync(GPUdeviceptr, GPUdeviceptr, size_t, GPUstream_t);
GPUresult gpuMemHostAlloc(void**, size_t, unsigned int);
GPUresult gpuMemFreeHost(void*);
GPUresult gpuMemAlloc(GPUdeviceptr*, size_t);
GPUresult gpuMemFree(GPUdeviceptr);
GPUresult gpuMemPoolCreate(GPUmemoryPool*, const GPUmemPoolProps*);
GPUresult gpuMemPoolDestroy(GPUmemoryPool);
GPUresult gpuMemAllocFromPoolAsync(GPUdeviceptr*, size_t, GPUmemoryPool, GPUstream_t);
GPUresult gpuMemFreeAsync(GPUdeviceptr, GPUstream_t);
GPUresult gpuCtxGetStreamPriorityRange(int*, int*);
GPUresult gpuStreamCreateWithPriority(GPUstream_t*, unsigned int, int);
GPUresult gpuStreamQuery(GPUstream_t);
GPUresult gpuStreamDestroy(GPUstream_t);
GPUresult gpuEventSynchronize(GPUevent_t);
GPUresult gpuEventRecord(GPUevent_t, GPUstream_t);
GPUresult gpuStreamWaitEvent(GPUstream_t, GPUevent_t, unsigned int);
GPUresult gpuEventQuery(GPUevent_t);
GPUresult gpuEventDestroy(GPUevent_t);
GPUresult gpuEventCreate(GPUevent_t, unsigned int);
GPUresult gpuMemcpyHtoD(GPUdeviceptr, const void*, size_t);
GPUresult gpuMemcpy2DAsync(const GPU_MEMCPY2D*, GPUstream_t);
GPUresult gpuMemcpy2D(const GPU_MEMCPY2D*);
GPUresult gpuMemcpyDtoH(void*, GPUdeviceptr, size_t);
GPUresult gpuMemcpyDtoDAsync(GPUdeviceptr, GPUdeviceptr, size_t, GPUstream_t);
GPUresult gpuMemcpyDtoD(GPUdeviceptr, GPUdeviceptr, size_t);
GPUresult gpuGetErrorName(GPUresult, const char**);
GPUresult gpuGetErrorString(GPUresult, const char**);
const char* GPUrtGetErrorName(gpuError_t);
const char* GPUrtGetErrorString(gpuError_t);
gpuError_t gpuGetLastError(void);
GPUresult gpuInit(unsigned int);
GPUresult gpuDeviceGetCount(int*);
GPUresult gpuDeviceGet(GPUdevice*, int);
GPUresult gpuPointerGetAttribute(void*, GPUpointer_attribute, GPUdeviceptr);
GPUresult gpuCtxCreate(GPUcontext*, unsigned int, GPUdevice);
GPUresult gpuCtxDestroy(GPUcontext);
GPUresult gpuDevicePrimaryCtxRetain(GPUcontext*, GPUdevice);
GPUresult gpuDevicePrimaryCtxRelease(GPUdevice);
GPUresult gpuCtxPushCurrent(GPUcontext);
GPUresult gpuCtxPopCurrent(GPUcontext*);
gpuError_t GPUrtLaunchKernel(const void*, dim3, dim3, void**, size_t, GPUstream_t);
GPUresult gpuMemPoolTrimTo(GPUmemoryPool, size_t);
GPUresult gpuDeviceGetDefaultMemPool(GPUmemoryPool*, GPUdevice);

// Atomics
template<typename T>
T atomicAnd(T* input, T output) {
    return 0;
};
template<typename T>
T atomicOr(T* input, T output) {
    return 0;
};
template<typename T>
T atomicAdd(T* input, T output) {
    return 0;
};
template<typename T>
T atomicMin(T* input, T output) {
    return 0;
};
template<typename T>
T atomicMax(T* input, T output) {
    return 0;
};

// Dummy BLAS
using blasHandle_t = void*;
enum blasStatus_t {};
blasStatus_t blasCreate(blasHandle_t);
blasStatus_t blasSetStream(blasHandle_t, GPUstream_t);
blasStatus_t blasDestroy(blasHandle_t);
const char* blasGetStatusName(blasStatus_t);
const char* blasGetStatusString(blasStatus_t);

#endif

}  // namespace kmm