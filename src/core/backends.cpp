
#include "kmm/core/backends.hpp"
#include "kmm/memops/types.hpp"

namespace kmm {

#if !defined(KMM_USE_CUDA) && !defined(KMM_USE_HIP)

GPUresult gpuCtxGetDevice(GPUdevice*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGetName(char*, int, GPUdevice) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGetAttribute(int*, GPUdevice_attribute, GPUdevice) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemGetInfo(size_t*, size_t*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyHtoDAsync(GPUdeviceptr, const void*, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoHAsync(void*, GPUdeviceptr, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamSynchronize(GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD8Async(GPUdeviceptr, unsigned char, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD16Async(GPUdeviceptr, unsigned short, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD32Async(GPUdeviceptr, unsigned int, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyAsync(GPUdeviceptr, GPUdeviceptr, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemHostAlloc(void**, size_t, unsigned int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemFreeHost(void*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemAlloc(GPUdeviceptr*, size_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemFree(GPUdeviceptr) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemPoolCreate(GPUmemoryPool*, const GPUmemPoolProps*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemPoolDestroy(GPUmemoryPool) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemAllocFromPoolAsync(GPUdeviceptr*, size_t, GPUmemoryPool, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemFreeAsync(GPUdeviceptr, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxGetStreamPriorityRange(int*, int*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamCreateWithPriority(GPUstream_t*, unsigned int, int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamQuery(GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamDestroy(GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult cuStreamDestroy(GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventSynchronize(GPUevent_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventRecord(GPUevent_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamWaitEvent(GPUstream_t, GPUevent_t, unsigned int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventQuery(GPUevent_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventDestroy(GPUevent_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventCreate(GPUevent_t, unsigned int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyHtoD(GPUdeviceptr, const void*, size_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpy2DAsync(const GPU_MEMCPY2D*, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpy2D(const GPU_MEMCPY2D*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoH(void*, GPUdeviceptr, size_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoDAsync(GPUdeviceptr, GPUdeviceptr, size_t, GPUstream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoD(GPUdeviceptr, GPUdeviceptr, size_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuGetErrorName(GPUresult, const char**) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuGetErrorString(GPUresult, const char**) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

const char* GPUrtGetErrorName(gpuError_t) {
    return "";
}

const char* GPUrtGetErrorString(gpuError_t) {
    return "";
}

gpuError_t gpuGetLastError(void) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuInit(unsigned int) {
    return GPUresult(GPU_ERROR_NO_DEVICE);
}

GPUresult gpuDeviceGetCount(int*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGet(GPUdevice*, int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuPointerGetAttribute(void*, GPUpointer_attribute, GPUdeviceptr) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxCreate(GPUcontext*, unsigned int, GPUdevice) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxDestroy(GPUcontext) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDevicePrimaryCtxRetain(GPUcontext*, GPUdevice) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDevicePrimaryCtxRelease(GPUdevice) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxPushCurrent(GPUcontext) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxPopCurrent(GPUcontext*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}
gpuError_t GPUrtLaunchKernel(const void*, dim3, dim3, void**, size_t, GPUstream_t) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemPoolTrimTo(GPUmemoryPool, size_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuDeviceGetDefaultMemPool(GPUmemoryPool*, GPUdevice) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

blasStatus_t blasCreate(blasHandle_t) {
    return blasStatus_t(1);
}

blasStatus_t blasSetStream(blasHandle_t, GPUstream_t) {
    return blasStatus_t(1);
}

blasStatus_t blasDestroy(blasHandle_t) {
    return blasStatus_t(1);
}

const char* blasGetStatusName(blasStatus_t) {
    return "";
}

const char* blasGetStatusString(blasStatus_t) {
    return "";
}

void execute_gpu_fill_async(
    GPUstream_t stream,
    GPUdeviceptr dst_buffer,
    size_t nbytes,
    const void* pattern,
    size_t pattern_nbytes
) {
    return;
}

void execute_gpu_reduction_async(
    GPUstream_t stream,
    GPUdeviceptr src_buffer,
    GPUdeviceptr dst_buffer,
    ReductionDef reduction
) {
    return;
}

void execute_gpu_fill_async(GPUstream_t stream, GPUdeviceptr dst_buffer, const FillDef& fill) {
    return;
}

#endif

}  // namespace kmm