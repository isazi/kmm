
#include "kmm/internals/backends.hpp"

namespace kmm {

dim3::dim3(int x, int y, int z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

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

GPUresult gpuMemcpyHtoDAsync(GPUdeviceptr, const void*, size_t, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoHAsync(void*, GPUdeviceptr, size_t, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamSynchronize(stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD8Async(GPUdeviceptr, unsigned char, size_t, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD16Async(GPUdeviceptr, unsigned short, size_t, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemsetD32Async(GPUdeviceptr, unsigned int, size_t, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyAsync(GPUdeviceptr, GPUdeviceptr, size_t, stream_t) {
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

GPUresult gpuMemAllocFromPoolAsync(GPUdeviceptr*, size_t, GPUmemoryPool, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemFreeAsync(GPUdeviceptr, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuCtxGetStreamPriorityRange(int* , int*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamCreateWithPriority(stream_t*, unsigned int, int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamQuery(stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult cuStreamDestroy(stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventSynchronize(event_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventRecord(event_t , stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuStreamWaitEvent(stream_t , event_t , unsigned int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventQuery(event_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventDestroy(event_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuEventCreate(event_t , unsigned int) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyHtoD(GPUdeviceptr, const void*, size_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpy2DAsync(const GPU_MEMCPY2D*, stream_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpy2D(const GPU_MEMCPY2D*) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoH(void*, GPUdeviceptr, size_t) {
    return GPUresult(GPU_ERROR_UNKNOWN);
}

GPUresult gpuMemcpyDtoDAsync(GPUdeviceptr, GPUdeviceptr, size_t, stream_t) {
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

GPUresult gpuInit (unsigned int) {
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
gpuError_t GPUrtLaunchKernel(const void*, dim3, dim3, void**, size_t, stream_t) {
    return gpuError_t(GPU_ERROR_UNKNOWN);
}

blasStatus_t blasCreate(blasHandle_t) {
    return blasStatus_t(1);
}

blasStatus_t blasSetStream(blasHandle_t, stream_t) {
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

} // kmm