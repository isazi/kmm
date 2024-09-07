#include "kmm/internals/memory_allocator.hpp"

namespace kmm {

struct MemoryAllocatorImpl::Device {
    CudaContextHandle context;
    CUmemoryPool memory_pool;
    size_t bytes_in_use = 0;
    size_t bytes_limit = std::numeric_limits<size_t>::max();

    CudaStream h2d_stream;
    CudaStream d2h_stream;
    CudaStream alloc_stream;
    CudaStream dealloc_stream;

    Device(MemoryDeviceInfo info) :
        context(info.context),
        h2d_stream(info.h2d_stream),
        d2h_stream(info.d2h_stream),
        alloc_stream(info.alloc_stream),
        dealloc_stream(info.dealloc_stream),
        bytes_limit(info.num_bytes_limit) {
        CudaContextGuard guard {context};

        CUdevice device;
        KMM_CUDA_CHECK(cuCtxGetDevice(&device));

        size_t total_bytes;
        size_t free_bytes;
        KMM_CUDA_CHECK(cuMemGetInfo(&free_bytes, &total_bytes));

        bytes_limit = std::min(
            info.num_bytes_limit,
            saturating_sub(total_bytes, info.num_bytes_keep_available));

        CUmemPoolProps props;
        ::bzero(&props, sizeof(CUmemPoolProps));
        props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
        props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = device;

        KMM_CUDA_CHECK(cuMemPoolCreate(&memory_pool, &props));
    }

    ~Device() {
        CudaContextGuard guard {context};
        KMM_ASSERT(bytes_in_use == 0);
        KMM_CUDA_CHECK(cuMemPoolDestroy(memory_pool));
    }
};

struct MemoryAllocatorImpl::DeferredDeletion {
    CudaEventSet dependencies;
    void* data;
};

MemoryAllocatorImpl::MemoryAllocatorImpl(
    std::shared_ptr<CudaStreamManager> streams,
    std::vector<MemoryDeviceInfo> devices) :
    m_streams(streams) {}

MemoryAllocatorImpl::~MemoryAllocatorImpl() = default;

bool MemoryAllocatorImpl::allocate_device(
    DeviceId device_id,
    size_t nbytes,
    CUdeviceptr& ptr_out,
    CudaEvent& event_out) {
    auto& device = m_devices.at(device_id.get());

    if (device.bytes_limit - device.bytes_in_use < nbytes) {
        return false;
    }

    device.bytes_in_use += device.bytes_in_use;

    CUresult result;

    event_out = m_streams->with_stream(device.alloc_stream, [&](auto stream) {
        result = cuMemAllocFromPoolAsync(&ptr_out, nbytes, device.memory_pool, stream);
    });

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return false;
    }

    if (result != CUDA_SUCCESS) {
        throw CudaDriverException("`cuMemAllocFromPoolAsync` failed", result);
    }

    return true;
}

void MemoryAllocatorImpl::deallocate_device(
    DeviceId device_id,
    CUdeviceptr ptr,
    CudaEventSet deps) {
    auto& device = m_devices.at(device_id.get());

    m_streams->with_stream(device.dealloc_stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemFreeAsync(ptr, stream));
    });
}

void* MemoryAllocatorImpl::allocate_host(size_t nbytes) {
    void* addr;

    CudaContextGuard guard {m_streams->get(DeviceId(0))};
    KMM_CUDA_CHECK(cuMemHostAlloc(
        &addr,  //
        nbytes,
        CU_MEMHOSTALLOC_PORTABLE));

    return addr;
}

void MemoryAllocatorImpl::deallocate_host(void* addr, CudaEventSet deps) {
    // All events need to be ready before memory can be deallocated.
    if (m_streams->is_ready(deps)) {
        // If the events finished, we can deallocate now immediately
        CudaContextGuard guard {m_streams->get(DeviceId(0))};
        KMM_CUDA_CHECK(cuMemFreeHost(addr));
    } else {
        // Some events are still pending, add to deletion queue to defer it to a later moment in time.
        m_deferred_deletions.push_back({std::move(deps), addr});
    }
}

void MemoryAllocatorImpl::make_progress() {
    while (!m_deferred_deletions.empty()) {
        auto& p = m_deferred_deletions.back();

        if (!m_streams->is_ready(p.dependencies)) {
            break;
        }

        CudaContextGuard guard {m_streams->get(DeviceId(0))};
        KMM_CUDA_CHECK(cuMemFreeHost(p.data));
        m_deferred_deletions.pop_back();
    }
}

CudaEvent MemoryAllocatorImpl::copy_host_to_device(
    DeviceId device_id,
    const void* src_addr,
    CUdeviceptr dst_addr,
    size_t nbytes,
    CudaEventSet deps) {
    auto& device = m_devices.at(device_id);

    return m_streams->with_stream(device.h2d_stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemcpyHtoDAsync(dst_addr, src_addr, nbytes, stream));
    });
}

CudaEvent MemoryAllocatorImpl::copy_device_to_host(
    DeviceId device_id,
    CUdeviceptr src_addr,
    void* dst_addr,
    size_t nbytes,
    CudaEventSet deps) {
    auto& device = m_devices.at(device_id);

    return m_streams->with_stream(device.d2h_stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemcpyDtoHAsync(dst_addr, src_addr, nbytes, stream));
    });
}

}  // namespace kmm