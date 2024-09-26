#include "kmm/internals/allocator/gpu.hpp"

namespace kmm {

PinnedMemoryAllocator::PinnedMemoryAllocator(
    GPUContextHandle context,
    std::shared_ptr<GPUStreamManager> streams,
    size_t max_bytes) :
    DirectMemoryAllocator(streams, max_bytes),
    m_context(context) {
}

bool PinnedMemoryAllocator::allocate_impl(size_t nbytes, void*& addr_out) {
    GPUContextGuard guard{m_context};
    auto result = gpuMemHostAlloc(&addr_out, nbytes, GPU_MEMHOSTALLOC_PORTABLE | GPU_MEMHOSTALLOC_DEVICEMAP);

    if (result == GPU_SUCCESS) {
        return true;
    } else if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw GPUDriverException("error when calling `gpuMemHostAlloc`", result);
    }
}

void PinnedMemoryAllocator::deallocate_impl(void* addr, size_t nbytes) {
    GPUContextGuard guard{m_context};
    KMM_GPU_CHECK(gpuMemFreeHost(addr));
}

DeviceMemoryAllocator::DeviceMemoryAllocator(
    GPUContextHandle context,
    std::shared_ptr<GPUStreamManager> streams,
    size_t max_bytes) :
    DirectMemoryAllocator(streams, max_bytes),
    m_context(context) {
}

bool DeviceMemoryAllocator::allocate_impl(size_t nbytes, void*& addr_out) {
    GPUContextGuard guard{m_context};
    GPUdeviceptr ptr;
    auto result = gpuMemAlloc(&ptr, nbytes);

    if (result == GPU_SUCCESS) {
        addr_out = (void*)ptr;
        return true;
    } else if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw GPUDriverException("error when calling `cuMemAlloc`", result);
    }
}
void DeviceMemoryAllocator::deallocate_impl(void* addr, size_t nbytes) {
    GPUContextGuard guard{m_context};
    KMM_GPU_CHECK(gpuMemFree(GPUdeviceptr(addr)));
}


DevicePoolAllocator::DevicePoolAllocator(GPUContextHandle context, std::shared_ptr<GPUStreamManager> streams, size_t max_bytes):
    m_context(context),
    m_streams(streams),
    m_alloc_stream(streams->create_stream(context)),
    m_dealloc_stream(streams->create_stream(context)),
    m_bytes_limit(max_bytes)
{
    GPUContextGuard guard{m_context};

    GPUdevice device;
    KMM_GPU_CHECK(gpuCtxGetDevice(&device));

    GPUmemPoolProps props;
    ::bzero(&props, sizeof(GPUmemPoolProps));

    props.allocType = GPU_MEM_ALLOCATION_TYPE_PINNED;
    props.handleTypes = GPU_MEM_HANDLE_TYPE_NONE;
    props.location.type = GPU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = device;

    KMM_GPU_CHECK(gpuMemPoolCreate(&m_pool, &props));
}

DevicePoolAllocator::~DevicePoolAllocator() {
    for (auto d: m_pending_deallocs) {
        m_bytes_in_use -= d.nbytes;
        m_streams->wait_until_ready(d.event);
    }

    KMM_ASSERT(m_bytes_in_use == 0);

    GPUContextGuard guard{m_context};
    KMM_GPU_CHECK(gpuMemPoolDestroy(m_pool));
}

bool DevicePoolAllocator::allocate(size_t nbytes, void*& addr_out, GPUEventSet& deps_out) {
    while (true) {
        if (m_pending_deallocs.empty()) {
            break;
        }

        auto d = m_pending_deallocs.front();

        if (!m_streams->is_ready(d.event)) {
            if (nbytes <= m_bytes_limit - m_bytes_in_use) {
                break;
            }

            m_streams->wait_for_event(m_alloc_stream, d.event);
        }

        m_bytes_in_use -= d.nbytes;
        m_pending_deallocs.pop_front();
    }

    if (nbytes > m_bytes_limit - m_bytes_in_use) {
        return false;
    }

    GPUContextGuard guard{m_context};
    GPUdeviceptr device_ptr;
    GPUresult result = GPUresult(GPU_ERROR_UNKNOWN);

    auto event = m_streams->with_stream(m_alloc_stream, [&](auto stream) {
        result = gpuMemAllocFromPoolAsync(&device_ptr, nbytes, m_pool, stream);
    });

    if (result == GPU_SUCCESS) {
        m_bytes_in_use += nbytes;
        deps_out.insert(event);
        addr_out = (void*)device_ptr;
        return true;
    } else if (result == GPU_ERROR_OUT_OF_MEMORY) {
        return false;
    } else {
        throw GPUDriverException("error while calling `gpuMemAllocFromPoolAsync`", result);
    }
}

void DevicePoolAllocator::deallocate(void* addr, size_t nbytes, GPUEventSet deps) {
    GPUdeviceptr device_ptr = (GPUdeviceptr)addr;

    auto event = m_streams->with_stream(m_dealloc_stream, deps, [&](auto stream) {
        KMM_GPU_CHECK(gpuMemFreeAsync(device_ptr, stream));
    });

    m_pending_deallocs.push_back({
        .addr = addr,
        .nbytes = nbytes,
        .event = event
    });
}}