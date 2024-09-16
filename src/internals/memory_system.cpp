#include <unordered_map>

#include "spdlog/spdlog.h"

#include "kmm/internals/memory_system.hpp"

namespace kmm {

struct MemorySystem::Device {
    KMM_NOT_COPYABLE(Device)

  public:
    CudaContextHandle context;
    std::unique_ptr<MemoryAllocator> allocator;

    CudaStream alloc_stream;
    CudaStream dealloc_stream;
    CudaStream h2d_stream;
    CudaStream d2h_stream;
    CudaStream h2d_hi_stream;  // high priority stream
    CudaStream d2h_hi_stream;  // high priority stream

    Device(
        CudaContextHandle context,
        std::unique_ptr<MemoryAllocator> allocator,
        CudaStreamManager& streams) :
        context(context),
        allocator(std::move(allocator)),
        h2d_stream(streams.create_stream(context, false)),
        d2h_stream(streams.create_stream(context, false)),
        h2d_hi_stream(streams.create_stream(context, true)),
        d2h_hi_stream(streams.create_stream(context, true)) {}
};

MemorySystem::MemorySystem(
    std::shared_ptr<CudaStreamManager> streams,
    std::vector<CudaContextHandle> device_contexts,
    std::unique_ptr<MemoryAllocator> host_mem,
    std::vector<std::unique_ptr<MemoryAllocator>> device_mems) :
    m_streams(streams),
    m_host(std::move(host_mem))

{
    KMM_ASSERT(device_contexts.size() == device_mems.size());

    for (size_t i = 0; i < device_contexts.size(); i++) {
        m_devices.push_back(
            std::make_unique<Device>(device_contexts[i], std::move(device_mems[i]), *streams));
    }
}

MemorySystem::~MemorySystem() {}

void MemorySystem::make_progress() {
    m_host->make_progress();

    for (const auto& device : m_devices) {
        device->allocator->make_progress();
    }
}

bool MemorySystem::allocate(
    MemoryId memory_id,
    size_t nbytes,
    void*& ptr_out,
    CudaEventSet& deps_out) {
    if (memory_id.is_device()) {
        return m_devices.at(memory_id.as_device())->allocator->allocate(nbytes, ptr_out, deps_out);
    } else {
        return m_host->allocate(nbytes, ptr_out, deps_out);
    }
}

void MemorySystem::deallocate(MemoryId memory_id, void* ptr, size_t nbytes, CudaEventSet deps) {
    if (memory_id.is_device()) {
        return m_devices.at(memory_id.as_device())
            ->allocator->deallocate(ptr, nbytes, std::move(deps));
    } else {
        return m_host->deallocate(ptr, nbytes, std::move(deps));
    }
}

// Copies smaller than this threshold are put onto a high priority stream. This can improve
// performance since small copy jobs (like copying a single number) are prioritized over large
// slow copy jobs of several gigabytes.
static constexpr size_t HIGH_PRIORITY_THRESHOLD = 1024L * 1024;

CudaEvent MemorySystem::copy_host_to_device(
    DeviceId device_id,
    const void* src_addr,
    CUdeviceptr dst_addr,
    size_t nbytes,
    CudaEventSet deps) {
    auto& device = *m_devices.at(device_id);
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.h2d_hi_stream : device.h2d_stream;

    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemcpyHtoDAsync(dst_addr, src_addr, nbytes, stream));
    });
}

CudaEvent MemorySystem::copy_device_to_host(
    DeviceId device_id,
    CUdeviceptr src_addr,
    void* dst_addr,
    size_t nbytes,
    CudaEventSet deps) {
    auto& device = *m_devices.at(device_id);
    auto stream = nbytes <= HIGH_PRIORITY_THRESHOLD ? device.d2h_hi_stream : device.d2h_stream;

    return m_streams->with_stream(stream, deps, [&](auto stream) {
        KMM_CUDA_CHECK(cuMemcpyDtoHAsync(dst_addr, src_addr, nbytes, stream));
    });
}

}  // namespace kmm