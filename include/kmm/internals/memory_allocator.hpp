#include "kmm/internals/cuda_stream_manager.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class MemoryAllocator {
  public:
    virtual ~MemoryAllocator() = default;

    virtual void make_progress() {}

    virtual void* allocate_host(size_t nbytes) = 0;

    virtual void deallocate_host(void* addr, CudaEventSet deps = {}) = 0;

    virtual bool allocate_device(
        DeviceId device_id,
        size_t nbytes,
        CUdeviceptr& ptr_out,
        CudaEvent& event_out) = 0;

    virtual void deallocate_device(DeviceId device_id, CUdeviceptr ptr, CudaEventSet deps) = 0;

    virtual CudaEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        CUdeviceptr dst_addr,
        size_t nbytes,
        CudaEventSet deps) = 0;

    virtual CudaEvent copy_device_to_host(
        DeviceId device_id,
        CUdeviceptr src_addr,
        void* dst_addr,
        size_t nbytes,
        CudaEventSet deps) = 0;
};

struct MemoryDeviceInfo {
    // Maximum number of bytes that can be allocated
    size_t num_bytes_limit = std::numeric_limits<size_t>::max();

    // The number of bytes that should remain available on the device for other CUDA frameworks
    size_t num_bytes_keep_available = 100'000'000;
};

class MemoryAllocatorImpl: public MemoryAllocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemoryAllocatorImpl)

  public:
    MemoryAllocatorImpl(
        std::shared_ptr<CudaStreamManager> streams,
        std::vector<MemoryDeviceInfo> devices);

    ~MemoryAllocatorImpl();

    void make_progress() final;

    void* allocate_host(size_t nbytes) final;

    void deallocate_host(void* addr, CudaEventSet deps = {}) final;

    bool allocate_device(
        DeviceId device_id,
        size_t nbytes,
        CUdeviceptr& ptr_out,
        CudaEvent& event_out) final;

    void deallocate_device(DeviceId device_id, CUdeviceptr ptr, CudaEventSet deps = {}) final;

    CudaEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        CUdeviceptr dst_addr,
        size_t nbytes,
        CudaEventSet deps) final;

    CudaEvent copy_device_to_host(
        DeviceId device_id,
        CUdeviceptr src_addr,
        void* dst_addr,
        size_t nbytes,
        CudaEventSet deps) final;

  private:
    struct Device;
    struct DeferredDeletion;

    std::shared_ptr<CudaStreamManager> m_streams;
    std::vector<Device> m_devices;
    std::vector<DeferredDeletion> m_deferred_deletions;
};
}  // namespace kmm