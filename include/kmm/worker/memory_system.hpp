#include "kmm/allocators/base.hpp"
#include "kmm/core/macros.hpp"
#include "kmm/worker/device_stream_manager.hpp"

namespace kmm {

class MemorySystemBase {
  public:
    virtual ~MemorySystemBase() = default;

    virtual bool allocate_host(size_t nbytes, void** ptr_out, DeviceEventSet* deps_out) = 0;
    virtual void deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps) = 0;

    virtual bool allocate_device(
        DeviceId device_id,
        size_t nbytes,
        GPUdeviceptr* ptr_out,
        DeviceEventSet* deps_out
    ) = 0;

    virtual void deallocate_device(
        DeviceId device_id,
        GPUdeviceptr ptr,
        size_t nbytes,
        DeviceEventSet deps
    ) = 0;

    virtual DeviceEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        GPUdeviceptr dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) = 0;

    virtual DeviceEvent copy_device_to_host(
        DeviceId device_id,
        GPUdeviceptr src_addr,
        void* dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) = 0;
};

class MemorySystem: public MemorySystemBase {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemorySystem)

  public:
    MemorySystem(
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::vector<GPUContextHandle> device_contexts,
        std::unique_ptr<AsyncAllocator> host_mem,
        std::vector<std::unique_ptr<AsyncAllocator>> device_mem
    );

    ~MemorySystem();

    void make_progress();

    void trim_host(size_t bytes_remaining = 0);
    void trim_device(size_t bytes_remaining = 0);

    bool allocate_host(size_t nbytes, void** ptr_out, DeviceEventSet* deps_out) final;
    void deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps) final;

    bool allocate_device(
        DeviceId device_id,
        size_t nbytes,
        GPUdeviceptr* ptr_out,
        DeviceEventSet* deps_out
    ) final;

    void deallocate_device(DeviceId device_id, GPUdeviceptr ptr, size_t nbytes, DeviceEventSet deps)
        final;

    DeviceEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        GPUdeviceptr dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) final;

    DeviceEvent copy_device_to_host(
        DeviceId device_id,
        GPUdeviceptr src_addr,
        void* dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    ) final;

  private:
    struct Device;

    std::shared_ptr<DeviceStreamManager> m_streams;
    std::unique_ptr<AsyncAllocator> m_host;
    std::unique_ptr<Device> m_devices[MAX_DEVICES];
};
}  // namespace kmm