#include "kmm/allocators/base.hpp"
#include "kmm/internals/device_stream_manager.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class MemorySystem {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemorySystem)

  public:
    MemorySystem(
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::vector<CudaContextHandle> device_contexts,
        std::unique_ptr<AsyncAllocator> host_mem,
        std::vector<std::unique_ptr<AsyncAllocator>> device_mem
    );

    ~MemorySystem();

    void make_progress();

    void trim_host(size_t bytes_remaining = 0);
    void trim_device(size_t bytes_remaining = 0);

    bool allocate_host(size_t nbytes, void** ptr_out, DeviceEventSet* deps_out);
    void deallocate_host(void* ptr, size_t nbytes, DeviceEventSet deps = {});

    bool allocate_device(
        DeviceId device_id,
        size_t nbytes,
        CUdeviceptr* ptr_out,
        DeviceEventSet* deps_out
    );

    void deallocate_device(
        DeviceId device_id,
        CUdeviceptr ptr,
        size_t nbytes,
        DeviceEventSet deps = {}
    );

    DeviceEvent fill_host(
        void* dst_addr,
        size_t nbytes,
        const std::vector<uint8_t>& fill_pattern,
        DeviceEventSet deps = {}
    );

    DeviceEvent fill_device(
        DeviceId device_id,
        CUdeviceptr dst_addr,
        size_t nbytes,
        const std::vector<uint8_t>& fill_pattern,
        DeviceEventSet deps = {}
    );

    DeviceEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        CUdeviceptr dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    );

    DeviceEvent copy_device_to_host(
        DeviceId device_id,
        CUdeviceptr src_addr,
        void* dst_addr,
        size_t nbytes,
        DeviceEventSet deps
    );

  private:
    struct Device;

    std::shared_ptr<DeviceStreamManager> m_streams;
    std::unique_ptr<AsyncAllocator> m_host;
    std::unique_ptr<Device> m_devices[MAX_DEVICES];
};
}  // namespace kmm