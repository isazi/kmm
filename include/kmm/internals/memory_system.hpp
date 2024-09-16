#include "kmm/internals/allocator/base.hpp"
#include "kmm/internals/cuda_stream_manager.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class MemorySystem {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemorySystem)

  public:
    MemorySystem(
        std::shared_ptr<CudaStreamManager> streams,
        std::vector<CudaContextHandle> device_contexts,
        std::unique_ptr<MemoryAllocator> host_mem,
        std::vector<std::unique_ptr<MemoryAllocator>> device_mem);

    ~MemorySystem();

    void make_progress();

    bool allocate(MemoryId memory_id, size_t nbytes, void*& ptr_out, CudaEventSet& deps_out);

    void deallocate(MemoryId memory_id, void* ptr, size_t nbytes, CudaEventSet deps = {});

    CudaEvent copy_host_to_device(
        DeviceId device_id,
        const void* src_addr,
        CUdeviceptr dst_addr,
        size_t nbytes,
        CudaEventSet deps);

    CudaEvent copy_device_to_host(
        DeviceId device_id,
        CUdeviceptr src_addr,
        void* dst_addr,
        size_t nbytes,
        CudaEventSet deps);

  private:
    struct Device;

    std::shared_ptr<CudaStreamManager> m_streams;
    std::unique_ptr<MemoryAllocator> m_host;
    std::vector<std::unique_ptr<Device>> m_devices;
};
}  // namespace kmm