#pragma once

#include <unordered_map>

#include "cuda_stream_manager.hpp"

#include "kmm/core/identifiers.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

static constexpr size_t MAX_DEVICES = 4;
using TransactionId = uint64_t;

struct MemoryDeviceInfo {
    CudaContextHandle context;

    // Maximum number of bytes that can be allocated
    size_t num_bytes_limit = std::numeric_limits<size_t>::max();

    // The number of bytes that should remain available on the device for other CUDA frameworks
    size_t num_bytes_keep_available = 100'000'000;

    CudaStreamId alloc_stream;
    CudaStreamId h2d_stream;
    CudaStreamId d2h_stream;
};

class MemoryManager {
  public:
    struct BufferMeta;
    struct DeviceMeta;
    struct Request;

    MemoryManager(
        std::shared_ptr<CudaStreamManager> stream_manager,
        std::vector<MemoryDeviceInfo> devices);
    ~MemoryManager();

    std::shared_ptr<Request> create_request(
        TransactionId transaction_id,
        MemoryId device_id,
        BufferId buffer_id,
        AccessMode mode);
    void delete_request(const std::shared_ptr<Request>& req);
    void delete_request_async(const std::shared_ptr<Request>& req, CudaEvent event);

    BufferId create_buffer(BufferLayout layout);
    void delete_buffer(BufferId);

    std::optional<CudaEvent> acquire_allocation_async(const std::shared_ptr<Request>& req);
    CudaEventSet acquire_access_async(const std::shared_ptr<Request>& req);

    void* get_host_pointer(const std::shared_ptr<Request>& req);
    CUdeviceptr get_device_pointer(const std::shared_ptr<Request>& req, DeviceId device_id);

    void trim_device_memory(DeviceId device_id, size_t num_bytes_max = 0);

  private:
    void insert_into_lru(DeviceId device_id, BufferMeta& buffer, bool hint_last_access);
    void remove_from_lru(DeviceId device_id, BufferMeta& buffer);

    bool try_free_device_memory(DeviceId device_id);
    bool try_allocate_device_async(DeviceId device_id, BufferMeta& buffer);
    void deallocate_device_async(DeviceId device_id, BufferMeta& buffer);

    void allocate_host(BufferMeta& buffer);

    CudaEvent copy_h2d(DeviceId device_id, BufferMeta& buffer);
    CudaEvent copy_d2h(DeviceId device_id, BufferMeta& buffer);

    CudaEventSet update_data_host_async(TransactionId trans_id, BufferId buffer_id);
    CudaEventSet update_data_device_async(
        TransactionId trans_id,
        DeviceId device_id,
        BufferId buffer_id);

    void lock_allocation_host(TransactionId trans_id, BufferId buffer_id);
    std::optional<CudaEvent> lock_allocation_device_async(
        TransactionId trans_id,
        DeviceId device_id,
        BufferId buffer_id);

    void unlock_allocation_host(
        TransactionId trans_id,
        BufferId buffer_id,
        AccessMode mode,
        CudaEvent event);
    void unlock_allocation_device_async(
        TransactionId trans_id,
        DeviceId device_id,
        BufferId buffer_id,
        AccessMode mode,
        CudaEvent event,
        bool hint_last_access = false);

    uint64_t next_buffer_id = 1;
    std::unordered_map<BufferId, std::unique_ptr<BufferMeta>> m_buffers;
    std::shared_ptr<CudaStreamManager> m_streams;
    std::unique_ptr<DeviceMeta> m_devices[MAX_DEVICES];
};

using MemoryRequest = std::shared_ptr<MemoryManager::Request>;

}  // namespace kmm