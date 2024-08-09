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

    bool acquire_allocation_async(const std::shared_ptr<Request>& req, CudaStreamId stream);
    void acquire_access_async(const std::shared_ptr<Request>& req, CudaStreamId stream);

    void* get_host_pointer(const std::shared_ptr<Request>& req);
    CUdeviceptr get_device_pointer(const std::shared_ptr<Request>& req, DeviceId device_id);

    void trim_device_memory(CudaStreamId stream, DeviceId device_id, size_t num_bytes_max = 0);

  private:
    void insert_into_lru(DeviceId device_id, BufferMeta& buffer, bool hint_last_access);
    void remove_from_lru(DeviceId device_id, BufferMeta& buffer);

    bool try_free_device_memory(CudaStreamId stream, DeviceId device_id);
    bool try_allocate_device_async(CudaStreamId stream, DeviceId device_id, BufferMeta& buffer);
    void deallocate_device_async(CudaStreamId stream, DeviceId device_id, BufferMeta& buffer);

    void allocate_host(BufferMeta& buffer);

    void copy_h2d(CudaStreamId stream, DeviceId device_id, BufferMeta& buffer);
    void copy_d2h(CudaStreamId stream, DeviceId device_id, BufferMeta& buffer);

    void update_data_host_async(TransactionId trans_id, BufferId buffer_id, CudaStreamId stream);
    void update_data_device_async(
        TransactionId trans_id,
        DeviceId device_id,
        BufferId buffer_id,
        CudaStreamId stream);

    void lock_allocation_host(TransactionId trans_id, BufferId buffer_id);
    bool lock_allocation_device_async(
        TransactionId trans_id,
        DeviceId device_id,
        BufferId buffer_id,
        CudaStreamId stream);

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