#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "kmm/core/buffer.hpp"
#include "kmm/internals/cuda_stream_manager.hpp"

namespace kmm {

using TransactionId = uint64_t;

struct MemoryDeviceInfo {
    CudaContextHandle context;

    // Maximum number of bytes that can be allocated
    size_t num_bytes_limit = std::numeric_limits<size_t>::max();

    // The number of bytes that should remain available on the device for other CUDA frameworks
    size_t num_bytes_keep_available = 100'000'000;

    CudaStream alloc_stream;
    CudaStream dealloc_stream;
    CudaStream h2d_stream;
    CudaStream d2h_stream;
};

class MemoryManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemoryManager)

  public:
    struct Request;
    struct Buffer;
    struct Device;
    struct Transaction;
    struct DeferredDeletion;

    MemoryManager(
        std::shared_ptr<CudaStreamManager> streams,
        std::vector<MemoryDeviceInfo> devices);
    ~MemoryManager();

    void make_progress();
    bool is_idle() const;

    std::shared_ptr<Transaction> create_transaction(std::shared_ptr<Transaction> parent = nullptr);

    void create_buffer(BufferId id, BufferLayout layout);
    void delete_buffer(BufferId id);

    std::shared_ptr<Request> create_request(
        BufferId buffer_id,
        MemoryId memory_id,
        AccessMode mode,
        std::shared_ptr<Transaction> parent);
    bool poll_request(Request& req, CudaEventSet& deps_out);
    void release_request(std::shared_ptr<Request> req, CudaEvent event = {});

    BufferAccessor get_accessor(Request& req);

  private:
    void insert_into_lru(DeviceId device_id, Buffer& buffer);
    void remove_from_lru(DeviceId device_id, Buffer& buffer);

    bool try_free_device_memory(DeviceId device_id);
    bool try_allocate_device_async(DeviceId device_id, Buffer& buffer);
    void deallocate_device_async(DeviceId device_id, Buffer& buffer);

    void allocate_host(Buffer& buffer);
    void deallocate_host(Buffer& buffer);

    void lock_allocation_host(Buffer& buffer, Request& req);
    bool lock_allocation_device(DeviceId device_id, Buffer& buffer, Request& req);

    void unlock_allocation_host(Buffer& buffer, Request& req);
    void unlock_allocation_device(DeviceId device_id, Buffer& buffer, Request& req);

    std::optional<DeviceId> find_valid_device_entry(const Buffer& buffer) const;
    bool is_access_allowed(MemoryId memory_id, const Buffer& buffer, AccessMode mode) const;
    bool try_lock_access(MemoryId memory_id, Buffer& buffer, Request& req);
    void unlock_access(MemoryId memory_id, Buffer& buffer, Request& req, CudaEvent event);

    CudaEvent make_entry_valid(MemoryId memory_id, Buffer& buffer);
    void initiate_transfers(
        MemoryId memory_id,
        Buffer& buffer,
        Request& req,
        CudaEventSet& deps_out);

    CudaEvent copy_h2d(DeviceId device_id, Buffer& buffer);
    CudaEvent copy_d2h(DeviceId device_id, Buffer& buffer);

    void add_to_allocation_queue(DeviceId device_id, Request& req) const;
    void remove_from_allocation_queue(DeviceId device_id, Request& req) const;

    bool is_out_of_memory(DeviceId device_id, Transaction& trans);

    uint64_t m_next_transaction_id = 1;
    std::shared_ptr<CudaStreamManager> m_streams;
    std::unordered_map<BufferId, std::shared_ptr<Buffer>> m_buffers;
    std::unordered_set<std::shared_ptr<Request>> m_active_requests;
    std::unique_ptr<Device> m_devices[MAX_DEVICES];
    std::vector<DeferredDeletion> m_deferred_deletions;
};

using MemoryRequest = std::shared_ptr<MemoryManager::Request>;

}  // namespace kmm