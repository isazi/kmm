#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "kmm/core/buffer.hpp"
#include "kmm/internals/cuda_stream_manager.hpp"
#include "kmm/internals/memory_allocator.hpp"

namespace kmm {

using TransactionId = uint64_t;

class MemoryManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemoryManager)

  public:
    struct Request;
    struct Buffer;
    struct Device;
    struct Transaction;

    MemoryManager(std::unique_ptr<MemoryAllocator> allocator);
    ~MemoryManager();

    void make_progress();
    bool is_idle(CudaStreamManager& streams) const;

    std::shared_ptr<Transaction> create_transaction(std::shared_ptr<Transaction> parent = nullptr);

    std::shared_ptr<Buffer> create_buffer(BufferLayout layout);
    void delete_buffer(std::shared_ptr<Buffer> buffer);

    std::shared_ptr<Request> create_request(
        std::shared_ptr<Buffer> buffer,
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
    bool is_access_allowed(const Buffer& buffer, MemoryId memory_id, AccessMode mode) const;
    void poll_access_queue(Buffer& buffer) const;
    bool try_lock_access(Buffer& buffer, Request& req);
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

    void add_to_buffer_access_queue(Buffer& buffer, Request& req) const;
    void remove_from_buffer_access_queue(Buffer& buffer, Request& req) const;

    bool is_out_of_memory(DeviceId device_id, Transaction& trans);

    void check_consistency() const;

    uint64_t m_next_transaction_id = 1;
    std::unordered_set<std::shared_ptr<Buffer>> m_buffers;
    std::unordered_set<std::shared_ptr<Request>> m_active_requests;
    std::unique_ptr<Device[]> m_devices;
    std::unique_ptr<MemoryAllocator> m_allocator;
};

using MemoryRequest = std::shared_ptr<MemoryManager::Request>;

}  // namespace kmm