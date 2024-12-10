#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "kmm/core/buffer.hpp"
#include "kmm/internals/device_stream_manager.hpp"
#include "kmm/internals/memory_system.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

using TransactionId = uint64_t;

class MemoryManager {
    KMM_NOT_COPYABLE_OR_MOVABLE(MemoryManager)

  public:
    struct Request;
    struct Buffer;
    struct Device;
    struct Transaction;

    MemoryManager(std::shared_ptr<MemorySystem> memory);
    ~MemoryManager();

    void make_progress();
    bool is_idle(DeviceStreamManager& streams) const;

    std::shared_ptr<Transaction> create_transaction(std::shared_ptr<Transaction> parent = nullptr);

    std::shared_ptr<Buffer> create_buffer(BufferLayout layout, std::string name="");
    void delete_buffer(std::shared_ptr<Buffer> buffer);

    std::shared_ptr<Request> create_request(
        std::shared_ptr<Buffer> buffer,
        MemoryId memory_id,
        AccessMode mode,
        std::shared_ptr<Transaction> parent
    );
    Poll poll_request(Request& req, DeviceEventSet* deps_out);
    void release_request(std::shared_ptr<Request> req, DeviceEvent event = {});

    static BufferAccessor get_accessor(Request& req);

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

    static void unlock_allocation_host(Buffer& buffer, Request& req);
    void unlock_allocation_device(DeviceId device_id, Buffer& buffer, Request& req);

    static std::optional<DeviceId> find_valid_device_entry(const Buffer& buffer);
    static bool is_access_allowed(const Buffer& buffer, const Request& req);
    static void poll_access_queue(Buffer& buffer);
    static void unlock_access(MemoryId memory_id, Buffer& buffer, Request& req, DeviceEvent event);

    void make_entry_valid(MemoryId memory_id, Buffer& buffer, DeviceEventSet* deps_out);
    bool try_lock_access(
        MemoryId memory_id,
        Buffer& buffer,
        Request& req,
        DeviceEventSet* deps_out
    );

    DeviceEvent fill_buffer(
        MemoryId memory_id,
        Buffer& buffer,
        const std::vector<uint8_t>& fill_pattern
    );
    DeviceEvent copy_h2d(DeviceId device_id, Buffer& buffer);
    DeviceEvent copy_d2h(DeviceId device_id, Buffer& buffer);

    void add_to_allocation_queue(DeviceId device_id, Request& req) const;
    void remove_from_allocation_queue(DeviceId device_id, Request& req) const;

    static void add_to_buffer_access_queue(Buffer& buffer, Request& req);
    static void remove_from_buffer_access_queue(Buffer& buffer, Request& req);

    bool is_out_of_memory(DeviceId device_id, Request& req);

    void check_consistency() const;

    std::shared_ptr<MemorySystem> m_memory;
    std::unique_ptr<Device[]> m_devices;
    std::unordered_set<std::shared_ptr<Buffer>> m_buffers;
    std::unordered_set<std::shared_ptr<Request>> m_active_requests;
    uint64_t m_next_transaction_id = 1;
    uint64_t m_next_request_id = 1;
};

using MemoryRequest = std::shared_ptr<MemoryManager::Request>;
using MemoryRequestList = std::vector<std::shared_ptr<MemoryManager::Request>>;

}  // namespace kmm