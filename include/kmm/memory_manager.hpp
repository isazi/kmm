#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

#include "kmm/types.hpp"

namespace kmm {

class Allocation {
  public:
    virtual ~Allocation() = default;
};

class Completion {
  public:
    virtual ~Completion() = default;
    virtual void complete() = 0;
};

class Memory {
  public:
    virtual ~Memory() = default;
    virtual std::optional<std::unique_ptr<Allocation>> allocate(DeviceId id, size_t num_bytes) = 0;
    virtual void deallocate(DeviceId id, std::unique_ptr<Allocation> allocation) = 0;

    virtual void copy_async(
        DeviceId src_id,
        const Allocation* src_alloc,
        DeviceId dst_id,
        const Allocation* dst_alloc,
        size_t num_bytes,
        std::unique_ptr<Completion> completion) = 0;

    virtual bool is_copy_possible(DeviceId src_id, DeviceId dst_id) = 0;
};

class Waker {
  public:
    virtual ~Waker() = default;
    virtual void wakeup() const = 0;
};

enum class PollResult { Pending, Ready };

class MemoryManager: public std::enable_shared_from_this<MemoryManager> {
  public:
    class Request;

    MemoryManager(std::shared_ptr<Memory> memory);
    ~MemoryManager();
    void create_buffer(PhysicalBufferId buffer_id, const BufferLayout&);
    void delete_buffer(PhysicalBufferId buffer_id);

    std::shared_ptr<Request> create_request(
        PhysicalBufferId buffer_id,
        DeviceId device_id,
        bool writable,
        std::shared_ptr<Waker> waker);

    PollResult poll_request(const std::shared_ptr<Request>&);
    PollResult poll_requests(
        const std::shared_ptr<Request>* begin,
        const std::shared_ptr<Request>* end);

    const Allocation* view_buffer(const std::shared_ptr<Request>&);

    void delete_request(
        const std::shared_ptr<Request>&,
        std::optional<std::string> poison_reason = {});

  private:
    static constexpr size_t MAX_DEVICES = 5;
    class CompletionImpl;
    struct DataTransfer;
    struct BufferState;
    struct Entry;
    struct Resource;
    struct LRULinks;

    struct BufferDeleter {
        void operator()(BufferState*) const;
    };

    PollResult evict_buffer(DeviceId device_id, BufferState* buffer);

    DataTransfer& initiate_transfer(DeviceId src_id, DeviceId dst_id, BufferState* buffer);
    void complete_transfer(PhysicalBufferId buffer_id, DeviceId dst_id);

    PollResult submit_buffer_lock(const std::shared_ptr<Request>& request) const;
    PollResult poll_buffer_lock(const std::shared_ptr<Request>& request) const;

    PollResult poll_buffer_data(const std::shared_ptr<Request>& request);
    PollResult poll_buffer_exclusive(const std::shared_ptr<Request>& request);

    PollResult submit_request_to_resource(DeviceId, const std::shared_ptr<Request>& request);
    PollResult poll_resource(DeviceId device_id, const std::shared_ptr<Request>& request);

    bool try_lock_buffer_for_request(
        MemoryManager::BufferState* buffer,
        const std::shared_ptr<Request>& request) const;
    void unlock_buffer_for_request(
        MemoryManager::BufferState* buffer,
        const std::shared_ptr<Request>& request) const;

    void increment_buffer_users(DeviceId device_id, BufferState* buffer);
    void decrement_buffer_users(DeviceId device_id, BufferState* buffer);

    void remove_buffer_from_lru(DeviceId device_id, MemoryManager::BufferState* buffer);

    std::array<std::unique_ptr<Resource>, MAX_DEVICES> m_resources;
    std::unordered_map<PhysicalBufferId, std::unique_ptr<BufferState, BufferDeleter>> m_buffers =
        {};
    std::shared_ptr<Memory> m_memory;
};

using MemoryRequest = std::shared_ptr<MemoryManager::Request>;

}  // namespace kmm