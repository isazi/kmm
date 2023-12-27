#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "kmm/block.hpp"
#include "kmm/memory.hpp"
#include "kmm/types.hpp"

namespace kmm {

class MemoryManager: public std::enable_shared_from_this<MemoryManager> {
  public:
    static constexpr size_t MAX_DEVICES = 5;
    static constexpr MemoryId HOST_MEMORY = MemoryId(0);
    class Request;
    struct DataTransfer;

    MemoryManager(std::unique_ptr<Memory> memory);
    ~MemoryManager();

    BufferId create_buffer(const BlockLayout&);
    PollResult delete_buffer(BufferId buffer_id, const std::shared_ptr<const Waker>& waker);

    std::shared_ptr<Request> create_request(
        BufferId buffer_id,
        MemoryId memory_id,
        bool writable,
        std::shared_ptr<const Waker> waker);
    const MemoryAllocation* view_buffer(const std::shared_ptr<Request>&);
    void delete_request(const std::shared_ptr<Request>&);

    PollResult poll_requests(
        const std::shared_ptr<Request>* begin,
        const std::shared_ptr<Request>* end);

    PollResult poll_request(const std::shared_ptr<Request>& request) {
        return poll_requests(&request, &request + 1);
    }

    PollResult poll_requests(const std::vector<std::shared_ptr<Request>>& requests) {
        return poll_requests(&*requests.begin(), &*requests.end());
    }

    void complete_transfer(DataTransfer& transfer);

  private:
    struct BufferState;
    struct Entry;
    struct Resource;
    struct LRULinks;

    struct BufferDeleter {
        void operator()(BufferState*) const;
    };

    PollResult poll_request_impl(const std::shared_ptr<Request>& request);

    PollResult evict_buffer(
        MemoryId memory_id,
        BufferState* buffer,
        const std::shared_ptr<Request>& request);

    std::optional<std::shared_ptr<MemoryManager::DataTransfer>> initiate_transfer(
        MemoryId src_id,
        MemoryId dst_id,
        BufferState* buffer);

    PollResult submit_buffer_lock(const std::shared_ptr<Request>& request) const;
    PollResult poll_buffer_lock(const std::shared_ptr<Request>& request) const;

    PollResult poll_buffer_data(const std::shared_ptr<Request>& request);
    PollResult poll_buffer_exclusive(const std::shared_ptr<Request>& request);

    PollResult submit_request_to_resource(MemoryId, const std::shared_ptr<Request>& request);
    PollResult poll_resource(MemoryId memory_id, const std::shared_ptr<Request>& request);

    bool try_lock_buffer_for_request(
        MemoryManager::BufferState* buffer,
        const std::shared_ptr<Request>& request) const;
    void unlock_buffer_for_request(
        MemoryManager::BufferState* buffer,
        const std::shared_ptr<Request>& request) const;

    void increment_buffer_users(MemoryId memory_id, BufferState* buffer);
    void decrement_buffer_users(MemoryId memory_id, BufferState* buffer);
    void add_buffer_to_lru(MemoryId memory_id, MemoryManager::BufferState* buffer);
    void remove_buffer_from_lru(MemoryId memory_id, MemoryManager::BufferState* buffer);

    std::array<std::unique_ptr<Resource>, MAX_DEVICES> m_resources;
    std::unordered_map<BufferId, std::unique_ptr<BufferState, BufferDeleter>> m_buffers;
    std::unique_ptr<Memory> m_memory;
    uint64_t m_next_buffer_id = 1;
};

using MemoryRequest = std::shared_ptr<MemoryManager::Request>;

}  // namespace kmm