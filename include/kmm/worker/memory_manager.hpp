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

/**
 * There are three possible access modes:
 * * `Read`: Only read from a buffer. Multiple readers can concurrently read from different memories.
 * * `ReadWrite`: Read and write from a buffer. Only one reader-writer can be active concurrently.
 *
 * * `Write`: Only write to a buffer. Multiple writes can be active concurrently, however
 *            concurrent writers can only write to the same memory.
 */
enum struct AccessMode { Read, ReadWrite, Write };

class MemoryManager: public std::enable_shared_from_this<MemoryManager> {
  public:
    static constexpr size_t MAX_DEVICES = 5;
    static constexpr MemoryId HOST_MEMORY = MemoryId(0);

    struct Request;
    struct Operation;
    struct TransferOperation;
    struct FailedOperation;
    struct Resource;
    struct Buffer;
    struct Transaction;

    MemoryManager(
        std::unique_ptr<Memory> memory,
        std::optional<MemoryId> storage_id = std::nullopt);
    ~MemoryManager();

    BufferId create_buffer(const BlockLayout&, std::vector<uint8_t> fill_pattern = {});
    void delete_buffer(BufferId buffer_id);

    void increment_buffer_refcount(BufferId buffer_id, size_t n = 1);
    void decrement_buffer_refcount(BufferId buffer_id, size_t n = 1);

    std::shared_ptr<Transaction> create_transaction(std::shared_ptr<const Waker> waker) const;
    std::shared_ptr<Request> create_request(
        BufferId buffer_id,
        MemoryId memory_id,
        AccessMode mode,
        std::shared_ptr<Transaction> parent);
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

    void complete_operation(TransferOperation& op);

  private:
    void delete_buffer_when_idle(Buffer* buffer);

    PollResult poll_request_impl(const std::shared_ptr<Request>& request);

    bool try_allocate_buffer(Buffer* buffer, MemoryId memory_id);

    std::optional<std::shared_ptr<Operation>> evict_buffer(MemoryId memory_id, Buffer* buffer);
    std::optional<std::shared_ptr<Operation>> evict_storage_buffer(Buffer* buffer);
    std::optional<std::shared_ptr<Operation>> evict_host_buffer(Buffer* buffer);
    std::optional<std::shared_ptr<Operation>> evict_device_buffer(
        MemoryId memory_id,
        Buffer* buffer);

    std::optional<std::shared_ptr<Operation>> poll_buffer_data(
        MemoryId memory_id,
        Buffer* buffer,
        bool exclusive);

    std::shared_ptr<Operation> initiate_transfer(Buffer* buffer, MemoryId src_id, MemoryId dst_id);
    std::shared_ptr<Operation> initiate_fill(
        Buffer* buffer,
        MemoryId memory_id,
        const std::vector<uint8_t>& fill_pattern);

    std::mutex m_mutex;
    std::array<std::unique_ptr<Resource>, MAX_DEVICES> m_resources;
    std::unordered_map<BufferId, std::unique_ptr<Buffer>> m_buffers;
    std::unique_ptr<Memory> m_memory;
    uint64_t m_next_buffer_id = 1;
    std::optional<MemoryId> m_storage_id;
};

using MemoryRequest = std::shared_ptr<MemoryManager::Request>;

}  // namespace kmm