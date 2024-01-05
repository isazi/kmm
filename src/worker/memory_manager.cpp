#include "spdlog/spdlog.h"

#include "kmm/block.hpp"
#include "kmm/memory.hpp"
#include "kmm/panic.hpp"
#include "kmm/result.hpp"
#include "kmm/types.hpp"
#include "kmm/worker/memory_manager.hpp"

namespace kmm {

template<typename Item>
struct IntrusiveQueue {
    void push_back(std::shared_ptr<Item> link) {
        auto* old_tail = std::exchange(m_tail, link.get());

        if (m_head == nullptr) {
            m_head = std::move(link);
        } else {
            link->prev = old_tail;
            old_tail->next = std::move(link);
        }
    }

    bool is_empty() const {
        return m_head == nullptr;
    }

    Item* front() const {
        return m_head.get();
    }

    std::shared_ptr<Item> pop_front() {
        auto* link = m_head.get();
        auto old_next = std::exchange(link->next, nullptr);

        if (old_next == nullptr) {
            m_tail = nullptr;
        } else {
            old_next->prev = nullptr;
        }

        return std::exchange(m_head, std::move(old_next));
    }

    std::shared_ptr<Item> pop(Item& link) {
        auto* old_prev = std::exchange(link.prev, nullptr);
        auto old_next = std::exchange(link.next, nullptr);

        if (old_next == nullptr) {
            m_tail = old_prev;
        } else {
            old_next->prev = old_prev;
        }

        if (old_prev == nullptr) {
            return std::exchange(m_head, std::move(old_next));
        } else {
            return std::exchange(old_prev->next, std::move(old_next));
        }
    }

  private:
    std::shared_ptr<Item> m_head = nullptr;
    Item* m_tail = nullptr;
};

struct AllocationGrant {
    AllocationGrant(MemoryManager::Request* parent) : parent(parent) {}

    bool is_pending() const {
        return status == Status::Queued;
    }

    bool is_error() const {
        return status == Status::Error;
    }

    enum struct Status {
        Free,
        Queued,
        Acquired,
        Error,
    } status = Status::Free;
    MemoryManager::Request* parent;
    std::shared_ptr<AllocationGrant> next = nullptr;
    AllocationGrant* prev = nullptr;
};

struct MemoryManager::Resource {
    Resource(MemoryId memory_id) : m_memory_id(memory_id) {}

    bool submit_allocate(std::shared_ptr<AllocationGrant> link, MemoryManager& manager);
    void deallocate(AllocationGrant& link, MemoryManager& manager);
    void make_progress(MemoryManager& manager);
    bool try_allocate(AllocationGrant& link, MemoryManager& manager);
    bool is_out_of_memory(AllocationGrant& link) const;
    Buffer* find_eviction_victim() const;

    void decrement_buffer_users(Buffer* buffer, MemoryManager& manager);
    void add_buffer_to_lru(Buffer* buffer);
    void increment_buffer_users(Buffer* buffer);
    void remove_buffer_from_lru(Buffer* buffer);

  private:
    MemoryId m_memory_id;

    std::shared_ptr<Operation> m_active_eviction = nullptr;
    IntrusiveQueue<AllocationGrant> m_queue;
    IntrusiveQueue<AllocationGrant> m_allocated;

    Buffer* m_lru_oldest = nullptr;
    Buffer* m_lru_newest = nullptr;
};

struct BufferEntry {
    enum struct Status {
        Unallocated,
        Valid,
        Invalid,
        IncomingTransfer,
    } status = Status::Unallocated;
    std::unique_ptr<MemoryAllocation> allocation = nullptr;
    MemoryManager::TransferOperation* incoming_operation = nullptr;
};

struct BufferLRULink {
    MemoryManager::Buffer* next = nullptr;
    MemoryManager::Buffer* prev = nullptr;
    size_t num_users = 0;
};

struct BufferLockGrant {
    explicit BufferLockGrant(const MemoryManager::Request* parent) : parent(parent) {}

    bool is_pending() const {
        return status != Status::Acquired;
    }

    enum struct Status {
        Free,
        Queued,
        Acquired,
    } status = Status::Free;
    const MemoryManager::Request* parent;
    std::shared_ptr<BufferLockGrant> next = nullptr;
    BufferLockGrant* prev = nullptr;
};

struct MemoryManager::Buffer {
    friend MemoryManager;

    Buffer(BufferId id, BlockLayout layout, std::vector<uint8_t> fill_pattern) :
        id(id),
        layout(layout),
        fill_pattern(std::move(fill_pattern)) {}

    void submit_lock(std::shared_ptr<BufferLockGrant> link);
    bool try_lock(MemoryId memory_id, AccessMode mode);
    void unlock(BufferLockGrant& link);

    BufferId id;
    BlockLayout layout;
    BufferEntry entries[MAX_DEVICES];
    BufferLRULink lru_links[MAX_DEVICES];
    std::vector<uint8_t> fill_pattern;
    size_t active_requests = 0;
    size_t refcount = 1;

  private:
    uint64_t m_num_readers = 0;
    uint64_t m_num_writers = 0;
    MemoryId m_writing_memory = MemoryId::invalid();
    IntrusiveQueue<BufferLockGrant> m_lock_queue;
};

struct MemoryManager::Operation: std::enable_shared_from_this<Operation> {
    bool is_running() const {
        std::unique_lock guard {m_mutex};
        return !m_completed;
    }

    Result<void> result() const {
        std::unique_lock guard {m_mutex};
        return m_result;
    }

    std::vector<std::shared_ptr<Request>> waiting_requests;
    uint64_t waiting_resources;

  protected:
    mutable std::mutex m_mutex;
    mutable bool m_completed = false;
    mutable Result<void> m_result;
};

struct MemoryManager::TransferOperation: CompletionHandler, Operation {
    TransferOperation(Buffer* buffer, MemoryId dst_id, MemoryId src_id = MemoryId::invalid()) :
        buffer(buffer),
        src_id(src_id),
        dst_id(dst_id) {}

    void initialize(MemoryManager& manager) const {
        std::unique_lock guard {m_mutex};
        if (!m_completed) {
            m_manager = manager.shared_from_this();
        }
    }

    void complete(Result<void> result) final {
        std::unique_lock guard {m_mutex};
        if (m_completed) {
            return;
        }

        m_completed = true;
        m_result = std::move(result);
        auto manager = std::exchange(m_manager, nullptr);

        guard.unlock();

        if (manager) {
            manager->complete_operation(*this);
        }
    }

    Buffer* const buffer;
    MemoryId const src_id;
    MemoryId const dst_id;

  private:
    mutable std::shared_ptr<MemoryManager> m_manager;
};

struct MemoryManager::FailedOperation: Operation {
    FailedOperation(ErrorPtr error) {
        m_completed = true;
        m_result = std::move(error);
    }
};

struct MemoryManager::Transaction {
    explicit Transaction(std::shared_ptr<const Waker> waker) : waker(std::move(waker)) {}

    std::shared_ptr<const Waker> waker;
};

struct MemoryManager::Request {
    Request(
        Buffer* buffer,
        MemoryId memory_id,
        AccessMode mode,
        std::shared_ptr<Transaction> parent) :
        buffer(buffer),
        memory_id(memory_id),
        mode(mode),
        parent(std::move(parent)),
        host_allocation(this),
        device_allocation(this),
        buffer_lock(this) {}

    enum struct Status {
        Init,
        WaitingForAccess,
        WaitingForData,
        Ready,
        Error,
        Terminated,
    };
    Status status = Status::Init;

    Buffer* buffer;
    MemoryId memory_id;
    AccessMode mode;
    ErrorPtr error;
    std::shared_ptr<Transaction> parent;

    AllocationGrant host_allocation;
    AllocationGrant device_allocation;
    BufferLockGrant buffer_lock;
    std::shared_ptr<Operation> ongoing_operation = nullptr;
};

MemoryManager::MemoryManager(std::unique_ptr<Memory> memory, std::optional<MemoryId> storage_id) :
    m_memory(std::move(memory)),
    m_storage_id(storage_id) {
    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        m_resources[i] = std::make_unique<Resource>(MemoryId(i));
    }
}

MemoryManager::~MemoryManager() = default;

std::shared_ptr<MemoryManager::Transaction> MemoryManager::create_transaction(
    std::shared_ptr<const Waker> waker) const {
    return std::make_shared<Transaction>(waker);
}

static size_t round_up_to_power_of_two(size_t align) {
    for (size_t i = 0; i < 63; i++) {
        if (align <= (size_t(1) << i)) {
            return size_t(1) << i;
        }
    }
    return 0;
}

static size_t round_up_to_multiple(size_t n, size_t k) {
    return n + ((n % k == 0) ? 0 : (k - n % k));
}

BufferId MemoryManager::create_buffer(
    const BlockLayout& layout,
    std::vector<uint8_t> fill_pattern) {
    std::lock_guard guard {m_mutex};

    size_t align = round_up_to_power_of_two(layout.alignment);
    size_t nbytes = round_up_to_multiple(layout.num_bytes, align);

    BufferId id = BufferId(m_next_buffer_id++, nbytes);
    m_buffers.insert({id, std::make_unique<Buffer>(id, layout, std::move(fill_pattern))});

    spdlog::debug("created buffer: id={}, size={}", id.get(), nbytes);
    return id;
}

void MemoryManager::delete_buffer(BufferId buffer_id) {
    decrement_buffer_refcount(buffer_id, std::numeric_limits<size_t>::max());
}

void MemoryManager::increment_buffer_refcount(BufferId buffer_id, size_t n) {
    std::lock_guard guard {m_mutex};
    auto* buffer = m_buffers.at(buffer_id).get();
    KMM_ASSERT(buffer->refcount > 0);
    buffer->refcount += n;
}

void MemoryManager::decrement_buffer_refcount(BufferId buffer_id, size_t n) {
    std::lock_guard guard {m_mutex};

    auto it = m_buffers.find(buffer_id);
    if (it == m_buffers.end()) {
        return;
    }

    auto* buffer = it->second.get();
    buffer->refcount = n < buffer->refcount ? buffer->refcount - n : 0;
    delete_buffer_when_idle(buffer);
}

void MemoryManager::delete_buffer_when_idle(Buffer* buffer) {
    if (buffer->refcount > 0 || buffer->active_requests > 0) {
        return;
    }

    KMM_ASSERT(buffer->m_num_writers == 0);
    KMM_ASSERT(buffer->m_num_readers == 0);

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        KMM_ASSERT(buffer->lru_links[i].num_users == 0);

        if (buffer->entries[i].status == BufferEntry::Status::IncomingTransfer) {
            return;
        }
    }

    spdlog::debug("deleting buffer: id={}", buffer->id);

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        auto& entry = buffer->entries[i];

        if (auto alloc = std::exchange(entry.allocation, nullptr)) {
            spdlog::debug("deallocating buffer memory: id={} memory={}", buffer->id, i);
            m_memory->deallocate(MemoryId(i), std::move(alloc));
            m_resources[i]->remove_buffer_from_lru(buffer);
        }

        entry.status = BufferEntry::Status::Unallocated;
    }

    m_buffers.erase(buffer->id);
}

std::shared_ptr<MemoryManager::Request> MemoryManager::create_request(
    BufferId buffer_id,
    MemoryId memory_id,
    AccessMode mode,
    std::shared_ptr<Transaction> parent) {
    std::lock_guard guard {m_mutex};
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto* buffer = m_buffers.at(buffer_id).get();
    buffer->active_requests += 1;

    auto req = std::make_shared<Request>(buffer, memory_id, mode, std::move(parent));
    spdlog::debug("create request for buffer: request={} buffer={}", (void*)req.get(), buffer_id);

    return req;
}

const MemoryAllocation* MemoryManager::view_buffer(const std::shared_ptr<Request>& req) {
    std::lock_guard guard {m_mutex};
    if (req->status == Request::Status::Error) {
        req->error.rethrow();
    }

    KMM_ASSERT(req->status == Request::Status::Ready);
    auto memory_id = req->memory_id;
    auto& entry = req->buffer->entries[memory_id];
    return entry.allocation.get();
}

void MemoryManager::delete_request(const std::shared_ptr<Request>& req) {
    std::lock_guard guard {m_mutex};
    if (req->status == Request::Status::Terminated) {
        return;
    }

    KMM_ASSERT(
        req->status == Request::Status::Init ||  //
        req->status == Request::Status::Error ||  //
        req->status == Request::Status::Ready);
    req->status = Request::Status::Terminated;

    spdlog::debug(
        "delete request for buffer: request={} buffer={}",
        (void*)req.get(),
        req->buffer->id);

    auto* buffer = req->buffer;
    buffer->active_requests -= 1;
    buffer->unlock(req->buffer_lock);

    m_resources[HOST_MEMORY]->deallocate(req->host_allocation, *this);
    m_resources[req->memory_id]->deallocate(req->device_allocation, *this);

    delete_buffer_when_idle(buffer);
}

void MemoryManager::complete_operation(TransferOperation& op) {
    std::lock_guard guard {m_mutex};
    auto* buffer = op.buffer;
    auto& dst_entry = buffer->entries[op.dst_id];

    KMM_ASSERT(dst_entry.incoming_operation == &op);
    dst_entry.incoming_operation = nullptr;

    bool success = bool(op.result());
    spdlog::debug(
        "transfer finished: buffer={} src={} dst={} success={}",
        op.buffer->id,
        op.src_id,
        op.dst_id,
        success);

    if (success) {
        dst_entry.status = BufferEntry::Status::Valid;
    }

    for (auto& request : op.waiting_requests) {
        if (poll_request_impl(request) == PollResult::Ready) {
            request->parent->waker->trigger_wakeup();
        }
    }

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        if ((op.waiting_resources & (1 << i)) != 0) {
            m_resources[i]->make_progress(*this);
        }
    }

    delete_buffer_when_idle(buffer);
}

PollResult MemoryManager::poll_requests(
    const std::shared_ptr<Request>* begin,
    const std::shared_ptr<Request>* end) {
    std::lock_guard guard {m_mutex};

    PollResult result = PollResult::Ready;

    for (const auto* it = begin; it != end; it++) {
        if (*it != nullptr && poll_request_impl(*it) == PollResult::Pending) {
            result = PollResult::Pending;
        }
    }

    return result;
}

PollResult MemoryManager::poll_request_impl(const std::shared_ptr<Request>& request) {
    while (true) {
        auto status = request->status;

        if (status == Request::Status::Init) {
            request->buffer->submit_lock(
                std::shared_ptr<BufferLockGrant> {request, &request->buffer_lock});

            m_resources[HOST_MEMORY]->submit_allocate(
                std::shared_ptr<AllocationGrant> {request, &request->host_allocation},
                *this);

            m_resources[request->memory_id]->submit_allocate(
                std::shared_ptr<AllocationGrant> {request, &request->device_allocation},
                *this);

            request->status = Request::Status::WaitingForAccess;
        } else if (status == Request::Status::WaitingForAccess) {
            if (request->buffer_lock.is_pending() ||  //
                request->host_allocation.is_pending() ||  //
                request->device_allocation.is_pending()) {
                return PollResult::Pending;
            }

            if (request->host_allocation.is_error() || request->device_allocation.is_error()) {
                request->status = Request::Status::Error;
                return PollResult::Ready;
            }

            request->status = Request::Status::WaitingForData;
        } else if (status == Request::Status::WaitingForData) {
            if (request->ongoing_operation) {
                if (request->ongoing_operation->is_running()) {
                    return PollResult::Pending;
                }

                auto ongoing_operation = std::exchange(request->ongoing_operation, nullptr);

                if (const auto* error = ongoing_operation->result().error_if_present()) {
                    request->error = *error;
                    request->status = Request::Status::Error;
                    request->ongoing_operation = nullptr;
                    continue;
                }
            }

            auto memory_id = request->memory_id;
            auto exclusive = request->mode != AccessMode::Read;

            if (auto transfer = poll_buffer_data(memory_id, request->buffer, exclusive)) {
                request->ongoing_operation = std::move(*transfer);
                request->ongoing_operation->waiting_requests.push_back(request);
                continue;
            }

            spdlog::debug(
                "granted request for buffer: request={} buffer={}",
                (void*)request.get(),
                request->buffer->id);
            request->status = Request::Status::Ready;
        } else if (
            status == Request::Status::Ready ||  //
            status == Request::Status::Error ||  //
            status == Request::Status::Terminated) {
            return PollResult::Ready;
        } else {
            KMM_PANIC("invalid request status");
        }
    }
}

bool MemoryManager::try_allocate_buffer(Buffer* buffer, MemoryId memory_id) {
    auto& entry = buffer->entries[memory_id];

    if (entry.status != BufferEntry::Status::Unallocated) {
        KMM_ASSERT(entry.allocation != nullptr);
        return true;
    }

    if (auto alloc = m_memory->allocate(memory_id, buffer->layout.num_bytes)) {
        KMM_ASSERT(entry.allocation == nullptr);
        m_resources[memory_id]->add_buffer_to_lru(buffer);
        entry.allocation = std::move(*alloc);
        entry.status = BufferEntry::Status::Invalid;
        return true;
    }

    return false;
}

std::optional<std::shared_ptr<MemoryManager::Operation>> MemoryManager::evict_buffer(
    MemoryId memory_id,
    Buffer* buffer) {
    spdlog::debug(
        "attempt to evict victim buffer from memory: buffer={} memory={}",
        buffer->id,
        memory_id);

    auto& entry = buffer->entries[memory_id];
    KMM_ASSERT(entry.status != BufferEntry::Status::Unallocated);
    KMM_ASSERT(entry.allocation != nullptr);
    KMM_ASSERT(buffer->lru_links[memory_id].num_users == 0);

    if (memory_id == m_storage_id) {
        return evict_storage_buffer(buffer);
    } else if (memory_id == HOST_MEMORY) {
        return evict_host_buffer(buffer);
    } else {
        return evict_device_buffer(memory_id, buffer);
    }
}

std::optional<std::shared_ptr<MemoryManager::Operation>> MemoryManager::evict_storage_buffer(
    Buffer* buffer) {
    return std::make_shared<FailedOperation>(ErrorPtr("cannot evict buffers from storage"));
}

std::optional<std::shared_ptr<MemoryManager::Operation>> MemoryManager::evict_host_buffer(
    Buffer* buffer) {
    if (!m_storage_id) {
        return std::make_shared<FailedOperation>(ErrorPtr("out of host memory"));
    }

    auto storage_id = m_storage_id.value();
    auto& host_entry = buffer->entries[HOST_MEMORY];
    auto& storage_entry = buffer->entries[storage_id];

    size_t valid_count = 0;
    MemoryId valid_id = MemoryId::invalid();

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        auto& peer_entry = buffer->entries[i];

        if (peer_entry.status == BufferEntry::Status::IncomingTransfer) {
            auto src_id = peer_entry.incoming_operation->src_id;
            auto dst_id = MemoryId(i);

            if (src_id == HOST_MEMORY || src_id == storage_id ||  //
                dst_id == HOST_MEMORY || dst_id == storage_id) {
                return peer_entry.incoming_operation->shared_from_this();
            }
        }

        if (peer_entry.status == BufferEntry::Status::Valid) {
            valid_count++;
            valid_id = MemoryId(i);
        }
    }

    if (storage_entry.status != BufferEntry::Status::Valid && valid_count > 0) {
        if (host_entry.status != BufferEntry::Status::Valid) {
            return initiate_transfer(buffer, valid_id, HOST_MEMORY);
        }

        if (storage_entry.status == BufferEntry::Status::Unallocated) {
            if (!try_allocate_buffer(buffer, storage_id)) {
                return std::make_shared<FailedOperation>(
                    ErrorPtr("failed to allocate storage memory"));
            }
        }

        return initiate_transfer(buffer, HOST_MEMORY, storage_id);
    }

    host_entry.status = BufferEntry::Status::Unallocated;
    auto alloc = std::exchange(host_entry.allocation, nullptr);

    m_memory->deallocate(HOST_MEMORY, std::move(alloc));
    m_resources[HOST_MEMORY]->remove_buffer_from_lru(buffer);

    return std::nullopt;
}

std::optional<std::shared_ptr<MemoryManager::Operation>> MemoryManager::evict_device_buffer(
    MemoryId memory_id,
    Buffer* buffer) {
    KMM_ASSERT(memory_id != HOST_MEMORY && memory_id != m_storage_id);
    auto& entry = buffer->entries[memory_id];

    if (entry.status == BufferEntry::Status::IncomingTransfer) {
        return entry.incoming_operation->shared_from_this();
    }

    size_t valid_count = 0;

    for (auto& peer_entry : buffer->entries) {
        if (peer_entry.status == BufferEntry::Status::IncomingTransfer) {
            if (peer_entry.incoming_operation->src_id == memory_id) {
                return peer_entry.incoming_operation->shared_from_this();
            }
        }

        if (peer_entry.status == BufferEntry::Status::Valid) {
            valid_count++;
        }
    }

    if (entry.status == BufferEntry::Status::Valid && valid_count == 1) {
        return initiate_transfer(buffer, memory_id, HOST_MEMORY);
    }

    entry.status = BufferEntry::Status::Unallocated;
    auto alloc = std::exchange(entry.allocation, nullptr);

    m_memory->deallocate(memory_id, std::move(alloc));
    m_resources[memory_id]->remove_buffer_from_lru(buffer);

    return std::nullopt;
}

std::optional<std::shared_ptr<MemoryManager::Operation>> MemoryManager::poll_buffer_data(
    MemoryId memory_id,
    Buffer* buffer,
    bool exclusive) {
    auto& entry = buffer->entries[memory_id];
    KMM_ASSERT(entry.status != BufferEntry::Status::Unallocated);

    if (entry.status == BufferEntry::Status::IncomingTransfer) {
        return entry.incoming_operation->shared_from_this();
    }

    if (entry.status == BufferEntry::Status::Invalid) {
        if (memory_id != HOST_MEMORY) {
            auto& host_entry = buffer->entries[HOST_MEMORY];
            if (host_entry.status == BufferEntry::Status::IncomingTransfer) {
                return host_entry.incoming_operation->shared_from_this();
            }

            if (host_entry.status == BufferEntry::Status::Valid) {
                return initiate_transfer(buffer, HOST_MEMORY, memory_id);
            }

            for (uint8_t i = 0; i < MAX_DEVICES; i++) {
                if (buffer->entries[i].status == BufferEntry::Status::Valid
                    && m_memory->is_copy_possible(MemoryId(i), memory_id)) {
                    return initiate_transfer(buffer, MemoryId(i), memory_id);
                }
            }
        }

        for (uint8_t i = 0; i < MAX_DEVICES; i++) {
            if (buffer->entries[i].status == BufferEntry::Status::Valid) {
                return initiate_transfer(buffer, MemoryId(i), HOST_MEMORY);
            }
        }

        if (!buffer->fill_pattern.empty()) {
            return initiate_fill(buffer, memory_id, buffer->fill_pattern);
        }

        entry.status = BufferEntry::Status::Valid;
    }

    if (exclusive) {
        for (uint8_t i = 0; i < MAX_DEVICES; i++) {
            if (i == memory_id) {
                continue;
            }

            auto& peer_entry = buffer->entries[i];
            if (peer_entry.status == BufferEntry::Status::IncomingTransfer) {
                return peer_entry.incoming_operation->shared_from_this();
            }

            if (peer_entry.status == BufferEntry::Status::Valid) {
                peer_entry.status = BufferEntry::Status::Invalid;
            }
        }
    }

    return std::nullopt;
}

std::shared_ptr<MemoryManager::Operation> MemoryManager::initiate_transfer(
    Buffer* buffer,
    MemoryId src_id,
    MemoryId dst_id) {
    spdlog::debug("initiate transfer: buffer={} src={} dst={}", buffer->id, src_id, dst_id);

    auto& src_entry = buffer->entries[src_id];
    auto& dst_entry = buffer->entries[dst_id];

    KMM_ASSERT(src_entry.status == BufferEntry::Status::Valid);
    KMM_ASSERT(dst_entry.status == BufferEntry::Status::Invalid);
    KMM_ASSERT(src_entry.allocation != nullptr);
    KMM_ASSERT(dst_entry.allocation != nullptr);
    KMM_ASSERT(dst_entry.incoming_operation == nullptr);

    auto op = std::make_shared<TransferOperation>(buffer, dst_id, src_id);

    m_memory->copy_async(
        src_id,
        src_entry.allocation.get(),
        0,
        dst_id,
        dst_entry.allocation.get(),
        0,
        buffer->layout.num_bytes,
        Completion(op));

    op->initialize(*this);

    if (op->is_running()) {
        dst_entry.status = BufferEntry::Status::IncomingTransfer;
        dst_entry.incoming_operation = op.get();
    } else {
        bool success = op->result().has_value();
        spdlog::debug(
            "transfer finished: buffer={} src={} dst={} success={}",
            buffer->id,
            src_id,
            dst_id,
            success);

        if (success) {
            dst_entry.status = BufferEntry::Status::Valid;
        }
    }

    return op;
}

std::shared_ptr<MemoryManager::Operation> MemoryManager::initiate_fill(
    Buffer* buffer,
    MemoryId memory_id,
    const std::vector<uint8_t>& fill_pattern) {
    spdlog::debug("initiate fill: buffer={} memory={}", buffer->id, memory_id);

    auto& entry = buffer->entries[memory_id];

    KMM_ASSERT(entry.status == BufferEntry::Status::Invalid);
    KMM_ASSERT(entry.allocation != nullptr);
    KMM_ASSERT(entry.incoming_operation == nullptr);

    auto op = std::make_shared<TransferOperation>(buffer, memory_id);

    m_memory->fill_async(
        memory_id,
        entry.allocation.get(),
        0,
        buffer->layout.num_bytes,
        fill_pattern,
        Completion(op));
    op->initialize(*this);

    if (op->is_running()) {
        entry.status = BufferEntry::Status::IncomingTransfer;
        entry.incoming_operation = op.get();
    } else if (op->result()) {
        entry.status = BufferEntry::Status::Valid;
    }

    return op;
}

bool MemoryManager::Resource::submit_allocate(
    std::shared_ptr<AllocationGrant> link,
    MemoryManager& manager) {
    KMM_ASSERT(link->status == AllocationGrant::Status::Free);

    if (m_queue.is_empty() && try_allocate(*link, manager)) {
        if (link->status == AllocationGrant::Status::Acquired) {
            m_allocated.push_back(std::move(link));
        }
    } else {
        link->status = AllocationGrant::Status::Queued;
        m_queue.push_back(std::move(link));
    }

    return false;
}

void MemoryManager::Resource::deallocate(AllocationGrant& link, MemoryManager& manager) {
    switch (link.status) {
        case AllocationGrant::Status::Free:
        case AllocationGrant::Status::Error:
            // nothing
            break;
        case AllocationGrant::Status::Queued:
            m_queue.pop(link);
            break;
        case AllocationGrant::Status::Acquired:
            m_allocated.pop(link);
            decrement_buffer_users(link.parent->buffer, manager);
            break;
    }

    link.status = AllocationGrant::Status::Free;
}

void MemoryManager::Resource::make_progress(MemoryManager& manager) {
    while (auto* head = m_queue.front()) {
        if (!try_allocate(*head, manager)) {
            break;
        }

        auto link = m_queue.pop(*head);
        if (link->status == AllocationGrant::Status::Acquired) {
            m_allocated.push_back(std::move(link));
        }
        head->parent->parent->waker->trigger_wakeup();
    }
}

bool MemoryManager::Resource::try_allocate(AllocationGrant& link, MemoryManager& manager) {
    while (true) {
        auto* buffer = link.parent->buffer;

        // First, try to allocate the buffer
        if (manager.try_allocate_buffer(buffer, m_memory_id)) {
            link.status = AllocationGrant::Status::Acquired;
            increment_buffer_users(buffer);
            return true;
        }

        // Second, if there is an eviction ongoing, wait until it completes.
        if (m_active_eviction) {
            if (m_active_eviction->is_running()) {
                return false;
            }

            auto active_eviction = std::exchange(m_active_eviction, nullptr);

            if (const auto* error = active_eviction->result().error_if_present()) {
                link.status = AllocationGrant::Status::Error;
                link.parent->error = *error;
                return true;
            }
        }

        // Third, check if we can evict a buffer that is not in use
        if (auto* victim = find_eviction_victim()) {
            if (auto transfer = manager.evict_buffer(m_memory_id, victim)) {
                m_active_eviction = std::move(*transfer);
                m_active_eviction->waiting_resources |= 1 << m_memory_id;
            } else {
                spdlog::debug(
                    "evicted buffer from memory: buffer={} memory={}",
                    victim->id,
                    m_memory_id);
            }

            continue;
        }

        // Fourth, check if we are out of memory
        if (is_out_of_memory(link)) {
            link.status = AllocationGrant::Status::Error;
            link.parent->error = ErrorPtr(fmt::format(
                "cannot allocate {} bytes in memory {}, out of memory",
                buffer->layout.num_bytes,
                m_memory_id.get()));
            return true;
        }

        return false;
    }
}

MemoryManager::Buffer* MemoryManager::Resource::find_eviction_victim() const {
    auto memory_id = m_memory_id;

    if (m_lru_oldest == nullptr) {
        return nullptr;
    }

    // Find which buffer to evict. We use the follow heuristics:
    // 1. First, find a buffer that is invalid. For these, we do not need to perform a transfer
    // 2. Otherwise, find a buffer that has no active requests.
    // 3. Otherwise, find a buffer that is large. We want to keep small buffers in memory
    // 4. Otherwise, just return the oldest buffer.
    for (auto* it = m_lru_oldest; it != nullptr; it = it->lru_links[memory_id].next) {
        if (it->entries[memory_id].status != BufferEntry::Status::Valid) {
            return it;
        }
    }

    for (auto* it = m_lru_oldest; it != nullptr; it = it->lru_links[memory_id].next) {
        if (it->active_requests == 0) {
            return it;
        }
    }

    for (auto* it = m_lru_oldest; it != nullptr; it = it->lru_links[memory_id].next) {
        if (it->layout.num_bytes > 1024 * 1024) {
            return it;
        }
    }

    return m_lru_oldest;
}

bool MemoryManager::Resource::is_out_of_memory(AllocationGrant& link) const {
    for (auto* item = m_allocated.front(); item != nullptr; item = item->next.get()) {
        if (item->parent->parent.get() != link.parent->parent.get()) {
            return false;
        }
    }

    const auto* buffer = link.parent->buffer;
    spdlog::warn(
        "failed to grant memory allocation to request, out of memory for"
        "device {} while allocating buffer {} of {} bytes",
        m_memory_id,
        buffer->id,
        buffer->layout.num_bytes);

    return true;
}

void MemoryManager::Resource::decrement_buffer_users(Buffer* buffer, MemoryManager& manager) {
    auto& links = buffer->lru_links[m_memory_id];
    links.num_users--;

    if (links.num_users == 0) {
        add_buffer_to_lru(buffer);
        make_progress(manager);
    }
}

void MemoryManager::Resource::add_buffer_to_lru(Buffer* buffer) {
    if (m_lru_oldest == nullptr) {
        m_lru_oldest = buffer;
        m_lru_newest = buffer;
    } else {
        auto* last = m_lru_newest;
        buffer->lru_links[m_memory_id].prev = last;
        last->lru_links[m_memory_id].next = buffer;
        m_lru_newest = buffer;
    }
}

void MemoryManager::Resource::increment_buffer_users(Buffer* buffer) {
    auto& links = buffer->lru_links[m_memory_id];
    if (links.num_users == 0) {
        remove_buffer_from_lru(buffer);
    }

    links.num_users++;
}

void MemoryManager::Resource::remove_buffer_from_lru(Buffer* buffer) {
    auto& links = buffer->lru_links[m_memory_id];
    auto* prev = std::exchange(links.prev, nullptr);
    auto* next = std::exchange(links.next, nullptr);

    if (next != nullptr) {
        next->lru_links[m_memory_id].prev = prev;
    } else {
        m_lru_newest = prev;
    }

    if (prev != nullptr) {
        prev->lru_links[m_memory_id].next = next;
    } else {
        m_lru_oldest = next;
    }
}

void MemoryManager::Buffer::submit_lock(std::shared_ptr<BufferLockGrant> link) {
    KMM_ASSERT(link->status == BufferLockGrant::Status::Free);

    if (m_lock_queue.is_empty() && try_lock(link->parent->memory_id, link->parent->mode)) {
        link->status = BufferLockGrant::Status::Acquired;
    } else {
        link->status = BufferLockGrant::Status::Queued;
        m_lock_queue.push_back(std::move(link));
    }
}

bool MemoryManager::Buffer::try_lock(MemoryId memory_id, AccessMode mode) {
    switch (mode) {
        case AccessMode::Read:
            if (m_num_writers == 0) {
                m_num_readers++;
                return true;
            }
            break;
        case AccessMode::ReadWrite:
            if (m_num_readers == 0 && m_num_writers == 0) {
                m_writing_memory = memory_id;
                m_num_readers++;
                m_num_writers++;
                return true;
            }
            break;
        case AccessMode::Write:
            if (m_num_readers == 0 && (m_num_writers == 0 || m_writing_memory == memory_id)) {
                m_writing_memory = memory_id;
                m_num_writers++;
                return true;
            }
            break;
    }

    return false;
}

void MemoryManager::Buffer::unlock(BufferLockGrant& link) {
    switch (link.status) {
        case BufferLockGrant::Status::Free:
            return;
        case BufferLockGrant::Status::Queued:
            m_lock_queue.pop(link);
            return;
        case BufferLockGrant::Status::Acquired:
            break;
    }

    link.status = BufferLockGrant::Status::Free;

    switch (link.parent->mode) {
        case AccessMode::Read:
            m_num_readers--;
            break;
        case AccessMode::ReadWrite:
            m_num_readers--;
            m_num_writers--;
            break;
        case AccessMode::Write:
            m_num_writers--;
            break;
    }

    while (!m_lock_queue.is_empty()) {
        auto* head = m_lock_queue.front();

        if (!try_lock(head->parent->memory_id, head->parent->mode)) {
            break;
        }

        head->status = BufferLockGrant::Status::Acquired;
        head->parent->parent->waker->trigger_wakeup();
        m_lock_queue.pop_front();
    }
}

}  // namespace kmm