#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "kmm/utils.hpp"
#include "kmm/worker/memory_manager.hpp"

namespace kmm {

class MemoryManager::Request {
  public:
    enum struct Status {
        Init,
        SubmitHostAllocation,
        PollHostAllocation,
        SubmitDeviceAllocation,
        PollDeviceAllocation,
        RequestAccess,
        PollAccess,
        PollData,
        Ready,
        Terminated
    };

    Status status = Request::Status::Init;
    BufferState* buffer = nullptr;
    MemoryId memory_id;
    bool writable;
    bool is_queued = false;
    std::shared_ptr<const Waker> waker;
};

class RequestList {
  public:
    bool push_back(std::shared_ptr<MemoryManager::Request> req) {
        if (!req->is_queued) {
            req->is_queued = true;
            m_list.push_back(std::move(req));
        }
    }

    std::optional<std::shared_ptr<MemoryManager::Request>> pop_front() {
        if (m_list.empty()) {
            return std::nullopt;
        }

        auto req = std::move(m_list.at(0));
        m_list.pop_front();
        req->is_queued = false;
        return req;
    }

  private:
    std::deque<std::shared_ptr<MemoryManager::Request>> m_list;
};

struct MemoryManager::Resource {
    std::optional<std::shared_ptr<Request>> front_waiter;
    RequestList waiters;
    BufferState* lru_oldest = nullptr;
};

struct MemoryManager::DataTransfer: IMemoryCompletion {
    DataTransfer(BufferState* buffer, MemoryId src_id, MemoryId dst_id) :
        buffer(buffer),
        src_id(src_id),
        dst_id(dst_id) {}

    bool initialize(MemoryManager& manager) {
        std::lock_guard guard {m_mutex};
        if (is_completed) {
            return false;
        }

        m_manager = manager.shared_from_this();
        return true;
    }

    void complete() override {
        std::unique_lock guard {m_mutex};
        if (!is_completed) {
            is_completed = true;

            auto manager = std::exchange(m_manager, nullptr);
            guard.unlock();

            if (manager) {
                manager->complete_transfer(*this);
            }
        }
    }

  public:
    BufferState* buffer;
    MemoryId src_id;
    MemoryId dst_id;
    RequestList requests;

  private:
    std::mutex m_mutex;
    bool is_completed = false;
    std::shared_ptr<MemoryManager> m_manager;
};

struct MemoryManager::Entry {
    enum struct Status { Valid, Invalid, IncomingTransfer };

    PollResult poll_status(const std::shared_ptr<Request>& request) {
        if (incoming_transfer) {
            incoming_transfer->requests.push_back(request);
            return PollResult::Pending;
        } else {
            return PollResult::Ready;
        }
    }

    Status status() {
        if (incoming_transfer) {
            return Status::IncomingTransfer;
        } else if (is_valid) {
            return Status::Valid;
        } else {
            return Status::Invalid;
        }
    }

    bool is_valid = false;
    std::shared_ptr<MemoryManager::DataTransfer> incoming_transfer;
    std::optional<std::unique_ptr<MemoryAllocation>> data = {};
};

struct MemoryManager::LRULinks {
    uint64_t num_users = 0;
    MemoryManager::BufferState* next = nullptr;
    MemoryManager::BufferState* prev = nullptr;
};

struct MemoryManager::BufferState {
    BufferState(const BufferState&) = delete;
    BufferState(BufferState&&) = delete;

    BufferId id;
    size_t num_bytes;
    uint64_t num_requests_active = 0;
    uint64_t num_readers = 0;
    uint64_t num_writers = 0;
    std::optional<std::shared_ptr<Request>> lock_waiters_head = {};
    RequestList lock_waiters = {};
    std::array<Entry, MAX_DEVICES> entries = {};
    std::array<LRULinks, MAX_DEVICES> links = {};
};

void MemoryManager::BufferDeleter::operator()(BufferState* ptr) const {
    return std::default_delete<BufferState> {}(ptr);
}

MemoryManager::MemoryManager(std::unique_ptr<Memory> memory) : m_memory(std::move(memory)) {
    KMM_ASSERT(m_memory);

    for (auto& resource : m_resources) {
        resource = std::make_unique<Resource>();
    }
}

MemoryManager::~MemoryManager() = default;

size_t size_to_align(size_t num_bytes) {
    size_t align = 1;
    while (align < num_bytes) {
        align *= 2;
    }
    return align;
}

BufferId MemoryManager::create_buffer(const BlockLayout& layout) {
    size_t num_bytes = layout.num_bytes;
    size_t align = 1;

    while (align < layout.alignment) {
        align *= 2;
    }

    // Add some padding bytes to make `num_bytes` a multiple of `align`
    num_bytes += (align - num_bytes % align) % align;

    auto id = BufferId(m_next_buffer_id++, num_bytes);
    auto state = std::unique_ptr<BufferState, BufferDeleter>(new BufferState {
        .id = id,
        .num_bytes = num_bytes,
    });

    m_buffers.insert({id, std::move(state)});
    return id;
}

PollResult MemoryManager::delete_buffer(BufferId id, const std::shared_ptr<const Waker>& waker) {
    auto it = m_buffers.find(id);

    if (it == m_buffers.end()) {
        return PollResult::Ready;
    }

    auto& buffer = it->second;
    KMM_ASSERT(buffer->num_requests_active == 0);

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        auto& entry = buffer->entries[i];
        KMM_ASSERT(entry.status() != Entry::Status::IncomingTransfer);

        auto& links = buffer->links[i];
        KMM_ASSERT(links.num_users == 0);
    }

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        auto& entry = buffer->entries[i];

        if (entry.data.has_value()) {
            auto alloc = std::move(*entry.data);

            entry.data = nullptr;
            entry.is_valid = false;

            remove_buffer_from_lru(MemoryId(i), buffer.get());
            m_memory->deallocate(MemoryId(i), std::move(alloc));
        }
    }

    m_buffers.erase(it);
}

std::shared_ptr<MemoryManager::Request> MemoryManager::create_request(
    BufferId buffer_id,
    MemoryId memory_id,
    bool writable,
    std::shared_ptr<const Waker> waker) {
    auto& buffer = m_buffers.at(buffer_id);
    buffer->num_requests_active += 1;

    return std::make_shared<Request>(Request {
        .buffer = buffer.get(),
        .memory_id = memory_id,
        .writable = writable,
        .waker = std::move(waker)});
}

void MemoryManager::delete_request(const std::shared_ptr<Request>& request) {
    if (request->status == Request::Status::Terminated) {
        return;
    }

    KMM_ASSERT(request->status == Request::Status::Ready);
    request->status = Request::Status::Terminated;

    auto* buffer = request->buffer;

    unlock_buffer_for_request(buffer, request);
    decrement_buffer_users(HOST_MEMORY, buffer);
    decrement_buffer_users(request->memory_id, buffer);

    buffer->num_requests_active -= 1;
}

const MemoryAllocation* MemoryManager::view_buffer(const std::shared_ptr<Request>& request) {
    KMM_ASSERT(request->status == Request::Status::Ready);
    auto& entry = request->buffer->entries.at(request->memory_id);
    return entry.data->get();
}

void MemoryManager::complete_transfer(DataTransfer& transfer) {
    auto* buffer = transfer.buffer;
    auto& dst_entry = buffer->entries[transfer.dst_id];

    KMM_ASSERT(dst_entry.incoming_transfer.get() == &transfer);
    dst_entry.incoming_transfer = nullptr;
    dst_entry.is_valid = true;

    while (auto req = transfer.requests.pop_front()) {
        if (poll_request_impl(*req) == PollResult::Ready) {
            (*req)->waker->trigger_wakeup();
        }
    }
}

PollResult MemoryManager::poll_requests(
    const std::shared_ptr<Request>* begin,
    const std::shared_ptr<Request>* end) {
    PollResult result = PollResult::Ready;

    for (const auto* it = begin; it != end; it++) {
        if (*it && poll_request_impl(*it) == PollResult::Pending) {
            result = PollResult::Pending;
        }
    }

    return result;
}

PollResult MemoryManager::poll_request_impl(const std::shared_ptr<Request>& request) {
    while (true) {
        if (request->is_queued) {
            return PollResult::Pending;
        }

        auto status = request->status;

        if (status == Request::Status::Init) {
            request->status = Request::Status::SubmitHostAllocation;

        } else if (status == Request::Status::SubmitHostAllocation) {
            if (submit_request_to_resource(HOST_MEMORY, request) == PollResult::Pending) {
                request->status = Request::Status::PollHostAllocation;
                return PollResult::Pending;
            }

            request->status = Request::Status::SubmitDeviceAllocation;

        } else if (status == Request::Status::PollHostAllocation) {
            if (poll_resource(HOST_MEMORY, request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            request->status = Request::Status::SubmitDeviceAllocation;
        } else if (status == Request::Status::SubmitDeviceAllocation) {
            if (request->memory_id != HOST_MEMORY) {
                if (submit_request_to_resource(request->memory_id, request)
                    == PollResult::Pending) {
                    request->status = Request::Status::PollDeviceAllocation;
                    return PollResult::Pending;
                }
            }

            request->status = Request::Status::RequestAccess;

        } else if (status == Request::Status::PollDeviceAllocation) {
            if (poll_resource(request->memory_id, request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            request->status = Request::Status::RequestAccess;

        } else if (status == Request::Status::RequestAccess) {
            if (submit_buffer_lock(request) == PollResult::Pending) {
                request->status = Request::Status::PollAccess;
                return PollResult::Pending;
            }

            request->status = Request::Status::PollData;

        } else if (status == Request::Status::PollAccess) {
            if (poll_buffer_lock(request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            request->status = Request::Status::PollData;

        } else if (status == Request::Status::PollData) {
            if (poll_buffer_data(request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            if (request->writable) {
                if (poll_buffer_exclusive(request) == PollResult::Pending) {
                    return PollResult::Pending;
                }
            }

            request->status = Request::Status::Ready;
        } else if (status == Request::Status::Ready || status == Request::Status::Terminated) {
            return PollResult::Ready;

        } else {
            throw std::runtime_error("invalid status");
        }
    }
}

PollResult MemoryManager::poll_buffer_data(const std::shared_ptr<Request>& request) {
    auto* buffer = request->buffer;
    auto& entry = buffer->entries.at(request->memory_id);

    if (entry.poll_status(request) == PollResult::Pending) {
        return PollResult::Pending;
    }

    if (entry.status() == Entry::Status::Valid) {
        return PollResult::Ready;
    }

    bool found_valid = false;
    auto src_id = MemoryId::invalid();
    auto dst_id = MemoryId(request->memory_id);

    if (dst_id == HOST_MEMORY) {
        for (uint8_t i = 0; i < MAX_DEVICES; i++) {
            if (buffer->entries[i].status() == Entry::Status::Valid) {
                found_valid = true;
                src_id = MemoryId(i);
                break;
            }
        }
    } else {
        // Host is waiting for a transfer
        auto& host_entry = buffer->entries[HOST_MEMORY];
        if (host_entry.poll_status(request) == PollResult::Pending) {
            return PollResult::Pending;
        }

        // First, check if it is possible to copy host to device
        if (host_entry.status() == Entry::Status::Valid) {
            found_valid = true;
            src_id = HOST_MEMORY;
        }

        // Next, check if it is possible to copy device to device
        if (!found_valid) {
            for (uint8_t i = 0; i < MAX_DEVICES; i++) {
                if (buffer->entries[i].status() == Entry::Status::Valid
                    && m_memory->is_copy_possible(MemoryId(i), dst_id)) {
                    found_valid = true;
                    src_id = MemoryId(i);
                    break;
                }
            }
        }

        // Finally, check if it is possible to copy device to host
        if (!found_valid) {
            for (uint8_t i = 0; i < MAX_DEVICES; i++) {
                if (buffer->entries[i].status() == Entry::Status::Valid) {
                    found_valid = true;
                    src_id = MemoryId(i);
                    dst_id = HOST_MEMORY;
                    break;
                }
            }
        }
    }

    if (!found_valid) {
        // No valid entry found, we just set the entry to valid now
        entry.is_valid = true;
        return PollResult::Ready;
    }

    if (auto transfer = initiate_transfer(src_id, dst_id, buffer)) {
        (*transfer)->requests.push_back(request);
        return PollResult::Pending;
    }

    return PollResult::Ready;
}

PollResult MemoryManager::poll_buffer_exclusive(const std::shared_ptr<Request>& request) {
    KMM_ASSERT(request->writable);

    auto* buffer = request->buffer;
    KMM_ASSERT(buffer->entries.at(request->memory_id).status() == Entry::Status::Valid);

    for (uint8_t memory_id = 0; memory_id < MAX_DEVICES; memory_id++) {
        auto& entry = buffer->entries[memory_id];

        if (request->memory_id == memory_id) {
            continue;
        }

        if (entry.poll_status(request) == PollResult::Pending) {
            return PollResult::Pending;
        }

        entry.is_valid = false;
    }

    return PollResult::Ready;
}

std::optional<std::shared_ptr<MemoryManager::DataTransfer>> MemoryManager::initiate_transfer(
    MemoryId src_id,
    MemoryId dst_id,
    BufferState* buffer) {
    auto& src_entry = buffer->entries[src_id];
    auto& dst_entry = buffer->entries[dst_id];

    KMM_ASSERT(src_entry.status() == Entry::Status::Valid);
    KMM_ASSERT(dst_entry.status() == Entry::Status::Invalid);

    auto transfer = std::make_shared<DataTransfer>(buffer, src_id, dst_id);

    m_memory->copy_async(
        src_id,
        src_entry.data.value().get(),
        0,
        dst_id,
        dst_entry.data.value().get(),
        0,
        buffer->num_bytes,
        MemoryCompletion(transfer));

    if (transfer->initialize(*this)) {
        dst_entry.incoming_transfer = transfer;
        return transfer;
    } else {
        return std::nullopt;
    }
}

PollResult MemoryManager::submit_buffer_lock(const std::shared_ptr<Request>& request) const {
    auto* buffer = request->buffer;
    if (buffer->lock_waiters_head.has_value()) {
        buffer->lock_waiters.push_back(request);
        return PollResult::Pending;
    } else if (!try_lock_buffer_for_request(buffer, request)) {
        buffer->lock_waiters_head = request;
        return PollResult::Pending;
    } else {
        return PollResult::Ready;
    }
}

PollResult MemoryManager::poll_buffer_lock(const std::shared_ptr<Request>& request) const {
    auto* buffer = request->buffer;

    if (buffer->lock_waiters_head == request && try_lock_buffer_for_request(buffer, request)) {
        if (auto head = buffer->lock_waiters.pop_front()) {
            // TODO: Maybe check if all requests are ready instead of just waking up the first one
            (*head)->waker->trigger_wakeup();
            buffer->lock_waiters_head = std::move(head);
        } else {
            buffer->lock_waiters_head = std::nullopt;
        }

        return PollResult::Ready;
    } else {
        return PollResult::Pending;
    }
}

bool MemoryManager::try_lock_buffer_for_request(
    MemoryManager::BufferState* buffer,
    const std::shared_ptr<Request>& request) const {
    if (buffer->num_writers > 0) {
        return false;
    }

    if (buffer->num_readers > 0 && request->writable) {
        return false;
    }

    if (request->writable) {
        buffer->num_writers++;
    } else {
        buffer->num_readers++;
    }

    return true;
}

void MemoryManager::unlock_buffer_for_request(
    MemoryManager::BufferState* buffer,
    const std::shared_ptr<Request>& request) const {
    if (request->writable) {
        KMM_ASSERT(buffer->num_writers > 0);
        buffer->num_writers--;
    } else {
        KMM_ASSERT(buffer->num_readers > 0);
        buffer->num_readers--;
    }

    // TODO: Maybe check if all requests are ready instead of just waking up the first one
    if (buffer->lock_waiters_head.has_value()) {
        auto req = *buffer->lock_waiters_head;
        req->waker->trigger_wakeup();
    }
}

PollResult MemoryManager::submit_request_to_resource(
    MemoryId memory_id,
    const std::shared_ptr<Request>& request) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& resource = m_resources[memory_id];

    if (resource->front_waiter.has_value()) {
        resource->waiters.push_back(request);
        return PollResult::Pending;
    }

    resource->front_waiter = request;
    return poll_resource(memory_id, request);
}

PollResult MemoryManager::poll_resource(
    MemoryId memory_id,
    const std::shared_ptr<Request>& request) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto* buffer = request->buffer;
    auto& resource = m_resources[memory_id];
    auto& buffer_entry = buffer->entries[memory_id];

    if (resource->front_waiter->get() != request.get()) {
        return PollResult::Pending;
    }

    while (!buffer_entry.data.has_value()) {
        if (auto alloc = m_memory->allocate(memory_id, buffer->num_bytes)) {
            buffer_entry.data = std::move(alloc);
            continue;
        }

        if (auto* victim = resource->lru_oldest) {
            if (evict_buffer(memory_id, victim, request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            continue;
        }

        // TODO: Check for out-of-memory

        return PollResult::Pending;
    }

    // Remove from LRU
    increment_buffer_users(memory_id, buffer);

    if (auto front = resource->waiters.pop_front()) {
        (*front)->waker->trigger_wakeup();
        resource->front_waiter = std::move(front);
    } else {
        resource->front_waiter = std::nullopt;
    }

    return PollResult::Ready;
}

void MemoryManager::decrement_buffer_users(MemoryId memory_id, MemoryManager::BufferState* buffer) {
    auto& links = buffer->links.at(memory_id);
    links.num_users--;

    if (links.num_users == 0) {
        add_buffer_to_lru(memory_id, buffer);
    }
}

void MemoryManager::increment_buffer_users(MemoryId memory_id, MemoryManager::BufferState* buffer) {
    KMM_ASSERT(memory_id < MAX_DEVICES);
    auto& links = buffer->links.at(memory_id);
    if (links.num_users == 0) {
        remove_buffer_from_lru(memory_id, buffer);
    }

    links.num_users++;
}

PollResult MemoryManager::evict_buffer(
    MemoryId memory_id,
    MemoryManager::BufferState* buffer,
    const std::shared_ptr<Request>& request) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& entry = buffer->entries[memory_id];
    auto& buffer_links = buffer->links[memory_id];

    KMM_ASSERT(entry.data.has_value());
    KMM_ASSERT(buffer_links.num_users == 0);

    if (entry.poll_status(request) == PollResult::Pending) {
        return PollResult::Pending;
    }

    if (entry.status() != Entry::Status::Valid) {
        return PollResult::Pending;
    }

    for (auto& other_entry : buffer->entries) {
        if (other_entry.incoming_transfer && other_entry.incoming_transfer->src_id == memory_id
            && other_entry.poll_status(request) == PollResult::Pending) {
            return PollResult::Pending;
        }
    }

    auto& host_entry = buffer->entries[HOST_MEMORY];
    if (host_entry.poll_status(request) == PollResult::Pending) {
        return PollResult::Pending;
    }

    if (host_entry.status() != Entry::Status::Valid) {
        if (auto transfer = initiate_transfer(memory_id, HOST_MEMORY, buffer)) {
            (*transfer)->requests.push_back(request);
            return PollResult::Pending;
        }
    }

    auto alloc = *std::move(entry.data);
    entry.data = std::nullopt;
    entry.is_valid = false;

    remove_buffer_from_lru(memory_id, buffer);

    m_memory->deallocate(memory_id, std::move(alloc));
    return PollResult::Ready;
}

void MemoryManager::add_buffer_to_lru(MemoryId memory_id, BufferState* buffer) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& resource = this->m_resources[memory_id];
    auto& links = buffer->links[memory_id];
    KMM_ASSERT(links.num_users == 0);

    auto* front = resource->lru_oldest;

    if (front != nullptr) {
        auto* back = front->links[memory_id].prev;

        back->links[memory_id].next = buffer;
        links.prev = back;

        links.next = front;
        front->links[memory_id].prev = buffer;
    } else {
        resource->lru_oldest = buffer;
        links.prev = buffer;
        links.next = buffer;

        if (resource->front_waiter.has_value()) {
            (*resource->front_waiter)->waker->trigger_wakeup();
        }
    }
}

void MemoryManager::remove_buffer_from_lru(MemoryId memory_id, BufferState* buffer) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& resource = m_resources[memory_id];
    auto& buffer_links = buffer->links[memory_id];
    KMM_ASSERT(buffer_links.num_users == 0);

    auto* prev = buffer_links.prev;
    auto* next = buffer_links.next;

    if (prev != next) {
        next->links[memory_id].prev = prev;
        prev->links[memory_id].next = next;
        resource->lru_oldest = next;
    } else {
        resource->lru_oldest = nullptr;
    }

    buffer_links.prev = nullptr;
    buffer_links.next = nullptr;
}

}  // namespace kmm