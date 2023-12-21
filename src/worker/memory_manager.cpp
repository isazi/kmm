#include <iostream>
#include <stdexcept>
#include <utility>

#include "kmm/utils.hpp"
#include "kmm/worker/memory_manager.hpp"

namespace kmm {

struct MemoryManager::TransferCompletion: public kmm::TransferCompletion {
    TransferCompletion(MemoryManager& manager, BufferId buffer_id, MemoryId dst_id);
    TransferCompletion(const TransferCompletion&) = delete;
    TransferCompletion(TransferCompletion&&) noexcept = delete;
    ~TransferCompletion() override;
    void complete() override;

  private:
    std::shared_ptr<MemoryManager> m_manager;
    BufferId m_buffer_id;
    MemoryId m_dst_id;
};

class RequestList {
  public:
    void push_back(std::shared_ptr<MemoryManager::Request> req) {
        m_list.push_back(std::move(req));
    }

    bool is_empty() const {
        return m_list.empty();
    }

    const std::shared_ptr<MemoryManager::Request>& front() const {
        return m_list.at(0);
    }

    std::shared_ptr<MemoryManager::Request> pop_front() {
        auto req = std::move(m_list.at(0));
        m_list.pop_front();
        return req;
    }

  private:
    std::deque<std::shared_ptr<MemoryManager::Request>> m_list;
};

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
    std::shared_ptr<DataTransfer> active_transfer = nullptr;
    BufferState* buffer;
    MemoryId memory_id;
    bool writable;
    std::shared_ptr<const Waker> waker;
};

struct MemoryManager::Resource {
    RequestList waiters;
    BufferState* lru_oldest = nullptr;
};

struct MemoryManager::DataTransfer {
    DataTransfer(MemoryId src_id) : src_id(src_id) {}

    bool is_done = false;
    MemoryId src_id = MemoryId::invalid();
    RequestList requests;
    std::vector<MemoryId> devices;
};

struct MemoryManager::Entry {
    enum struct Status { Valid, Invalid, IncomingTransfer };

    std::shared_ptr<MemoryManager::DataTransfer> incoming_transfer;
    Status status = Status::Invalid;
    std::optional<std::unique_ptr<MemoryAllocation>> data = {};
};

struct MemoryManager::LRULinks {
    uint64_t num_users = 0;
    MemoryManager::BufferState* next = nullptr;
    MemoryManager::BufferState* prev = nullptr;
};

struct MemoryManager::BufferState {
    BufferId id;
    size_t num_bytes;
    uint64_t num_requests_active = 0;
    uint64_t num_readers = 0;
    uint64_t num_writers = 0;
    RequestList waiters = {};
    std::array<Entry, MAX_DEVICES> entries = {};
    std::array<LRULinks, MAX_DEVICES> links = {};
};

void MemoryManager::BufferDeleter::operator()(BufferState* ptr) const {
    return std::default_delete<BufferState> {}(ptr);
}

MemoryManager::MemoryManager(std::shared_ptr<Memory> memory) : m_memory(std::move(memory)) {
    KMM_ASSERT(m_memory);

    for (auto& resource : m_resources) {
        resource = std::make_unique<Resource>();
    }
}

MemoryManager::~MemoryManager() = default;

BufferId MemoryManager::create_buffer(const BlockLayout& layout) {
    size_t num_bytes = layout.num_bytes;

    auto id = BufferId(m_next_buffer_id++, num_bytes);
    auto state = std::unique_ptr<BufferState, BufferDeleter>(new BufferState {
        .id = id,
        .num_bytes = num_bytes,
    });

    m_buffers.insert({id, std::move(state)});
    return id;
}

void MemoryManager::delete_buffer(BufferId id) {
    auto it = m_buffers.find(id);

    if (it == m_buffers.end()) {
        return;
    }

    auto& buffer = it->second;
    KMM_ASSERT(buffer->num_requests_active == 0);

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        auto& entry = buffer->entries[i];
        KMM_ASSERT(entry.status != Entry::Status::IncomingTransfer);

        auto& links = buffer->links[i];
        KMM_ASSERT(links.num_users == 0);
    }

    for (uint8_t i = 0; i < MAX_DEVICES; i++) {
        auto& entry = buffer->entries[i];

        if (entry.data.has_value()) {
            auto alloc = std::move(*entry.data);

            entry.data = nullptr;
            entry.status = Entry::Status::Invalid;

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
    decrement_buffer_users(request->memory_id, buffer);

    if (request->memory_id != HOST_MEMORY) {
        decrement_buffer_users(HOST_MEMORY, buffer);
    }

    buffer->num_requests_active -= 1;
}

const MemoryAllocation* MemoryManager::view_buffer(const std::shared_ptr<Request>& request) {
    KMM_ASSERT(request->status == Request::Status::Ready);
    auto& entry = request->buffer->entries.at(request->memory_id);
    return entry.data->get();
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

    if (request->active_transfer) {
        if (!request->active_transfer->is_done) {
            return PollResult::Pending;
        }

        request->active_transfer = nullptr;
    }

    if (entry.status == Entry::Status::IncomingTransfer) {
        entry.incoming_transfer->requests.push_back(request);
        request->active_transfer = entry.incoming_transfer;
        return PollResult::Pending;
    }

    if (entry.status == Entry::Status::Valid) {
        return PollResult::Ready;
    }

    bool found_valid = false;
    auto src_id = MemoryId::invalid();
    auto dst_id = MemoryId(request->memory_id);

    if (dst_id == HOST_MEMORY) {
        for (uint8_t i = 0; i < MAX_DEVICES; i++) {
            if (buffer->entries[i].status == Entry::Status::Valid) {
                found_valid = true;
                src_id = MemoryId(i);
                break;
            }
        }
    } else {
        // First, check if it is possible to copy host to device
        auto& host_entry = buffer->entries[HOST_MEMORY];
        if (host_entry.status == Entry::Status::Valid) {
            found_valid = true;
            src_id = HOST_MEMORY;
        }

        // Next, check if it is possible to copy device to device
        if (!found_valid) {
            for (uint8_t i = 0; i < MAX_DEVICES; i++) {
                if (buffer->entries[i].status == Entry::Status::Valid
                    && m_memory->is_copy_possible(MemoryId(i), dst_id)) {
                    found_valid = true;
                    src_id = MemoryId(i);
                    break;
                }
            }
        }

        // Next, check if the host is waiting for a transfer
        if (!found_valid) {
            if (host_entry.status == Entry::Status::IncomingTransfer) {
                host_entry.incoming_transfer->requests.push_back(request);
                request->active_transfer = host_entry.incoming_transfer;
                return PollResult::Pending;
            }
        }

        // Finally, check if it is possible to copy device to host
        if (!found_valid) {
            for (uint8_t i = 0; i < MAX_DEVICES; i++) {
                if (buffer->entries[i].status == Entry::Status::Valid) {
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
        entry.status = Entry::Status::Valid;
        return PollResult::Ready;
    }

    auto transfer = this->initiate_transfer(src_id, dst_id, buffer);
    transfer->requests.push_back(request);
    request->active_transfer = std::move(transfer);

    return PollResult::Pending;
}

PollResult MemoryManager::poll_buffer_exclusive(const std::shared_ptr<Request>& request) {
    KMM_ASSERT(request->active_transfer == nullptr);
    KMM_ASSERT(request->writable);

    auto* buffer = request->buffer;
    KMM_ASSERT(buffer->entries.at(request->memory_id).status == Entry::Status::Valid);

    for (uint8_t memory_id = 0; memory_id < MAX_DEVICES; memory_id++) {
        auto& entry = buffer->entries[memory_id];

        if (request->memory_id == memory_id) {
            continue;
        }

        if (entry.status == Entry::Status::IncomingTransfer) {
            entry.incoming_transfer->requests.push_back(request);
            request->active_transfer = entry.incoming_transfer;
            return PollResult::Pending;
        } else if (entry.status == Entry::Status::Valid) {
            entry.status = Entry::Status::Invalid;
        } else {
            KMM_ASSERT(entry.status == Entry::Status::Invalid);
        }
    }

    return PollResult::Ready;
}

std::shared_ptr<MemoryManager::DataTransfer> MemoryManager::initiate_transfer(
    MemoryId src_id,
    MemoryId dst_id,
    BufferState* buffer) {
    auto& src_entry = buffer->entries[src_id];
    auto& dst_entry = buffer->entries[dst_id];

    KMM_ASSERT(src_entry.status == Entry::Status::Valid);
    KMM_ASSERT(dst_entry.status == Entry::Status::Invalid);

    dst_entry.status = Entry::Status::IncomingTransfer;
    dst_entry.incoming_transfer = std::make_shared<DataTransfer>(src_id);

    auto completion = std::make_unique<TransferCompletion>(*this, buffer->id, dst_id);

    this->m_memory->copy_async(
        src_id,
        src_entry.data.value().get(),
        0,
        dst_id,
        dst_entry.data.value().get(),
        0,
        buffer->num_bytes,
        std::move(completion));

    return dst_entry.incoming_transfer;
}

PollResult MemoryManager::submit_buffer_lock(const std::shared_ptr<Request>& request) const {
    auto* buffer = request->buffer;
    if (buffer->waiters.is_empty() && try_lock_buffer_for_request(buffer, request)) {
        return PollResult::Ready;
    } else {
        buffer->waiters.push_back(request);
        return PollResult::Pending;
    }
}

PollResult MemoryManager::poll_buffer_lock(const std::shared_ptr<Request>& request) const {
    auto* buffer = request->buffer;

    if (buffer->waiters.front() == request && try_lock_buffer_for_request(buffer, request)) {
        buffer->waiters.pop_front();

        // TODO: Maybe check if all requests are ready instead of just waking up the first one
        if (!buffer->waiters.is_empty()) {
            buffer->waiters.front()->waker->trigger_wakeup();
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
    if (!buffer->waiters.is_empty()) {
        buffer->waiters.front()->waker->trigger_wakeup();
    }
}

PollResult MemoryManager::submit_request_to_resource(
    MemoryId memory_id,
    const std::shared_ptr<Request>& request) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& buffer = request->buffer;
    auto& resource = m_resources[memory_id];
    auto& buffer_entry = buffer->entries[memory_id];

    if (!resource->waiters.is_empty() || !buffer_entry.data.has_value()) {
        resource->waiters.push_back(request);
        return poll_resource(memory_id, request);
    }

    // Remove from LRU
    increment_buffer_users(memory_id, buffer);
    return PollResult::Ready;
}

PollResult MemoryManager::poll_resource(
    MemoryId memory_id,
    const std::shared_ptr<Request>& request) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& buffer = request->buffer;
    auto& resource = m_resources[memory_id];
    auto& buffer_entry = buffer->entries[memory_id];

    if (resource->waiters.front() != request) {
        return PollResult::Pending;
    }

    while (!buffer_entry.data.has_value()) {
        if (auto alloc = m_memory->allocate(memory_id, buffer->num_bytes)) {
            buffer_entry.data = std::move(alloc);
            continue;
        }

        if (auto* victim = resource->lru_oldest) {
            if (evict_buffer(memory_id, victim) == PollResult::Pending) {
                return PollResult::Pending;
            }

            continue;
        }

        // TODO: Check for out-of-memory

        return PollResult::Pending;
    }

    // Remove from LRU
    increment_buffer_users(memory_id, buffer);

    resource->waiters.pop_front();
    if (!resource->waiters.is_empty()) {
        resource->waiters.front()->waker->trigger_wakeup();
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
    auto& links = buffer->links.at(memory_id);
    links.num_users++;

    if (links.num_users == 1) {
        remove_buffer_from_lru(memory_id, buffer);
    }
}

void MemoryManager::complete_transfer(BufferId buffer_id, MemoryId dst_id) {
    auto& buffer = m_buffers.at(buffer_id);
    auto& entry = buffer->entries.at(dst_id);

    KMM_ASSERT(entry.status == Entry::Status::IncomingTransfer);
    entry.status = Entry::Status::Valid;

    auto transfer = std::move(entry.incoming_transfer);
    transfer->is_done = true;

    while (!transfer->requests.is_empty()) {
        auto req = transfer->requests.pop_front();
        req->waker->trigger_wakeup();
    }

    while (!transfer->devices.empty()) {
        KMM_TODO();
        //auto device = entry.incoming_transfer.devices.back();
        //entry.incoming_transfer.devices.pop_back();

        // TODO: wakeup resource
    }
}

PollResult MemoryManager::evict_buffer(MemoryId memory_id, MemoryManager::BufferState* buffer) {
    auto& entry = buffer->entries.at(memory_id);
    auto& buffer_links = buffer->links.at(memory_id);

    KMM_ASSERT(entry.data.has_value());
    KMM_ASSERT(buffer_links.num_users == 0);

    if (entry.status == Entry::Status::IncomingTransfer) {
        entry.incoming_transfer->devices.push_back(memory_id);
        return PollResult::Pending;
    }

    for (auto& other_entry : buffer->entries) {
        if (other_entry.status == Entry::Status::IncomingTransfer
            && other_entry.incoming_transfer->src_id == memory_id) {
            entry.incoming_transfer->devices.push_back(memory_id);
            return PollResult::Pending;
        }
    }

    auto alloc = std::move(*entry.data);
    entry.data = nullptr;
    entry.status = Entry::Status::Invalid;

    remove_buffer_from_lru(memory_id, buffer);

    m_memory->deallocate(memory_id, std::move(alloc));
    return PollResult::Ready;
}

void MemoryManager::add_buffer_to_lru(MemoryId memory_id, MemoryManager::BufferState* buffer) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& resource = this->m_resources[memory_id];
    auto& links = buffer->links[memory_id];
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

        if (!resource->waiters.is_empty()) {
            resource->waiters.front()->waker->trigger_wakeup();
        }
    }
}

void MemoryManager::remove_buffer_from_lru(MemoryId memory_id, MemoryManager::BufferState* buffer) {
    KMM_ASSERT(memory_id < MAX_DEVICES);

    auto& buffer_links = buffer->links[memory_id];
    auto& resource = this->m_resources[memory_id];
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

MemoryManager::TransferCompletion::TransferCompletion(
    MemoryManager& manager,
    BufferId buffer_id,
    MemoryId dst_id) :
    m_manager(manager.shared_from_this()),
    m_buffer_id(buffer_id),
    m_dst_id(dst_id) {}

void MemoryManager::TransferCompletion::complete() {
    if (auto m = std::move(m_manager)) {
        m->complete_transfer(m_buffer_id, m_dst_id);
    }
}

MemoryManager::TransferCompletion::~TransferCompletion() {
    if (m_manager) {
        std::cerr << "ERROR: Transfer for buffer " << m_buffer_id.get()
                  << " has been deleted without "
                     "completing it first, this will leak memory!"
                  << std::endl;
    }
}

}  // namespace kmm