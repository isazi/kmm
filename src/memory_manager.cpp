#include "kmm/memory_manager.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

#include "kmm/utils.hpp"

namespace kmm {

class MemoryManager::CompletionImpl: public Completion {
  public:
    CompletionImpl(MemoryManager& manager, PhysicalBufferId buffer_id, DeviceId dst_id);
    ~CompletionImpl() override;
    void complete() override;

  private:
    std::shared_ptr<MemoryManager> m_manager;
    PhysicalBufferId m_buffer_id;
    DeviceId m_dst_id;
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

enum struct RequestStatus {
    Init,
    SubmitHostAllocation,
    PollHostAllocation,
    SubmitDeviceAllocation,
    PollDeviceAllocation,
    RequestAccess,
    PollAccess,
    PollData,
    WaitingForTransfer,
    Ready,
    Terminated
};

class MemoryManager::Request {
  public:
    RequestStatus status;
    BufferState* buffer;
    DeviceId device_id;
    bool writable;
    std::shared_ptr<Waker> waker;
};

struct MemoryManager::Resource {
    RequestList waiters;
    BufferState* lru_oldest = nullptr;
};

struct MemoryManager::DataTransfer {
    DeviceId src_id = DeviceId::invalid();
    RequestList requests;
    std::vector<DeviceId> devices;
};

struct MemoryManager::Entry {
    enum struct Status { Valid, Invalid, IncomingTransfer };

    MemoryManager::DataTransfer incoming_transfer;
    Status status = Status::Invalid;
    std::optional<std::unique_ptr<Allocation>> data = {};
};

struct MemoryManager::LRULinks {
    uint64_t num_users = 0;
    MemoryManager::BufferState* next = nullptr;
    MemoryManager::BufferState* prev = nullptr;
};

struct MemoryManager::BufferState {
    PhysicalBufferId id;
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

void MemoryManager::create_buffer(PhysicalBufferId id, const BufferLayout& descr) {
    auto state = std::unique_ptr<BufferState, BufferDeleter>(new BufferState {
        .id = id,
        .num_bytes = descr.num_bytes,
    });

    m_buffers.insert({id, std::move(state)});
}

void MemoryManager::delete_buffer(PhysicalBufferId id) {
    if (auto it = m_buffers.find(id); it != m_buffers.end()) {
        auto& buffer = it->second;

        if (buffer->num_requests_active > 0) {
            throw std::runtime_error("cannot delete buffer while requests are still active");
        }

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

                remove_buffer_from_lru(DeviceId(i), buffer.get());
                m_memory->deallocate(DeviceId(i), std::move(alloc));
            }
        }

        m_buffers.erase(it);
    }
}

std::shared_ptr<MemoryManager::Request> MemoryManager::create_request(
    PhysicalBufferId buffer_id,
    DeviceId device_id,
    bool writable,
    std::shared_ptr<Waker> waker) {
    auto& buffer = m_buffers.at(buffer_id);
    buffer->num_requests_active += 1;

    auto request = std::make_shared<Request>(Request {
        .status = RequestStatus::Init,
        .buffer = buffer.get(),
        .device_id = device_id,
        .writable = writable,
        .waker = std::move(waker)});

    return request;
}

void MemoryManager::delete_request(
    const std::shared_ptr<Request>& request,
    std::optional<std::string> poison_reason) {
    if (request->status == RequestStatus::Terminated) {
        return;
    }

    KMM_ASSERT(request->status == RequestStatus::Ready);
    auto buffer = request->buffer;

    unlock_buffer_for_request(buffer, request);

    if (request->device_id != DeviceId(0)) {
        decrement_buffer_users(request->device_id, buffer);
    }

    decrement_buffer_users(DeviceId(0), buffer);

    buffer->num_requests_active -= 1;
    request->status = RequestStatus::Terminated;
}

const Allocation* MemoryManager::view_buffer(const std::shared_ptr<Request>& request) {
    KMM_ASSERT(request->status == RequestStatus::Ready);
    auto& entry = request->buffer->entries.at(request->device_id);
    return entry.data->get();
}

PollResult MemoryManager::poll_requests(
    const std::shared_ptr<Request>* begin,
    const std::shared_ptr<Request>* end) {
    PollResult result = PollResult::Ready;

    for (auto it = begin; it != end; it++) {
        if (poll_request(*it) == PollResult::Pending) {
            result = PollResult::Pending;
        }
    }

    return result;
}

PollResult MemoryManager::poll_request(const std::shared_ptr<Request>& request) {
    while (true) {
        auto status = request->status;

        if (status == RequestStatus::Init) {
            request->status = RequestStatus::SubmitHostAllocation;

        } else if (status == RequestStatus::SubmitHostAllocation) {
            if (submit_request_to_resource(DeviceId(0), request) == PollResult::Pending) {
                request->status = RequestStatus::PollHostAllocation;
                return PollResult::Pending;
            }

            request->status = RequestStatus::SubmitDeviceAllocation;

        } else if (status == RequestStatus::PollHostAllocation) {
            if (poll_resource(DeviceId(0), request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            request->status = RequestStatus::SubmitDeviceAllocation;
        } else if (status == RequestStatus::SubmitDeviceAllocation) {
            if (request->device_id != DeviceId(0)) {
                if (submit_request_to_resource(request->device_id, request)
                    == PollResult::Pending) {
                    request->status = RequestStatus::PollDeviceAllocation;
                    return PollResult::Pending;
                }
            }

            request->status = RequestStatus::RequestAccess;
        } else if (status == RequestStatus::PollDeviceAllocation) {
            if (poll_resource(request->device_id, request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            request->status = RequestStatus::RequestAccess;

        } else if (status == RequestStatus::RequestAccess) {
            if (submit_buffer_lock(request) == PollResult::Pending) {
                request->status = RequestStatus::PollAccess;
                return PollResult::Pending;
            }

            request->status = RequestStatus::PollData;

        } else if (status == RequestStatus::PollAccess) {
            if (poll_buffer_lock(request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            request->status = RequestStatus::PollData;

        } else if (status == RequestStatus::PollData) {
            if (poll_buffer_data(request) == PollResult::Pending) {
                return PollResult::Pending;
            }

            if (request->writable) {
                if (poll_buffer_exclusive(request) == PollResult::Pending) {
                    return PollResult::Pending;
                }
            }

            request->status = RequestStatus::Ready;

        } else if (status == RequestStatus::WaitingForTransfer) {
            // Nothing to do, wait for transfer to complete
            return PollResult::Pending;

        } else if (status == RequestStatus::Ready || status == RequestStatus::Terminated) {
            return PollResult::Ready;

        } else {
            throw std::runtime_error("invalid status");
        }
    }
}

PollResult MemoryManager::poll_buffer_data(const std::shared_ptr<Request>& request) {
    auto buffer = request->buffer;
    auto& entry = buffer->entries.at(request->device_id);

    if (entry.status == Entry::Status::IncomingTransfer) {
        entry.incoming_transfer.requests.push_back(request);
        request->status = RequestStatus::WaitingForTransfer;
        return PollResult::Pending;
    }

    if (entry.status == Entry::Status::Valid) {
        return PollResult::Ready;
    }

    bool found_valid = false;
    auto src_id = DeviceId::invalid();
    auto dst_id = DeviceId(request->device_id);

    if (request->device_id == DeviceId(0)) {
        for (uint8_t i = 1; i < MAX_DEVICES; i++) {
            if (buffer->entries[i].status == Entry::Status::Valid) {
                found_valid = true;
                src_id = DeviceId(i);
                break;
            }
        }
    } else {
        // First, check if it is possible to copy host to device
        if (buffer->entries[0].status == Entry::Status::Valid) {
            found_valid = true;
            src_id = DeviceId(0);
        }

        // Next, check if it is possible to copy device to device
        if (!found_valid) {
            for (uint8_t i = 1; i < MAX_DEVICES; i++) {
                if (buffer->entries[i].status == Entry::Status::Valid
                    && m_memory->is_copy_possible(DeviceId(i), dst_id)) {
                    found_valid = true;
                    src_id = DeviceId(i);
                    break;
                }
            }
        }

        // Next, check if the host is waiting for a transfer
        if (buffer->entries[0].status == Entry::Status::IncomingTransfer) {
            buffer->entries[0].incoming_transfer.requests.push_back(request);
            request->status = RequestStatus::WaitingForTransfer;
            return PollResult::Pending;
        }

        // Finally, check if it is possible to copy device to host
        if (!found_valid) {
            for (uint8_t i = 1; i < MAX_DEVICES; i++) {
                if (buffer->entries[i].status == Entry::Status::Valid) {
                    found_valid = true;
                    src_id = DeviceId(i);
                    dst_id = DeviceId(0);
                    break;
                }
            }
        }
    }

    if (found_valid) {
        auto& transfer = this->initiate_transfer(src_id, dst_id, buffer);
        transfer.requests.push_back(request);

        request->status = RequestStatus::WaitingForTransfer;

        return PollResult::Pending;
    } else {
        // No valid entry found, we just set the entry to valid now
        entry.status = Entry::Status::Valid;
        return PollResult::Ready;
    }
}

PollResult MemoryManager::poll_buffer_exclusive(const std::shared_ptr<Request>& request) {
    KMM_ASSERT(request->writable);

    auto buffer = request->buffer;
    KMM_ASSERT(buffer->entries.at(request->device_id).status == Entry::Status::Valid);

    for (uint8_t device_id = 0; device_id < MAX_DEVICES; device_id++) {
        auto& entry = buffer->entries[device_id];

        if (request->device_id == device_id) {
            continue;
        }

        if (entry.status == Entry::Status::IncomingTransfer) {
            entry.incoming_transfer.requests.push_back(request);
            request->status = RequestStatus::WaitingForTransfer;
            return PollResult::Pending;
        } else if (entry.status == Entry::Status::Valid) {
            entry.status = Entry::Status::Invalid;
        } else {
            KMM_ASSERT(entry.status == Entry::Status::Invalid);
        }
    }

    return PollResult::Ready;
}

MemoryManager::DataTransfer& MemoryManager::initiate_transfer(
    DeviceId src_id,
    DeviceId dst_id,
    BufferState* buffer) {
    auto& src_entry = buffer->entries[src_id];
    auto& dst_entry = buffer->entries[dst_id];

    KMM_ASSERT(src_entry.status == Entry::Status::Valid);
    KMM_ASSERT(dst_entry.status == Entry::Status::Invalid);

    dst_entry.status = Entry::Status::IncomingTransfer;
    dst_entry.incoming_transfer = DataTransfer {.src_id = src_id, .requests = {}, .devices = {}};

    auto completion = std::make_unique<CompletionImpl>(*this, buffer->id, dst_id);

    this->m_memory->copy_async(
        src_id,
        src_entry.data.value().get(),
        dst_id,
        dst_entry.data.value().get(),
        buffer->num_bytes,
        std::move(completion));

    return dst_entry.incoming_transfer;
}

PollResult MemoryManager::submit_buffer_lock(const std::shared_ptr<Request>& request) const {
    auto buffer = request->buffer;
    if (buffer->waiters.is_empty() && try_lock_buffer_for_request(buffer, request)) {
        return PollResult::Ready;
    } else {
        buffer->waiters.push_back(request);
        return PollResult::Pending;
    }
}

PollResult MemoryManager::poll_buffer_lock(const std::shared_ptr<Request>& request) const {
    auto buffer = request->buffer;

    if (buffer->waiters.front() == request && try_lock_buffer_for_request(buffer, request)) {
        buffer->waiters.pop_front();

        // TODO: Maybe check if all requests are ready instead of just waking up the first one
        if (!buffer->waiters.is_empty()) {
            buffer->waiters.front()->waker->wakeup();
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
        buffer->waiters.front()->waker->wakeup();
    }
}

PollResult MemoryManager::submit_request_to_resource(
    DeviceId device_id,
    const std::shared_ptr<Request>& request) {
    auto& buffer = request->buffer;
    auto& resource = m_resources.at(device_id);
    auto& buffer_entry = buffer->entries.at(device_id);

    if (!resource->waiters.is_empty() || !buffer_entry.data.has_value()) {
        resource->waiters.push_back(request);
        return poll_resource(device_id, request);
    }

    // Remove from LRU
    increment_buffer_users(device_id, buffer);
    return PollResult::Ready;
}

PollResult MemoryManager::poll_resource(
    DeviceId device_id,
    const std::shared_ptr<Request>& request) {
    auto& buffer = request->buffer;
    auto& resource = m_resources.at(device_id);
    auto& buffer_entry = buffer->entries.at(device_id);

    if (resource->waiters.front() != request) {
        return PollResult::Pending;
    }

    while (!buffer_entry.data.has_value()) {
        if (auto alloc = m_memory->allocate(device_id, buffer->num_bytes)) {
            buffer_entry.data = std::move(alloc);
            continue;
        }

        if (auto victim = resource->lru_oldest) {
            if (evict_buffer(device_id, victim) == PollResult::Pending) {
                return PollResult::Pending;
            }

            continue;
        }

        // TODO: Check for out of memory

        return PollResult::Pending;
    }

    // Remove from LRU
    increment_buffer_users(device_id, buffer);

    resource->waiters.pop_front();
    if (!resource->waiters.is_empty()) {
        resource->waiters.front()->waker->wakeup();
    }

    return PollResult::Ready;
}

void MemoryManager::decrement_buffer_users(DeviceId device_id, MemoryManager::BufferState* buffer) {
    auto& resource = m_resources.at(device_id);
    auto& buffer_links = buffer->links.at(device_id);
    buffer_links.num_users--;

    if (buffer_links.num_users == 0) {
        if (resource->lru_oldest) {
            auto& front = resource->lru_oldest;
            auto& back = resource->lru_oldest->links.at(device_id).prev;

            back->links.at(device_id).next = buffer;
            buffer_links.prev = back;

            buffer_links.next = front;
            front->links.at(device_id).prev = buffer;
        } else {
            resource->lru_oldest = buffer;
            buffer_links.prev = buffer;
            buffer_links.next = buffer;

            if (!resource->waiters.is_empty()) {
                resource->waiters.front()->waker->wakeup();
            }
        }
    }
}

void MemoryManager::increment_buffer_users(DeviceId device_id, MemoryManager::BufferState* buffer) {
    auto& resource = m_resources.at(device_id);
    auto& buffer_links = buffer->links.at(device_id);
    buffer_links.num_users++;

    if (buffer_links.num_users == 1) {
        remove_buffer_from_lru(device_id, buffer);
    }
}

void MemoryManager::complete_transfer(PhysicalBufferId buffer_id, DeviceId dst_id) {
    auto& buffer = m_buffers.at(buffer_id);
    auto& entry = buffer->entries.at(dst_id);

    KMM_ASSERT(entry.status == Entry::Status::IncomingTransfer);
    entry.status = Entry::Status::Valid;

    while (!entry.incoming_transfer.requests.is_empty()) {
        auto req = entry.incoming_transfer.requests.pop_front();

        if (req->status == RequestStatus::WaitingForTransfer) {
            req->status = RequestStatus::PollData;
            req->waker->wakeup();
        } else {
        }
    }

    while (!entry.incoming_transfer.devices.empty()) {
        KMM_TODO();
        //auto device = entry.incoming_transfer.devices.back();
        //entry.incoming_transfer.devices.pop_back();

        // TODO: wakeup resource
    }

    entry.incoming_transfer = DataTransfer {};
}

PollResult MemoryManager::evict_buffer(DeviceId device_id, MemoryManager::BufferState* buffer) {
    auto& entry = buffer->entries.at(device_id);
    auto& buffer_links = buffer->links.at(device_id);

    KMM_ASSERT(entry.data.has_value());
    KMM_ASSERT(buffer_links.num_users == 0);

    if (entry.status == Entry::Status::IncomingTransfer) {
        entry.incoming_transfer.devices.push_back(device_id);
        return PollResult::Pending;
    }

    for (auto& other_entry : buffer->entries) {
        if (other_entry.status == Entry::Status::IncomingTransfer
            && other_entry.incoming_transfer.src_id == device_id) {
            entry.incoming_transfer.devices.push_back(device_id);
            return PollResult::Pending;
        }
    }

    auto alloc = std::move(*entry.data);
    entry.data = nullptr;
    entry.status = Entry::Status::Invalid;

    remove_buffer_from_lru(device_id, buffer);

    m_memory->deallocate(device_id, std::move(alloc));
    return PollResult::Ready;
}

void MemoryManager::remove_buffer_from_lru(DeviceId device_id, MemoryManager::BufferState* buffer) {
    auto& buffer_links = buffer->links.at(device_id);
    auto& resource = this->m_resources.at(device_id);
    auto prev = buffer_links.prev;
    auto next = buffer_links.next;

    if (prev != next) {
        next->links.at(device_id).prev = prev;
        prev->links.at(device_id).next = next;
        resource->lru_oldest = next;
    } else {
        resource->lru_oldest = nullptr;
    }

    buffer_links.prev = nullptr;
    buffer_links.next = nullptr;
}

MemoryManager::CompletionImpl::CompletionImpl(
    MemoryManager& manager,
    PhysicalBufferId buffer_id,
    DeviceId dst_id) :
    m_manager(manager.shared_from_this()),
    m_buffer_id(buffer_id),
    m_dst_id(dst_id) {}

void MemoryManager::CompletionImpl::complete() {
    if (m_manager) {
        std::exchange(m_manager, {})->complete_transfer(m_buffer_id, m_dst_id);
    }
}

MemoryManager::CompletionImpl::~CompletionImpl() {
    if (m_manager) {
        std::cerr << "ERROR: Transfer for buffer " << m_buffer_id.get()
                  << " has been deleted without "
                     "completing it first, this will leak memory!"
                  << std::endl;
    }
}

}  // namespace kmm