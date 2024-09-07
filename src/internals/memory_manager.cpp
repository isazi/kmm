#include <chrono>
#include <cstring>

#include "fmt/format.h"

#include "kmm/internals/memory_manager.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

struct BufferEntry {
    bool is_allocated = false;
    bool is_valid = false;

    size_t num_allocation_locks = 0;

    // let `<=` the happens-before operator. Then it should ALWAYS hold that
    // * allocation_event <= epoch_event
    // * epoch_event <= write_events
    // * write_events <= access_events
    CudaEvent allocation_event;
    CudaEvent epoch_event;
    CudaEventSet write_events;
    CudaEventSet access_events;
};

struct HostEntry: public BufferEntry {
    void* data = nullptr;
};

struct DeviceEntry: public BufferEntry {
    CUdeviceptr data = 0;

    MemoryManager::Buffer* lru_older = nullptr;
    MemoryManager::Buffer* lru_newer = nullptr;
};

struct MemoryManager::Buffer {
    KMM_NOT_COPYABLE_OR_MOVABLE(Buffer);

  public:
    Buffer(BufferLayout layout) : id(id), layout(layout) {}

    BufferId id;
    BufferLayout layout;
    HostEntry host_entry;
    DeviceEntry device_entry[MAX_DEVICES];
    size_t num_requests_active = 0;
    bool is_deleted = false;

    Request* access_head = nullptr;
    Request* access_current = nullptr;
    Request* access_tail = nullptr;

    KMM_INLINE BufferEntry& entry(MemoryId memory_id) {
        if (memory_id.is_host()) {
            return host_entry;
        } else {
            return device_entry[memory_id.as_device()];
        }
    }
};

struct MemoryManager::Transaction {
    uint64_t id;
    std::shared_ptr<Transaction> parent;
    std::chrono::system_clock::time_point created_at = std::chrono::system_clock::now();
};

struct MemoryManager::Request {
    KMM_NOT_COPYABLE_OR_MOVABLE(Request);

  public:
    Request(
        std::shared_ptr<Buffer> buffer,
        MemoryId memory_id,
        AccessMode mode,
        std::shared_ptr<Transaction> parent) :
        buffer(std::move(buffer)),
        memory_id(memory_id),
        mode(mode),
        parent(std::move(parent)) {}

    enum struct Status { Init, Allocated, Locked, Ready, Deleted };
    Status status = Status::Init;
    std::shared_ptr<Buffer> buffer;
    MemoryId memory_id;
    AccessMode mode;
    std::shared_ptr<Transaction> parent;
    std::chrono::system_clock::time_point created_at = std::chrono::system_clock::now();

    bool allocation_acquired = false;
    Request* allocation_next = nullptr;
    Request* allocation_prev = nullptr;

    bool access_acquired = false;
    Request* access_next = nullptr;
    Request* access_prev = nullptr;
};

struct MemoryManager::Device {
    KMM_NOT_COPYABLE_OR_MOVABLE(Device);

  public:
    Buffer* lru_oldest = nullptr;
    Buffer* lru_newest = nullptr;

    Request* allocation_head = nullptr;
    Request* allocation_current = nullptr;
    Request* allocation_tail = nullptr;

    Device() = default;
};

MemoryManager::MemoryManager(
    std::shared_ptr<CudaStreamManager> streams,
    std::unique_ptr<MemoryAllocator> allocator) :
    m_streams(streams),
    m_allocator(std::move(allocator)),
    m_devices(std::make_unique<Device[]>(MAX_DEVICES)) {}

MemoryManager::~MemoryManager() {
    KMM_ASSERT(m_buffers.empty());
}

void MemoryManager::make_progress() {
    m_allocator->make_progress();
}

bool MemoryManager::is_idle() const {
    bool result = true;

    for (const auto& buffer : m_buffers) {
        for (auto& e : buffer->device_entry) {
            result &= m_streams->is_ready(e.access_events);
            result &= m_streams->is_ready(e.write_events);
            result &= m_streams->is_ready(e.epoch_event);
            result &= m_streams->is_ready(e.allocation_event);
        }

        auto& e = buffer->host_entry;
        result &= m_streams->is_ready(e.access_events);
        result &= m_streams->is_ready(e.write_events);
        result &= m_streams->is_ready(e.epoch_event);
        result &= m_streams->is_ready(e.allocation_event);
    }

    return result;
}

std::shared_ptr<MemoryManager::Buffer> MemoryManager::create_buffer(BufferLayout layout) {
    // Size cannot be zero
    if (layout.size_in_bytes == 0) {
        layout.size_in_bytes = 1;
    }

    // Make sure alignment is power of two and size is multiple of alignment
    layout.alignment = round_up_to_power_of_two(layout.alignment);
    layout.size_in_bytes = round_up_to_multiple(layout.size_in_bytes, layout.alignment);

    auto buffer = std::make_shared<Buffer>(layout);
    m_buffers.emplace(buffer);
    return buffer;
}

void MemoryManager::delete_buffer(std::shared_ptr<Buffer> buffer) {
    if (buffer->is_deleted) {
        return;
    }

    KMM_ASSERT(buffer->num_requests_active == 0);
    buffer->is_deleted = true;
    m_buffers.erase(buffer);

    deallocate_host(*buffer);

    for (size_t i = 0; i < MAX_DEVICES; i++) {
        deallocate_device_async(DeviceId(i), *buffer);
    }
}

std::shared_ptr<MemoryManager::Transaction> MemoryManager::create_transaction(
    std::shared_ptr<Transaction> parent) {
    auto id = m_next_transaction_id++;
    return std::make_shared<Transaction>(Transaction {id, parent});
}

std::shared_ptr<MemoryManager::Request> MemoryManager::create_request(
    std::shared_ptr<Buffer> buffer,
    MemoryId memory_id,
    AccessMode mode,
    std::shared_ptr<Transaction> parent) {
    KMM_ASSERT(!buffer->is_deleted);
    buffer->num_requests_active++;

    auto req = std::make_shared<Request>(buffer, memory_id, mode, parent);
    m_active_requests.insert(req);

    if (memory_id.is_device()) {
        add_to_allocation_queue(memory_id.as_device(), *req);
    }

    add_to_buffer_access_queue(*buffer, *req);
    return req;
}

bool MemoryManager::poll_request(Request& req, CudaEventSet& deps_out) {
    auto& buffer = *req.buffer;
    auto memory_id = req.memory_id;

    if (req.status == Request::Status::Init) {
        if (memory_id.is_host()) {
            lock_allocation_host(buffer, req);
        } else {
            if (!lock_allocation_device(memory_id.as_device(), buffer, req)) {
                return false;
            }
        }

        deps_out.insert(*m_streams, buffer.entry(memory_id).allocation_event);
        req.status = Request::Status::Allocated;
    }

    if (req.status == Request::Status::Allocated) {
        if (!try_lock_access(buffer, req)) {
            return false;
        }

        req.status = Request::Status::Locked;
    }

    if (req.status == Request::Status::Locked) {
        initiate_transfers(req.memory_id, buffer, req, deps_out);
        req.status = Request::Status::Ready;
    }

    if (req.status == Request::Status::Ready) {
        return true;
    }

    throw std::runtime_error("cannot poll a deleted request");
}

void MemoryManager::release_request(std::shared_ptr<Request> req, CudaEvent event) {
    auto memory_id = req->memory_id;
    auto& buffer = *req->buffer;
    auto status = std::exchange(req->status, Request::Status::Deleted);

    if (status == Request::Status::Ready) {
        status = Request::Status::Locked;
    }

    if (status == Request::Status::Locked) {
        unlock_access(memory_id, buffer, *req, event);
        status = Request::Status::Allocated;
    }

    if (status == Request::Status::Allocated) {
        if (memory_id.is_host()) {
            unlock_allocation_host(buffer, *req);
        } else {
            unlock_allocation_device(memory_id.as_device(), buffer, *req);
        }

        status = Request::Status::Init;
    }

    if (status == Request::Status::Init) {
        remove_from_buffer_access_queue(buffer, *req);

        if (memory_id.is_device()) {
            remove_from_allocation_queue(memory_id.as_device(), *req);
        }

        buffer.num_requests_active--;
        m_active_requests.erase(req);
    }
}

void MemoryManager::add_to_allocation_queue(DeviceId device_id, Request& req) const {
    auto& device = m_devices[device_id];
    auto* tail = device.allocation_tail;

    if (tail == nullptr) {
        device.allocation_head = &req;
    } else {
        tail->allocation_next = &req;
        req.allocation_prev = tail;
    }

    device.allocation_tail = &req;

    if (device.allocation_current == nullptr) {
        device.allocation_current = &req;
    }
}

void MemoryManager::remove_from_allocation_queue(DeviceId device_id, Request& req) const {
    auto& device = m_devices[device_id];
    auto* prev = std::exchange(req.allocation_prev, nullptr);
    auto* next = std::exchange(req.allocation_next, nullptr);

    if (prev != nullptr) {
        prev->allocation_next = next;
    } else {
        KMM_ASSERT(device.allocation_head == &req);
        device.allocation_head = next;
    }

    if (next != nullptr) {
        next->allocation_prev = prev;
    } else {
        KMM_ASSERT(device.allocation_tail == &req);
        device.allocation_tail = prev;
    }

    if (device.allocation_current == &req) {
        device.allocation_current = next;
    }
}

void MemoryManager::add_to_buffer_access_queue(Buffer& buffer, Request& req) const {
    auto* old_tail = buffer.access_tail;

    if (old_tail == nullptr) {
        buffer.access_head = &req;
    } else {
        old_tail->access_next = &req;
        req.access_prev = old_tail;
    }

    buffer.access_tail = &req;

    if (buffer.access_current == nullptr) {
        buffer.access_current = &req;
    }
}

void MemoryManager::remove_from_buffer_access_queue(Buffer& buffer, Request& req) const {
    auto* prev = std::exchange(req.access_prev, nullptr);
    auto* next = std::exchange(req.access_next, nullptr);

    if (prev != nullptr) {
        prev->access_next = next;
    } else {
        KMM_ASSERT(buffer.access_head == &req);
        buffer.access_head = next;
    }

    if (next != nullptr) {
        next->access_prev = prev;
    } else {
        KMM_ASSERT(buffer.access_tail == &req);
        buffer.access_tail = prev;
    }

    if (buffer.access_current == &req) {
        buffer.access_current = next;
    }

    // Poll queue, release the lock might allow another request to gain access
    if (req.access_acquired) {
        req.access_acquired = false;
        poll_access_queue(buffer);
    }
}

BufferAccessor MemoryManager::get_accessor(Request& req) {
    KMM_ASSERT(req.status == Request::Status::Ready);
    const auto& buffer = *req.buffer;
    void* address;

    if (req.memory_id.is_host()) {
        address = buffer.host_entry.data;
    } else {
        address = reinterpret_cast<void*>(buffer.device_entry[req.memory_id.as_device()].data);
    }

    return BufferAccessor {
        .buffer_id = buffer.id,
        .memory_id = req.memory_id,
        .layout = buffer.layout,
        .is_writable = req.mode != AccessMode::Read,
        .address = address};
}

void MemoryManager::insert_into_lru(DeviceId device_id, Buffer& buffer) {
    auto& device_entry = buffer.device_entry[device_id];
    auto& device = m_devices[device_id];

    KMM_ASSERT(device_entry.is_allocated);
    KMM_ASSERT(device_entry.num_allocation_locks == 0);

    auto* prev = device.lru_newest;
    if (prev != nullptr) {
        prev->device_entry[device_id].lru_newer = &buffer;
    }

    device_entry.lru_older = prev;
    device_entry.lru_newer = nullptr;
    device.lru_newest = &buffer;
}

void MemoryManager::remove_from_lru(DeviceId device_id, Buffer& buffer) {
    auto& device_entry = buffer.device_entry[device_id];
    auto& device = m_devices[device_id];

    KMM_ASSERT(device_entry.is_allocated);
    KMM_ASSERT(device_entry.num_allocation_locks == 0);

    auto* prev = device_entry.lru_newer;
    auto* next = device_entry.lru_older;

    if (prev != nullptr) {
        prev->device_entry[device_id].lru_newer = next;
    } else {
        device.lru_newest = next;
    }

    if (next != nullptr) {
        next->device_entry[device_id].lru_older = prev;
    } else {
        device.lru_oldest = prev;
    }
}

bool MemoryManager::try_free_device_memory(DeviceId device_id) {
    auto& device = m_devices[device_id];

    if (device.lru_oldest == nullptr) {
        return false;
    }

    auto& victim = *device.lru_oldest;

    if (victim.device_entry[device_id].is_valid) {
        bool valid_anywhere = victim.host_entry.is_valid;

        for (size_t i = 0; i < MAX_DEVICES; i++) {
            if (i != device_id) {
                valid_anywhere |= victim.device_entry[i].is_valid;
            }
        }

        if (!valid_anywhere) {
            if (!victim.host_entry.is_allocated) {
                allocate_host(victim);
            }

            copy_d2h(device_id, victim);
        }
    }

    deallocate_device_async(device_id, victim);
    return true;
}

bool MemoryManager::try_allocate_device_async(DeviceId device_id, Buffer& buffer) {
    auto& device = m_devices[device_id];
    auto& device_entry = buffer.device_entry[device_id];

    if (device_entry.is_allocated) {
        return true;
    }

    KMM_ASSERT(device_entry.num_allocation_locks == 0);
    CudaEvent event;

    if (!m_allocator
             ->allocate_device(device_id, buffer.layout.size_in_bytes, device_entry.data, event)) {
        return false;
    }

    device_entry.is_allocated = true;
    device_entry.is_valid = false;
    device_entry.allocation_event = event;
    device_entry.epoch_event = event;
    device_entry.access_events = {event};
    device_entry.write_events = {event};
    return true;
}

void MemoryManager::deallocate_device_async(DeviceId device_id, Buffer& buffer) {
    auto& device = m_devices[device_id];
    auto& device_entry = buffer.device_entry[device_id];

    if (!device_entry.is_allocated) {
        return;
    }

    KMM_ASSERT(device_entry.num_allocation_locks == 0);
    KMM_ASSERT(buffer.access_head == nullptr);
    KMM_ASSERT(buffer.access_current == nullptr);
    KMM_ASSERT(buffer.access_tail == nullptr);

    m_allocator->deallocate_device(
        device_id,
        device_entry.data,
        std::move(device_entry.access_events));
    remove_from_lru(device_id, buffer);

    device_entry.is_allocated = false;
    device_entry.is_valid = false;
    device_entry.allocation_event = CudaEvent {};
    device_entry.epoch_event = CudaEvent {};
    device_entry.write_events.clear();
    device_entry.access_events.clear();
    device_entry.data = 0;
}

void MemoryManager::allocate_host(Buffer& buffer) {
    auto& host_entry = buffer.host_entry;

    KMM_ASSERT(host_entry.is_allocated == false);
    KMM_ASSERT(host_entry.num_allocation_locks == 0);

    host_entry.data = m_allocator->allocate_host(buffer.layout.size_in_bytes);

    host_entry.is_allocated = true;
    host_entry.is_valid = false;
}

void MemoryManager::deallocate_host(Buffer& buffer) {
    auto& host_entry = buffer.host_entry;
    if (!host_entry.is_allocated) {
        return;
    }

    KMM_ASSERT(host_entry.num_allocation_locks == 0);
    KMM_ASSERT(buffer.access_head == nullptr);
    KMM_ASSERT(buffer.access_current == nullptr);
    KMM_ASSERT(buffer.access_tail == nullptr);

    m_allocator->deallocate_host(host_entry.data, std::move(host_entry.access_events));

    host_entry.data = nullptr;
    host_entry.is_allocated = false;
    host_entry.is_valid = false;
    host_entry.epoch_event = CudaEvent {};
    host_entry.write_events.clear();
    host_entry.access_events.clear();
    host_entry.data = nullptr;
}

void MemoryManager::lock_allocation_host(Buffer& buffer, Request& req) {
    auto& host_entry = buffer.host_entry;

    if (!host_entry.is_allocated) {
        allocate_host(buffer);
    }

    host_entry.num_allocation_locks++;
}

bool MemoryManager::lock_allocation_device(DeviceId device_id, Buffer& buffer, Request& req) {
    auto& device = m_devices[device_id];

    if (device.allocation_current != &req) {
        return false;
    }

    KMM_ASSERT(req.allocation_acquired == false);
    KMM_ASSERT(device.allocation_current == &req);

    auto& device_entry = buffer.device_entry[device_id];

    if (!device_entry.is_allocated) {
        while (true) {
            if (try_allocate_device_async(device_id, buffer)) {
                break;
            }

            if (try_free_device_memory(device_id)) {
                continue;
            }

            if (is_out_of_memory(device_id, *req.parent)) {
                throw std::runtime_error(fmt::format(
                    "cannot allocate {} bytes on device {}, out of memory",
                    buffer.layout.size_in_bytes,
                    device_id.get()));
            }

            return false;
        }
    }

    if (device_entry.num_allocation_locks == 0) {
        remove_from_lru(device_id, buffer);
    }

    req.allocation_acquired = true;
    device.allocation_current = req.allocation_next;
    device_entry.num_allocation_locks++;
    return true;
}

void MemoryManager::unlock_allocation_host(Buffer& buffer, Request& req) {
    auto& host_entry = buffer.host_entry;

    KMM_ASSERT(host_entry.is_allocated);
    KMM_ASSERT(host_entry.is_valid);
    KMM_ASSERT(host_entry.num_allocation_locks > 0);

    host_entry.num_allocation_locks--;
}

void MemoryManager::unlock_allocation_device(DeviceId device_id, Buffer& buffer, Request& req) {
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(device_entry.is_allocated);
    KMM_ASSERT(device_entry.is_valid);
    KMM_ASSERT(device_entry.num_allocation_locks > 0);

    device_entry.num_allocation_locks--;

    if (device_entry.num_allocation_locks == 0) {
        insert_into_lru(device_id, buffer);
    }
}

std::optional<DeviceId> MemoryManager::find_valid_device_entry(const Buffer& buffer) const {
    for (size_t device_id = 0; device_id < MAX_DEVICES; device_id++) {
        if (buffer.device_entry[device_id].is_valid) {
            return DeviceId(device_id);
        }
    }

    return std::nullopt;
}

bool MemoryManager::is_access_allowed(const Buffer& buffer, MemoryId memory_id, AccessMode mode)
    const {
    for (auto* it = buffer.access_head; it != buffer.access_current; it = it->access_next) {
        // Two exclusive requests can never be granted access simultaneously
        if (mode == AccessMode::Exclusive || it->mode == AccessMode::Exclusive) {
            return false;
        }

        // Two non-read requests can only be granted simultaneously if operating on the same memory.
        if (mode != AccessMode::Read || it->mode != AccessMode::Read) {
            if (memory_id != it->memory_id) {
                return false;
            }
        }
    }

    return true;
}

void MemoryManager::poll_access_queue(Buffer& buffer) const {
    while (buffer.access_current != nullptr) {
        auto* req = buffer.access_current;

        if (!is_access_allowed(buffer, req->memory_id, req->mode)) {
            return;
        }

        req->access_acquired = true;
        buffer.access_current = buffer.access_current->access_next;
    }
}

bool MemoryManager::try_lock_access(Buffer& buffer, Request& req) {
    poll_access_queue(buffer);
    return req.access_acquired;
}

void MemoryManager::unlock_access(
    MemoryId memory_id,
    Buffer& buffer,
    Request& req,
    CudaEvent event) {
    bool is_writer = req.mode != AccessMode::Read;
    BufferEntry& entry = buffer.entry(memory_id);

    entry.access_events.insert(*m_streams, event);

    if (is_writer) {
        entry.write_events.insert(*m_streams, event);

        if (req.mode == AccessMode::Exclusive) {
            entry.epoch_event = event;
        }
    }
}

void MemoryManager::initiate_transfers(
    MemoryId memory_id,
    Buffer& buffer,
    Request& req,
    CudaEventSet& deps_out) {
    bool is_writer = req.mode != AccessMode::Read;
    auto& entry = buffer.entry(memory_id);

    auto event = make_entry_valid(memory_id, buffer);
    deps_out.insert(*m_streams, event);

    if (is_writer) {
        if (!memory_id.is_host()) {
            buffer.host_entry.is_valid = false;
            deps_out.insert(*m_streams, buffer.host_entry.access_events);
        }

        // Invalidate all _other_ device entries
        for (auto& peer_entry : buffer.device_entry) {
            if (&peer_entry != &entry) {
                peer_entry.is_valid = false;
                deps_out.insert(*m_streams, peer_entry.access_events);
            }
        }

        if (req.mode == AccessMode::Exclusive) {
            deps_out.insert(*m_streams, entry.access_events);
        }
    }
}

CudaEvent MemoryManager::make_entry_valid(MemoryId memory_id, Buffer& buffer) {
    auto& entry = buffer.entry(memory_id);

    KMM_ASSERT(entry.is_allocated);

    if (!entry.is_valid) {
        if (memory_id.is_host()) {
            if (auto src_id = find_valid_device_entry(buffer)) {
                return copy_d2h(*src_id, buffer);
            }
        } else {
            auto device_id = memory_id.as_device();

            if (buffer.host_entry.is_valid) {
                return copy_h2d(device_id, buffer);
            }

            if (auto src_id = find_valid_device_entry(buffer)) {
                if (!buffer.host_entry.is_allocated) {
                    allocate_host(buffer);
                }

                copy_d2h(*src_id, buffer);
                return copy_h2d(device_id, buffer);
            }
        }
    }

    entry.is_valid = true;
    return entry.epoch_event;
}

CudaEvent MemoryManager::copy_h2d(DeviceId device_id, Buffer& buffer) {
    auto& device = m_devices[device_id];
    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(host_entry.is_allocated && device_entry.is_allocated);
    KMM_ASSERT(host_entry.is_valid && !device_entry.is_valid);

    CudaEventSet deps;
    deps.insert(*m_streams, device_entry.access_events);
    deps.insert(*m_streams, host_entry.write_events);

    auto event = m_allocator->copy_host_to_device(
        device_id,
        host_entry.data,
        device_entry.data,
        buffer.layout.size_in_bytes,
        std::move(deps));

    host_entry.access_events.insert(*m_streams, event);
    device_entry.epoch_event = event;
    device_entry.access_events = {event};
    device_entry.write_events = {event};

    device_entry.is_valid = true;
    return event;
}

CudaEvent MemoryManager::copy_d2h(DeviceId device_id, Buffer& buffer) {
    auto& device = m_devices[device_id];
    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(host_entry.is_allocated && device_entry.is_allocated);
    KMM_ASSERT(!host_entry.is_valid && device_entry.is_valid);

    CudaEventSet deps;
    deps.insert(*m_streams, device_entry.write_events);
    deps.insert(*m_streams, host_entry.access_events);

    auto event = m_allocator->copy_device_to_host(
        device_id,
        device_entry.data,
        host_entry.data,
        buffer.layout.size_in_bytes,
        std::move(deps));

    device_entry.access_events.insert(*m_streams, event);
    host_entry.epoch_event = event;
    host_entry.access_events = {event};
    host_entry.write_events = {event};

    host_entry.is_valid = true;
    return event;
}

bool MemoryManager::is_out_of_memory(DeviceId device_id, Transaction& trans) {
    std::unordered_set<const Transaction*> waiting_transactions;
    auto& device = m_devices[device_id];

    // First, iterate over the requests that are waiting for allocation. Mark all the
    // related transactions as `waiting` by adding them to `waiting_transactions`
    for (auto* it = device.allocation_current; it != nullptr; it = it->allocation_next) {
        auto* p = it->parent.get();

        while (p != nullptr) {
            waiting_transactions.insert(p);
            p = p->parent.get();
        }
    }

    // Next, iterate over the requests that have been granted an allocation. If the associated
    // transaction of one of the requests has not been marked as waiting, we are not out of memory
    // since that transaction will release its memory again at some point in the future.
    for (auto* it = device.allocation_head; it != device.allocation_current;
         it = it->allocation_next) {
        auto* p = it->parent.get();

        if (waiting_transactions.find(p) == waiting_transactions.end()) {
            return false;
        }
    }

    return true;
}

}  // namespace kmm