#include <chrono>
#include <cstring>

#include "fmt/format.h"
#include "spdlog/spdlog.h"

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
    GPUEventSet allocation_event;
    GPUEventSet epoch_event;
    GPUEventSet write_events;
    GPUEventSet access_events;
};

struct HostEntry: public BufferEntry {
    void* data = nullptr;
};

struct DeviceEntry: public BufferEntry {
    GPUdeviceptr data = 0;

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

MemoryManager::MemoryManager(std::unique_ptr<MemorySystem> allocator) :
    m_allocator(std::move(allocator)),
    m_devices(std::make_unique<Device[]>(MAX_DEVICES)) {}

MemoryManager::~MemoryManager() {
    KMM_ASSERT(m_buffers.empty());
}

void MemoryManager::make_progress() {
    m_allocator->make_progress();
}

bool MemoryManager::is_idle(GPUStreamManager& streams) const {
    bool result = true;

    for (const auto& buffer : m_buffers) {
        for (auto& e : buffer->device_entry) {
            result &= streams.is_ready(e.access_events);
            result &= streams.is_ready(e.write_events);
            result &= streams.is_ready(e.epoch_event);
            result &= streams.is_ready(e.allocation_event);
        }

        auto& e = buffer->host_entry;
        result &= streams.is_ready(e.access_events);
        result &= streams.is_ready(e.write_events);
        result &= streams.is_ready(e.epoch_event);
        result &= streams.is_ready(e.allocation_event);
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

    check_consistency();
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

bool MemoryManager::poll_request(Request& req, GPUEventSet& deps_out) {
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

        deps_out.insert(buffer.entry(memory_id).allocation_event);
        req.status = Request::Status::Allocated;
    }

    if (req.status == Request::Status::Allocated) {
        if (!try_lock_access(memory_id, buffer, req, deps_out)) {
            return false;
        }

        req.status = Request::Status::Ready;
    }

    if (req.status == Request::Status::Ready) {
        return true;
    }

    throw std::runtime_error("cannot poll a deleted request");
}

void MemoryManager::release_request(std::shared_ptr<Request> req, GPUEvent event) {
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
    } else {
        device.lru_oldest = &buffer;
    }

    device_entry.lru_older = prev;
    device_entry.lru_newer = nullptr;
    device.lru_newest = &buffer;

    check_consistency();
}

void MemoryManager::remove_from_lru(DeviceId device_id, Buffer& buffer) {
    auto& device_entry = buffer.device_entry[device_id];
    auto& device = m_devices[device_id];

    KMM_ASSERT(device_entry.is_allocated);
    KMM_ASSERT(device_entry.num_allocation_locks == 0);

    auto* prev = std::exchange(device_entry.lru_newer, nullptr);
    auto* next = std::exchange(device_entry.lru_older, nullptr);

    if (prev != nullptr) {
        prev->device_entry[device_id].lru_older = next;
    } else {
        device.lru_newest = next;
    }

    if (next != nullptr) {
        next->device_entry[device_id].lru_newer = prev;
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
    spdlog::trace(
        "allocate {} bytes from device {} for buffer {}",
        buffer.layout.size_in_bytes,
        device_id,
        (void*)&buffer);

    void* ptr_out;
    GPUEventSet events;
    auto result = m_allocator->allocate(device_id, buffer.layout.size_in_bytes, ptr_out, events);

    if (!result) {
        return false;
    }

    device_entry.data = (GPUdeviceptr)ptr_out;
    device_entry.is_allocated = true;
    device_entry.is_valid = false;
    device_entry.allocation_event = events;
    device_entry.epoch_event = events;
    device_entry.access_events = events;
    device_entry.write_events = events;
    return true;
}

void MemoryManager::deallocate_device_async(DeviceId device_id, Buffer& buffer) {
    auto& device = m_devices[device_id];
    auto& device_entry = buffer.device_entry[device_id];

    if (!device_entry.is_allocated) {
        return;
    }

    size_t size_in_bytes = buffer.layout.size_in_bytes;
    spdlog::trace(
        "free {} bytes for buffer {} on device {} (dependencies={})",
        size_in_bytes,
        (void*)&buffer,
        device_id,
        device_entry.access_events);

    KMM_ASSERT(device_entry.num_allocation_locks == 0);
    KMM_ASSERT(buffer.access_head == nullptr);
    KMM_ASSERT(buffer.access_current == nullptr);
    KMM_ASSERT(buffer.access_tail == nullptr);

    m_allocator->deallocate(
        device_id,
        (void*)device_entry.data,
        size_in_bytes,
        std::move(device_entry.access_events));
    remove_from_lru(device_id, buffer);

    device_entry.is_allocated = false;
    device_entry.is_valid = false;
    device_entry.allocation_event.clear();
    device_entry.epoch_event.clear();
    device_entry.write_events.clear();
    device_entry.access_events.clear();
    device_entry.data = 0;

    check_consistency();
}

void MemoryManager::allocate_host(Buffer& buffer) {
    auto& host_entry = buffer.host_entry;
    size_t size_in_bytes = buffer.layout.size_in_bytes;

    KMM_ASSERT(host_entry.is_allocated == false);
    KMM_ASSERT(host_entry.num_allocation_locks == 0);

    spdlog::trace("allocate {} bytes from host", size_in_bytes, (void*)&buffer);

    GPUEventSet events;
    void* ptr;
    bool success = m_allocator->allocate(MemoryId::host(), size_in_bytes, ptr, events);

    if (!success) {
        throw std::runtime_error("could not allocate, out of host memory");
    }

    host_entry.data = ptr;
    host_entry.is_allocated = true;
    host_entry.is_valid = false;
    host_entry.allocation_event = events;
    host_entry.epoch_event = events;
    host_entry.access_events = events;
    host_entry.write_events = events;
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

    size_t size_in_bytes = buffer.layout.size_in_bytes;
    spdlog::trace(
        "free {} bytes on host (dependencies={})",
        size_in_bytes,
        (void*)&buffer,
        host_entry.access_events);

    m_allocator->deallocate(
        MemoryId::host(),
        host_entry.data,
        size_in_bytes,
        std::move(host_entry.access_events));

    host_entry.data = nullptr;
    host_entry.is_allocated = false;
    host_entry.is_valid = false;
    host_entry.epoch_event.clear();
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
    spdlog::trace(
        "lock allocation on host of buffer {} for request {}",
        (void*)&buffer,
        (void*)&req);
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
            // Try to allocate
            if (try_allocate_device_async(device_id, buffer)) {
                break;
            }

            // No memory available, try to free memory
            if (try_free_device_memory(device_id)) {
                continue;
            }

            if (is_out_of_memory(device_id, req)) {
                throw std::runtime_error(fmt::format(
                    "cannot allocate {} bytes on device {}, out of memory",
                    buffer.layout.size_in_bytes,
                    device_id.get()));
            }

            return false;
        }
    } else {
        if (device_entry.num_allocation_locks == 0) {
            remove_from_lru(device_id, buffer);
        }
    }

    req.allocation_acquired = true;
    device.allocation_current = req.allocation_next;
    device_entry.num_allocation_locks++;

    spdlog::trace(
        "lock allocation on device {} of buffer {} for request {}",
        device_id,
        (void*)&buffer,
        (void*)&req);

    return true;
}

void MemoryManager::unlock_allocation_host(Buffer& buffer, Request& req) {
    auto& host_entry = buffer.host_entry;

    KMM_ASSERT(host_entry.is_allocated);
    KMM_ASSERT(host_entry.is_valid);
    KMM_ASSERT(host_entry.num_allocation_locks > 0);

    host_entry.num_allocation_locks--;
    spdlog::trace(
        "unlock allocation on host of buffer {} for request {}",
        (void*)&buffer,
        (void*)&req);
}

void MemoryManager::unlock_allocation_device(DeviceId device_id, Buffer& buffer, Request& req) {
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(device_entry.is_allocated);
    KMM_ASSERT(device_entry.is_valid);
    KMM_ASSERT(device_entry.num_allocation_locks > 0);

    device_entry.num_allocation_locks--;
    spdlog::trace(
        "unlock allocation on device {} of buffer {} for request {}",
        device_id,
        (void*)&buffer,
        (void*)&req);

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
        spdlog::trace(
            "access to buffer {} was granted to request {} (memory={}, mode={})",
            (void*)&buffer,
            (void*)req,
            req->memory_id,
            req->mode);

        buffer.access_current = buffer.access_current->access_next;
    }
}

void MemoryManager::unlock_access(
    MemoryId memory_id,
    Buffer& buffer,
    Request& req,
    GPUEvent event) {
    spdlog::trace(
        "access to buffer {} was revoked from request {} (memory={}, mode={}, GPU event={})",
        (void*)&buffer,
        (void*)&req,
        req.memory_id,
        req.mode,
        event);

    bool is_writer = req.mode != AccessMode::Read;
    BufferEntry& entry = buffer.entry(memory_id);

    entry.access_events.insert(event);

    if (is_writer) {
        entry.write_events.insert(event);

        if (req.mode == AccessMode::Exclusive) {
            entry.epoch_event.insert(event);
        }
    }
}

bool MemoryManager::try_lock_access(
    MemoryId memory_id,
    Buffer& buffer,
    Request& req,
    GPUEventSet& deps_out) {
    if (!req.access_acquired) {
        poll_access_queue(buffer);

        if (!req.access_acquired) {
            return false;
        }
    }

    bool is_writer = req.mode != AccessMode::Read;
    auto& entry = buffer.entry(memory_id);

    make_entry_valid(memory_id, buffer, deps_out);

    if (is_writer) {
        if (!memory_id.is_host()) {
            buffer.host_entry.is_valid = false;
            deps_out.insert(buffer.host_entry.access_events);
        }

        // Invalidate all _other_ device entries
        for (auto& peer_entry : buffer.device_entry) {
            if (&peer_entry != &entry) {
                peer_entry.is_valid = false;
                deps_out.insert(peer_entry.access_events);
            }
        }

        if (req.mode == AccessMode::Exclusive) {
            deps_out.insert(entry.access_events);
        }
    }

    return true;
}

void MemoryManager::make_entry_valid(MemoryId memory_id, Buffer& buffer, GPUEventSet& deps_out) {
    auto& entry = buffer.entry(memory_id);

    KMM_ASSERT(entry.is_allocated);

    if (!entry.is_valid) {
        if (memory_id.is_host()) {
            if (auto src_id = find_valid_device_entry(buffer)) {
                deps_out.insert(copy_d2h(*src_id, buffer));
                return;
            }
        } else {
            auto device_id = memory_id.as_device();

            if (buffer.host_entry.is_valid) {
                deps_out.insert(copy_h2d(device_id, buffer));
                return;
            }

            if (auto src_id = find_valid_device_entry(buffer)) {
                if (!buffer.host_entry.is_allocated) {
                    allocate_host(buffer);
                }

                deps_out.insert(copy_h2d(device_id, buffer));
                return;
            }
        }
    }

    entry.is_valid = true;
    deps_out.insert(entry.epoch_event);
}

GPUEvent MemoryManager::copy_h2d(DeviceId device_id, Buffer& buffer) {
    spdlog::trace(
        "copy {} bytes from host to device {} for buffer {}",
        buffer.layout.size_in_bytes,
        device_id,
        (void*)&buffer);

    auto& device = m_devices[device_id];
    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(host_entry.is_allocated && device_entry.is_allocated);
    KMM_ASSERT(host_entry.is_valid && !device_entry.is_valid);

    GPUEventSet deps;
    deps.insert(device_entry.access_events);
    deps.insert(host_entry.write_events);

    auto event = m_allocator->copy_host_to_device(
        device_id,
        host_entry.data,
        device_entry.data,
        buffer.layout.size_in_bytes,
        std::move(deps));

    host_entry.access_events.insert(event);
    device_entry.epoch_event = {event};
    device_entry.access_events = {event};
    device_entry.write_events = {event};

    device_entry.is_valid = true;
    return event;
}

GPUEvent MemoryManager::copy_d2h(DeviceId device_id, Buffer& buffer) {
    spdlog::trace(
        "copy {} bytes from device to host {} for buffer {}",
        buffer.layout.size_in_bytes,
        device_id,
        (void*)&buffer);

    auto& device = m_devices[device_id];
    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(host_entry.is_allocated && device_entry.is_allocated);
    KMM_ASSERT(!host_entry.is_valid && device_entry.is_valid);

    GPUEventSet deps;
    deps.insert(device_entry.write_events);
    deps.insert(host_entry.access_events);

    auto event = m_allocator->copy_device_to_host(
        device_id,
        device_entry.data,
        host_entry.data,
        buffer.layout.size_in_bytes,
        std::move(deps));

    device_entry.access_events.insert(event);
    host_entry.epoch_event.insert(event);
    host_entry.access_events.insert(event);
    host_entry.write_events.insert(event);

    host_entry.is_valid = true;
    return event;
}

bool MemoryManager::is_out_of_memory(DeviceId device_id, Request& req) {
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

    spdlog::error(
        "out of memory for device {}, failed to allocate {} bytes of buffer {} for request {}",
        device_id,
        req.buffer->layout.size_in_bytes,
        (void*)req.buffer.get(),
        (void*)&req);

    spdlog::error("following buffers are currently allocated: ");

    for (const auto& buffer : m_buffers) {
        auto entry = buffer->entry(device_id);

        if (entry.is_allocated) {
            spdlog::error(
                " - buffer {} ({} bytes, {} allocation locks)",
                (void*)buffer.get(),
                buffer->layout.size_in_bytes,
                entry.num_allocation_locks);
        }
    }

    return true;
}

void MemoryManager::check_consistency() const {}

// This is here to check the consistency of the data structures while debugging.
/*
void MemoryManager::check_consistency() const {
    for (size_t i = 0; i < MAX_DEVICES; i++) {
        std::unordered_set<Buffer*> available_buffers;
        auto id = DeviceId(i);
        auto& device = m_devices[i];

        auto* prev = (Buffer*) nullptr;
        auto* current = device.lru_oldest;

        while (current != nullptr) {
            available_buffers.insert(current);
            auto& entry = current->device_entry[id];

            KMM_ASSERT(entry.num_allocation_locks == 0);
            KMM_ASSERT(entry.lru_older == prev);

            prev = current;
            current = entry.lru_newer;
        }

        KMM_ASSERT(prev == device.lru_newest);

        for (const auto& buffer: m_buffers) {
            auto& entry = buffer->device_entry[id];

            if (entry.is_allocated && entry.num_allocation_locks == 0) {
                KMM_ASSERT(available_buffers.find(buffer.get()) != available_buffers.end());
            }
        }
    }
}
*/

}  // namespace kmm