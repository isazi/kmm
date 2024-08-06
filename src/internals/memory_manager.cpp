#include <algorithm>
#include <cstring>
#include <unordered_set>

#include "kmm/internals/memory_manager.hpp"
#include "kmm/utils/integer_fun.hpp"
#include "kmm/utils/panic.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

struct MemoryManager::Request {
    enum struct Status { Init, Allocated, Acquired, Deleted };

    Status status = Status::Init;
    TransactionId transaction_id;
    MemoryId memory_id;
    BufferId buffer_id;
    AccessMode mode;
};

struct TransactionSet {
    small_vector<TransactionId, 4> m_transactions;

    void insert(TransactionId id) {
        KMM_ASSERT(!contains(id));
        m_transactions.push_back(id);
    }

    void remove(TransactionId id) {
        KMM_ASSERT(!contains(id));
        m_transactions.remove(id);
    }

    bool contains(TransactionId id) const {
        return m_transactions.contains(id);
    }

    bool is_empty() const {
        return m_transactions.is_empty();
    }
};

struct HostEntry {
    bool is_allocated = false;
    void* data = nullptr;
    bool is_valid = false;
    TransactionSet transactions_active;
    CudaEventSet write_events;
    CudaEventSet access_events;
};

struct DeviceEntry {
    CUdeviceptr data = 0;
    TransactionSet transactions_active;
    std::optional<CudaEvent> allocation_event;
    bool is_valid = false;
    CudaEventSet write_events;
    CudaEventSet access_events;

    MemoryManager::BufferMeta* lru_older = nullptr;
    MemoryManager::BufferMeta* lru_newer = nullptr;

    bool is_allocated() const {
        return allocation_event.has_value();
    }
};

struct MemoryManager::BufferMeta {
    KMM_NOT_COPYABLE_OR_MOVABLE(BufferMeta);

  public:
    BufferLayout layout;
    HostEntry host_entry;
    DeviceEntry device_entry[MAX_DEVICES];

    BufferMeta(BufferLayout layout) : layout(layout) {}

    size_t size_in_bytes() const {
        return layout.size_in_bytes;
    }
};

struct MemoryManager::DeviceMeta {
    KMM_NOT_COPYABLE_OR_MOVABLE(DeviceMeta);

  public:
    CudaContextHandle context;
    size_t num_bytes_limit = 0;
    size_t num_bytes_used = 0;
    CUmemoryPool memory_pool;
    BufferMeta* lru_oldest = nullptr;
    BufferMeta* lru_newest = nullptr;

    DeviceMeta(MemoryDeviceInfo info) : context(info.context) {
        CudaContextGuard guard {context};

        size_t ignore;
        size_t total_memory;
        KMM_CUDA_CHECK(cuMemGetInfo(&ignore, &total_memory));

        if (total_memory < info.num_bytes_keep_available) {
            num_bytes_limit = 0;
        } else if (total_memory - info.num_bytes_keep_available < info.num_bytes_limit) {
            num_bytes_limit = total_memory - info.num_bytes_keep_available;
        } else {
            num_bytes_limit = info.num_bytes_limit;
        }

        CUdevice device;
        KMM_CUDA_CHECK(cuCtxGetDevice(&device));

        CUmemPoolProps props;
        bzero(&props, sizeof(CUmemPoolProps));

        props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.location = CUmemLocation {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = device};

        KMM_CUDA_CHECK(cuMemPoolCreate(&memory_pool, &props));
    }

    ~DeviceMeta() {
        KMM_CUDA_CHECK(cuMemPoolDestroy(memory_pool));
    }
};

MemoryManager::MemoryManager(
    std::shared_ptr<CudaStreamManager> stream_manager,
    std::vector<MemoryDeviceInfo> devices) :
    m_streams(stream_manager) {
    KMM_ASSERT(devices.size() < MAX_DEVICES);

    for (size_t i = 0; i < devices.size(); i++) {
        m_devices[i] = std::make_unique<DeviceMeta>(devices[i]);
    }
}

MemoryManager::~MemoryManager() {}

BufferId MemoryManager::create_buffer(BufferLayout layout) {
    layout.alignment = std::min(round_up_to_power_of_two(layout.alignment), size_t(128));
    layout.size_in_bytes = round_up_to_multiple(layout.size_in_bytes, layout.alignment);

    auto id = BufferId(next_buffer_id++);
    m_buffers.emplace(id, std::make_unique<BufferMeta>(layout));
    return id;
}

void MemoryManager::delete_buffer(BufferId id) {
    KMM_TODO();
}

std::shared_ptr<MemoryManager::Request> MemoryManager::create_request(
    TransactionId transaction_id,
    MemoryId device_id,
    BufferId buffer_id,
    AccessMode mode) {
    return std::make_shared<Request>(
        Request {Request::Status::Init, transaction_id, device_id, buffer_id, mode});
}

void MemoryManager::lock_allocation_host(TransactionId transaction_id, BufferId buffer_id) {
    BufferMeta& buffer = *m_buffers.at(buffer_id);

    if (!buffer.host_entry.is_allocated) {
        allocate_host(buffer);
    }

    buffer.host_entry.transactions_active.insert(transaction_id);
}

void MemoryManager::unlock_allocation_host(
    TransactionId transaction_id,
    BufferId buffer_id,
    AccessMode mode,
    CudaEvent event) {
    auto& buffer = m_buffers.at(buffer_id);
    auto& host_entry = buffer->host_entry;

    host_entry.transactions_active.remove(transaction_id);
    host_entry.access_events.insert(*m_streams, event);

    // invalid device buffers
    if (mode != AccessMode::Read) {
        host_entry.write_events.insert(*m_streams, event);

        for (size_t i = 0; i < MAX_DEVICES; i++) {
            auto& device_entry = buffer->device_entry[i];
            device_entry.is_valid = false;
            device_entry.write_events.clear();
        }
    }
}

bool MemoryManager::lock_allocation_device_async(
    TransactionId transaction_id,
    DeviceId device_id,
    BufferId buffer_id,
    CudaStreamId stream) {
    auto& buffer = *m_buffers[buffer_id];
    auto& entry = buffer.device_entry[device_id];

    if (!entry.is_allocated()) {
        while (true) {
            // Try to allocate memory
            if (try_allocate_device_async(stream, device_id, buffer)) {
                break;
            }

            // Try to evict something. If successful, retry allocation
            if (try_free_device_memory(stream, device_id)) {
                continue;
            }

            // Out of memory?
            return false;
        }
    } else {
        m_streams->wait_for_event(stream, entry.allocation_event.value());
    }

    if (entry.transactions_active.is_empty()) {
        remove_from_lru(device_id, buffer);
    }

    entry.transactions_active.insert(transaction_id);
    return true;
}

void MemoryManager::unlock_allocation_device_async(
    TransactionId transaction_id,
    DeviceId device_id,
    BufferId buffer_id,
    AccessMode mode,
    CudaEvent event,
    bool hint_last_access) {
    auto& buffer = m_buffers[buffer_id];
    auto& entry = buffer->device_entry[device_id];

    entry.transactions_active.remove(transaction_id);
    entry.access_events.insert(*m_streams, event);

    if (mode != AccessMode::Read) {
        entry.write_events.insert(*m_streams, event);

        for (size_t i = 0; i < MAX_DEVICES; i++) {
            auto& device_entry = buffer->device_entry[i];

            if (device_entry.is_valid && i != device_id) {
                device_entry.is_valid = false;
                device_entry.write_events.clear();
            }
        }

        auto& host_entry = buffer->host_entry;
        if (host_entry.is_valid) {
            host_entry.is_valid = false;
            host_entry.write_events.clear();
        }
    }

    // Put into LRU
    if (entry.transactions_active.is_empty()) {
        insert_into_lru(device_id, *buffer, hint_last_access);
    }
}

void MemoryManager::delete_request_async(const MemoryRequest& req, CudaEvent event) {
    KMM_ASSERT(req->status == Request::Status::Acquired);
    req->status = Request::Status::Deleted;

    if (req->memory_id.is_device()) {
        unlock_allocation_device_async(
            req->transaction_id,
            req->memory_id.as_device(),
            req->buffer_id,
            req->mode,
            event);
    } else {
        unlock_allocation_host(req->transaction_id, req->buffer_id, req->mode, event);
    }
}

void MemoryManager::delete_request(const MemoryRequest& req) {
    return delete_request_async(req, CudaEvent {});
}

void* MemoryManager::get_host_pointer(const MemoryRequest& req) {
    KMM_ASSERT(req->status == Request::Status::Acquired);
    KMM_ASSERT(req->memory_id.is_host());

    auto& buffer = m_buffers.at(req->buffer_id);
    auto& host_entry = buffer->host_entry;

    KMM_ASSERT(host_entry.transactions_active.contains(req->transaction_id));
    return host_entry.data;
}

CUdeviceptr MemoryManager::get_device_pointer(const MemoryRequest& req, DeviceId device_id) {
    KMM_ASSERT(req->status == Request::Status::Acquired);
    KMM_ASSERT(req->memory_id.as_device() == device_id);

    auto& buffer = m_buffers.at(req->buffer_id);
    auto& device_entry = buffer->device_entry[device_id];

    KMM_ASSERT(device_entry.transactions_active.contains(req->transaction_id));
    return device_entry.data;
}

void MemoryManager::trim_device_memory(
    CudaStreamId stream,
    DeviceId device_id,
    size_t num_bytes_max) {
    auto& device = *m_devices[device_id];

    while (device.num_bytes_used > num_bytes_max) {
        if (!try_free_device_memory(stream, device_id)) {
            break;
        }
    }

    CudaContextGuard guard {device.context};
    KMM_CUDA_CHECK(cuMemPoolTrimTo(device.memory_pool, num_bytes_max));
}

void MemoryManager::insert_into_lru(DeviceId device_id, BufferMeta& buffer, bool hint_last_access) {
    auto& entry = buffer.device_entry[device_id];

    KMM_ASSERT(entry.transactions_active.is_empty());
    KMM_ASSERT(entry.lru_newer == nullptr);
    KMM_ASSERT(entry.lru_older == nullptr);

    auto& device = *m_devices[device_id];

    if (device.lru_oldest == nullptr) {
        device.lru_oldest = &buffer;
        device.lru_newest = &buffer;
    } else if (!hint_last_access) {
        auto* tail = device.lru_newest;
        tail->device_entry[device_id].lru_newer = &buffer;
        buffer.device_entry[device_id].lru_older = tail;

        device.lru_newest = &buffer;
    } else {
        auto* head = device.lru_oldest;
        head->device_entry[device_id].lru_older = &buffer;
        buffer.device_entry[device_id].lru_newer = head;

        device.lru_oldest = &buffer;
    }
}

bool MemoryManager::try_free_device_memory(CudaStreamId stream, DeviceId device_id) {
    auto& device = *m_devices[device_id];

    if (device.lru_oldest == nullptr) {
        return false;
    }

    auto& buffer = *device.lru_oldest;
    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(device_entry.transactions_active.is_empty());
    KMM_ASSERT(device_entry.is_allocated());
    KMM_ASSERT(device_entry.data != 0);

    if (device_entry.is_valid && !host_entry.is_valid) {
        copy_d2h(stream, device_id, buffer);
    }

    deallocate_device_async(stream, device_id, buffer);

    return true;
}

void MemoryManager::remove_from_lru(DeviceId device_id, BufferMeta& buffer) {
    auto& entry = buffer.device_entry[device_id];

    if (entry.lru_newer != nullptr) {
        entry.lru_newer->device_entry[device_id].lru_older = entry.lru_older;
    } else {
        m_devices[device_id]->lru_newest = entry.lru_older;
    }

    if (entry.lru_older != nullptr) {
        entry.lru_older->device_entry[device_id].lru_newer = entry.lru_newer;
    } else {
        m_devices[device_id]->lru_oldest = entry.lru_newer;
    }

    entry.lru_newer = nullptr;
    entry.lru_older = nullptr;
}

bool MemoryManager::acquire_allocation_async(const MemoryRequest& req, CudaStreamId stream) {
    KMM_ASSERT(req->status == Request::Status::Init);

    if (req->memory_id.is_device()) {
        bool success = lock_allocation_device_async(
            req->transaction_id,
            req->memory_id.as_device(),
            req->buffer_id,
            stream);

        if (!success) {
            return false;
        }
    } else {
        lock_allocation_host(req->transaction_id, req->buffer_id);
    }

    req->status = Request::Status::Allocated;
    return true;
}

void MemoryManager::acquire_access_async(const MemoryRequest& req, CudaStreamId stream) {
    KMM_ASSERT(req->status == Request::Status::Allocated);

    if (req->memory_id.is_device()) {
        update_data_device_async(
            req->transaction_id,
            req->memory_id.as_device(),
            req->buffer_id,
            stream);
    } else {
        update_data_host_async(req->transaction_id, req->buffer_id, stream);
    }

    req->status = Request::Status::Acquired;
}

void MemoryManager::update_data_host_async(
    TransactionId transaction_id,
    BufferId buffer_id,
    CudaStreamId stream) {
    auto& buffer = m_buffers[buffer_id];
    auto& host_entry = buffer->host_entry;

    KMM_ASSERT(host_entry.is_allocated);
    KMM_ASSERT(host_entry.transactions_active.contains(transaction_id));

    if (!host_entry.is_valid) {
        for (size_t i = 0; i < MAX_DEVICES; i++) {
            if (buffer->device_entry[i].is_valid) {
                copy_d2h(stream, static_cast<DeviceId>(i), *buffer);
                break;
            }
        }

        if (!host_entry.is_valid) {
            host_entry.is_valid = true;
        }
    }

    m_streams->wait_for_events(stream, host_entry.write_events);
}

void MemoryManager::update_data_device_async(
    TransactionId transaction_id,
    DeviceId device_id,
    BufferId buffer_id,
    CudaStreamId stream) {
    auto& buffer = m_buffers[buffer_id];
    auto& device_entry = buffer->device_entry[device_id];
    auto& host_entry = buffer->host_entry;

    KMM_ASSERT(device_entry.is_allocated());
    KMM_ASSERT(device_entry.transactions_active.contains(transaction_id));

    m_streams->wait_for_event(stream, device_entry.allocation_event.value());

    if (!device_entry.is_valid) {
        if (!host_entry.is_valid) {
            for (size_t i = 0; i < MAX_DEVICES; i++) {
                if (buffer->device_entry[i].is_valid) {
                    copy_d2h(stream, static_cast<DeviceId>(i), *buffer);
                    break;
                }
            }
        }

        if (host_entry.is_valid) {
            copy_h2d(stream, device_id, *buffer);
        }

        if (!device_entry.is_valid) {
            auto event = device_entry.allocation_event.value();
            device_entry.write_events.clear();
            device_entry.write_events.insert(*m_streams, event);
            device_entry.access_events.insert(*m_streams, event);
            m_streams->wait_for_event(stream, event);
        }
    }

    if (device_entry.is_valid) {
        m_streams->wait_for_events(stream, device_entry.write_events);
    }
}

void MemoryManager::allocate_host(BufferMeta& buffer) {
    auto& host_entry = buffer.host_entry;

    KMM_ASSERT(host_entry.is_allocated == false);
    KMM_ASSERT(host_entry.transactions_active.is_empty());

    KMM_CUDA_CHECK(
        cuMemHostAlloc(&host_entry.data, buffer.size_in_bytes(), CU_MEMHOSTALLOC_PORTABLE));

    host_entry.is_allocated = true;
    host_entry.is_valid = false;
    host_entry.write_events.clear();
    host_entry.access_events.clear();
}

bool MemoryManager::try_allocate_device_async(
    CudaStreamId stream,
    DeviceId device_id,
    BufferMeta& buffer) {
    auto& device = *m_devices[device_id];
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(device_entry.is_allocated() == false);
    KMM_ASSERT(device_entry.transactions_active.is_empty());

    // Buffer is larger than we can possibly allocate
    if (buffer.size_in_bytes() > device.num_bytes_limit - device.num_bytes_used) {
        return false;
    }

    CUresult result;

    {
        CudaContextGuard guard {device.context};
        result = cuMemAllocFromPoolAsync(
            &device_entry.data,
            buffer.size_in_bytes(),
            device.memory_pool,
            m_streams->get(stream));
    }

    if (result == CUDA_ERROR_OUT_OF_MEMORY) {
        return false;
    }

    if (result != CUDA_SUCCESS) {
        throw CudaDriverException("failed `cuMemAllocAsync`", result);
    }

    device.num_bytes_used += buffer.size_in_bytes();

    device_entry.allocation_event = m_streams->record_event(stream);
    device_entry.write_events.clear();
    return true;
}

void MemoryManager::deallocate_device_async(
    CudaStreamId stream,
    DeviceId device_id,
    BufferMeta& buffer) {
    auto& device = *m_devices[device_id];
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(device_entry.is_allocated());
    KMM_ASSERT(device_entry.transactions_active.is_empty());

    m_streams->wait_for_event(stream, *device_entry.allocation_event);
    m_streams->wait_for_events(stream, device_entry.access_events);

    {
        CudaContextGuard guard {device.context};
        KMM_CUDA_CHECK(cuMemFreeAsync(device_entry.data, m_streams->get(stream)));
    }

    device.num_bytes_used -= buffer.size_in_bytes();

    device_entry.allocation_event = std::nullopt;
    device_entry.data = 0;
    device_entry.write_events.clear();
    device_entry.access_events.clear();

    remove_from_lru(device_id, buffer);
}

void MemoryManager::copy_h2d(CudaStreamId stream, DeviceId device_id, BufferMeta& buffer) {
    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    KMM_ASSERT(host_entry.is_allocated && host_entry.is_valid);
    KMM_ASSERT(device_entry.is_allocated() && !device_entry.is_valid);

    m_streams->wait_for_events(stream, host_entry.write_events);
    m_streams->wait_for_event(stream, *device_entry.allocation_event);

    KMM_CUDA_CHECK(cuMemcpyHtoDAsync(
        device_entry.data,
        host_entry.data,
        buffer.size_in_bytes(),
        m_streams->get(stream)));

    auto event = m_streams->record_event(stream);
    device_entry.write_events.clear();
    device_entry.write_events.insert(*m_streams, event);
    host_entry.access_events.insert(*m_streams, event);
    device_entry.is_valid = true;
}

void MemoryManager::copy_d2h(CudaStreamId stream, DeviceId device_id, BufferMeta& buffer) {
    auto& host_entry = buffer.host_entry;
    auto& device_entry = buffer.device_entry[device_id];

    if (!host_entry.is_allocated) {
        allocate_host(buffer);
    }

    KMM_ASSERT(host_entry.is_allocated && !host_entry.is_valid);
    KMM_ASSERT(device_entry.is_allocated() && device_entry.is_valid);
    m_streams->wait_for_events(stream, device_entry.write_events);

    KMM_CUDA_CHECK(cuMemcpyDtoHAsync(
        host_entry.data,
        device_entry.data,
        buffer.size_in_bytes(),
        m_streams->get(stream)));

    auto event = m_streams->record_event(stream);
    host_entry.write_events.clear();
    host_entry.write_events.insert(*m_streams, event);
    device_entry.access_events.insert(*m_streams, event);
    host_entry.is_valid = true;
}

}  // namespace kmm