#include "kmm/internals/operation.hpp"
#include "kmm/internals/scheduler.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

PollResult Operation::poll(Scheduler& scheduler) {
    if (m_completed) {
        return PollResult::Ready;
    }

    if (poll_impl(scheduler) == PollResult::Pending) {
        return PollResult::Pending;
    }

    m_completed = true;
    return PollResult::Ready;
}

PollResult JoinOperation::poll_impl(Scheduler& scheduler) {
    for (const auto& dep : m_dependencies) {
        if (!dep->is_ready()) {
            return PollResult::Pending;
        }
    }

    return PollResult::Ready;
}

std::optional<CudaEvent> JoinOperation::as_cuda_event() const {
    std::optional<CudaEvent> cuda_event;
    size_t num_pending = 0;

    for (const auto& dep : m_dependencies) {
        if (!dep->is_ready()) {
            cuda_event = dep->as_cuda_event();
            num_pending++;
        }
    }

    return num_pending == 1 ? cuda_event : std::nullopt;
}

ExecuteHostOperation::ExecuteHostOperation(
    EventId id,
    std::shared_ptr<HostTask> task,
    std::vector<BufferRequest> buffers) :
    Operation(id),
    m_task(std::move(task)),
    m_buffers(std::move(buffers)) {}

PollResult ExecuteHostOperation::poll_impl(Scheduler& scheduler) {
    CudaStreamId stream = {};  // TODO: What stream to use?

    if (m_status == Status::Init) {
        for (auto& req : m_buffers) {
            auto mode = req.is_write ? AccessMode::Exclusive : AccessMode::Read;
            req.request =
                scheduler.m_memory->create_request(id(), req.memory_id, req.buffer_id, mode);
        }

        m_status = Status::AcquireAllocation;
    }

    if (m_status == Status::AcquireAllocation) {
        while (m_index < m_buffers.size()) {
            auto& buf = m_buffers[m_index];

            if (!scheduler.m_memory->acquire_allocation_async(buf.request, stream)) {
                return PollResult::Pending;
            }

            m_index++;
        }

        m_status = Status::AcquireData;
        m_index = 0;
    }

    if (m_status == Status::AcquireData) {
        while (m_index < m_buffers.size()) {
            auto& buf = m_buffers[m_index];
            auto& req = buf.request;

            if (buf.dependency && !(*buf.dependency)->is_ready()) {
                return PollResult::Pending;
            }

            scheduler.m_memory->acquire_access_async(req, stream);
            m_index++;
        }

        m_status = Status::Ready;
    }

    if (m_status == Status::Ready) {
        for (const auto& buf : m_buffers) {
            scheduler.m_memory->get_host_pointer(buf.request);
        }

        // TODO: submit to host

        m_status = Status::Running;
    }

    if (m_status == Status::Running) {
        if (m_future.wait_for(std::chrono::system_clock::duration {})
            == std::future_status::timeout) {
            return PollResult::Pending;
        }

        for (const auto& buf : m_buffers) {
            scheduler.m_memory->delete_request(buf.request);
        }

        m_status = Status::Completed;
    }

    return PollResult::Ready;
}

DeviceOperation::DeviceOperation(EventId id, DeviceId device, std::vector<BufferRequest> buffers) :
    Operation(id),
    m_device(device),
    m_buffers(std::move(buffers)) {}

void DeviceOperation::schedule_onto_stream(CudaStreamId stream, Scheduler& scheduler) {
    KMM_ASSERT(m_scheduled == false);
    m_stream = stream;
    m_scheduled = true;

    for (auto& req : m_buffers) {
        auto mode = req.is_write ? AccessMode::Exclusive : AccessMode::Read;
        req.request = scheduler.m_memory->create_request(id(), req.memory_id, req.buffer_id, mode);
    }
}

std::optional<CudaEvent> DeviceOperation::as_cuda_event() const {
    if (m_status == Status::Running) {
        return m_event;
    }

    return std::nullopt;
}

PollResult DeviceOperation::poll_impl(Scheduler& scheduler) {
    if (m_status == Status::Init) {
        if (!m_scheduled) {
            return PollResult::Pending;
        }

        m_status = Status::AcquireMemory;
    }

    if (m_status == Status::AcquireMemory) {
        while (m_index < m_buffers.size()) {
            auto& req = m_buffers[m_index].request;

            if (!scheduler.m_memory->acquire_allocation_async(req, m_stream)) {
                return PollResult::Pending;
            }

            m_index++;
        }

        m_status = Status::AcquireData;
        m_index = 0;
    }

    if (m_status == Status::AcquireData) {
        while (m_index < m_buffers.size()) {
            auto& buf = m_buffers[m_index];
            auto& req = buf.request;

            if (const auto& dep = buf.dependency) {
                if (auto e = (*dep)->as_cuda_event()) {
                    scheduler.m_streams->wait_for_event(m_stream, *e);
                } else if (!(*dep)->is_ready()) {
                    return PollResult::Pending;
                }
            }

            scheduler.m_memory->acquire_access_async(req, m_stream);
            m_index++;
        }

        m_status = Status::Ready;
    }

    if (m_status == Status::Ready) {
        std::vector<BufferAccessor> accessors;

        for (const auto& req : m_buffers) {
            auto address = scheduler.m_memory->get_device_pointer(req.request, m_device);

            accessors.push_back(BufferAccessor {
                .buffer_id = req.buffer_id,
                .memory_id = req.memory_id,
                .is_writable = req.is_write,
                .address = reinterpret_cast<void*>(address)  //
            });
        }

        submit_async(scheduler, m_stream, std::move(accessors));

        scheduler.m_streams->wait_on_default_stream(m_stream);
        m_event = scheduler.m_streams->record_event(m_stream);

        for (const auto& buf : m_buffers) {
            scheduler.m_memory->delete_request_async(buf.request, m_event);
        }

        m_status = Status::Running;
    }

    if (m_status == Status::Running) {
        if (!scheduler.m_streams->is_ready(m_event)) {
            return PollResult::Pending;
        }

        m_status = Status::Completed;
    }

    return PollResult::Ready;
}

void ExecuteDeviceOperation::submit_async(
    Scheduler& scheduler,
    CudaStreamId stream,
    std::vector<BufferAccessor> accessors) {
    TaskContext context {std::move(accessors)};

    m_task->submit(scheduler.m_streams->get(stream), context);
}

CopyDeviceOperation::CopyDeviceOperation(
    EventId id,
    DeviceId device,
    BufferRequest src_buffer,
    BufferRequest dst_buffer,
    CopySpecification operation) :
    DeviceOperation(id, device, {src_buffer, dst_buffer}),
    m_operation(operation) {}

void CopyDeviceOperation::submit_async(
    Scheduler& scheduler,
    CudaStreamId stream,
    std::vector<BufferAccessor> accessors) {
    KMM_ASSERT(accessors.size() == 2);
    KMM_ASSERT(accessors[1].is_writable);
    KMM_ASSERT(m_operation.minimum_source_bytes_needed() >= accessors[0].layout.size_in_bytes);
    KMM_ASSERT(m_operation.minimum_destination_bytes_needed() >= accessors[1].layout.size_in_bytes);

    m_operation.simplify();
    KMM_ASSERT(m_operation.effective_dimensionality() == 0);

    auto src_id = accessors[0].memory_id;
    auto dst_id = accessors[1].memory_id;

    if (src_id.is_host() && dst_id.is_device()) {
        KMM_CUDA_CHECK(cuMemcpyHtoDAsync(
            CUdeviceptr(accessors[1].address) + m_operation.dst_offset,
            static_cast<const char*>(accessors[0].address) + m_operation.src_offset,
            m_operation.element_size,
            scheduler.m_streams->get(stream)));
    } else if (src_id.is_device() && dst_id.is_host()) {
        KMM_CUDA_CHECK(cuMemcpyDtoHAsync(
            static_cast<char*>(accessors[1].address) + m_operation.dst_offset,
            CUdeviceptr(accessors[0].address) + m_operation.src_offset,
            m_operation.element_size,
            scheduler.m_streams->get(stream)));
    } else if (src_id.is_device() && dst_id.is_device()) {
        KMM_CUDA_CHECK(cuMemcpyDtoDAsync(
            CUdeviceptr(accessors[1].address) + m_operation.dst_offset,
            CUdeviceptr(accessors[0].address) + m_operation.src_offset,
            m_operation.element_size,
            scheduler.m_streams->get(stream)));
    } else {
        KMM_PANIC("host to host copies are not supported by CopyDeviceOperation");
    }
}
}  // namespace kmm