#pragma once

#include <future>
#include <memory>
#include <vector>

#include "cuda_stream_manager.hpp"
#include "memory_manager.hpp"

#include "kmm/core/copy_specification.hpp"
#include "kmm/core/task.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

class Scheduler;
class Operation;
using OperationList = small_vector<std::shared_ptr<Operation>>;

enum struct PollResult { Pending, Ready };

class Operation {
  public:
    Operation(EventId id) : m_id(id) {}
    virtual ~Operation() = default;

    virtual std::optional<CudaEvent> as_cuda_event() const {
        return std::nullopt;
    };

    virtual PollResult poll_impl(Scheduler&) = 0;

    PollResult poll(Scheduler&);

    EventId id() const {
        return m_id;
    }

    bool is_ready() const {
        return m_completed;
    }

  private:
    EventId m_id;
    bool m_completed = false;
};

class JoinOperation: public Operation {
  public:
    JoinOperation(EventId id, OperationList deps) :
        Operation(id),
        m_dependencies(std::move(deps)) {}

    std::optional<CudaEvent> as_cuda_event() const;
    PollResult poll_impl(Scheduler& scheduler);

  private:
    OperationList m_dependencies;
};

struct BufferRequest {
    MemoryId memory_id;
    BufferId buffer_id;
    bool is_write;
    std::optional<std::shared_ptr<Operation>> dependency;
    MemoryRequest request;
};

class ExecuteHostOperation: public Operation {
    enum struct Status { Init, AcquireAllocation, AcquireData, Ready, Running, Completed };

  public:
    ExecuteHostOperation(
        EventId id,
        std::shared_ptr<HostTask> task,
        std::vector<BufferRequest> buffers);
    PollResult poll_impl(Scheduler&) override;

  private:
    std::future<void> m_future;
    Status m_status = Status::Init;
    size_t m_index = 0;
    std::shared_ptr<HostTask> m_task;
    std::vector<BufferRequest> m_buffers;
};

class DeviceOperation: public Operation {
    enum struct Status { Init, AcquireMemory, AcquireData, Ready, Running, Completed };

  public:
    DeviceOperation(EventId id, DeviceId device, std::vector<BufferRequest> buffers);

    virtual void submit_async(
        Scheduler& scheduler,
        CudaStreamId stream,
        std::vector<BufferAccessor> accessors) = 0;

    std::optional<CudaEvent> as_cuda_event() const final;
    PollResult poll_impl(Scheduler&) final;

    void schedule_onto_stream(CudaStreamId stream, Scheduler& scheduler);

    bool is_scheduled() const {
        return m_scheduled;
    }

  protected:
    Status m_status = Status::Init;
    DeviceId m_device;
    size_t m_index = 0;
    bool m_scheduled = false;
    CudaStreamId m_stream;
    CudaEvent m_event;
    std::vector<BufferRequest> m_buffers;
};

class ExecuteDeviceOperation: public DeviceOperation {
  public:
    ExecuteDeviceOperation(
        EventId id,
        DeviceId device,
        std::shared_ptr<DeviceTask> task,
        std::vector<BufferRequest> buffers) :
        DeviceOperation(id, device, std::move(buffers)),
        m_task(std::move(task)) {}

    void submit_async(
        Scheduler& scheduler,
        CudaStreamId stream,
        std::vector<BufferAccessor> accessors) final;

  private:
    std::shared_ptr<DeviceTask> m_task;
};

class CopyDeviceOperation: public DeviceOperation {
  public:
    CopyDeviceOperation(
        EventId id,
        DeviceId device,
        BufferRequest src_buffer,
        BufferRequest dst_buffer,
        CopySpecification operation);

    void submit_async(
        Scheduler& scheduler,
        CudaStreamId stream,
        std::vector<BufferAccessor> accessors) final;

  private:
    CopySpecification m_operation;
};

}  // namespace kmm