
#include "kmm/internals/executor.hpp"

namespace kmm {

class Operation {
  public:
    std::shared_ptr<TaskNode> job;

    Operation(std::shared_ptr<TaskNode> job) : job(job) {}
    ~Operation() = default;
    virtual Poll poll(Executor&) = 0;
};

struct HostOperation: public Operation {
    std::shared_ptr<HostTask> task;
    size_t current_request = 0;
    std::vector<MemoryRequest> requests;
    CudaEventSet dependencies;

    HostOperation(
        std::shared_ptr<TaskNode> job,
        std::shared_ptr<HostTask> task,
        std::vector<MemoryRequest> requests,
        CudaEventSet dependencies) :
        Operation(std::move(job)),
        task(std::move(task)),
        requests(std::move(requests)),
        dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) {
        bool all_ready = true;

        for (size_t i = 0; i < requests.size(); i++) {
            if (!executor.m_memory->poll_request(*requests[i], dependencies)) {
                all_ready = false;
            }
        }

        if (!all_ready) {
            return Poll::Pending;
        }

        if (!executor.m_streams->is_ready(dependencies)) {
            return Poll::Pending;
        }

        auto accessors = std::vector<BufferAccessor>();
        for (size_t i = 0; i < requests.size(); i++) {
            accessors.push_back(executor.m_memory->get_accessor(*requests[i]));
        }

        task->execute(TaskContext {accessors});

        for (size_t i = 0; i < requests.size(); i++) {
            executor.m_memory->release_request(requests[i]);
        }

        executor.m_scheduler->set_complete(job);
        return Poll::Ready;
    }
};

struct DeviceOperation: public Operation {
    enum struct Status { Pending, Running, Done } status = Status::Pending;

    size_t stream_index;
    std::shared_ptr<DeviceTask> task;
    std::vector<MemoryRequest> requests;
    CudaEventSet dependencies;
    CudaEvent event;

    DeviceOperation(
        std::shared_ptr<TaskNode> job,
        size_t stream_index,
        std::shared_ptr<DeviceTask> task,
        std::vector<MemoryRequest> requests,
        CudaEventSet dependencies) :
        Operation(std::move(job)),
        stream_index(stream_index),
        task(std::move(task)),
        requests(std::move(requests)),
        dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) {
        if (status == Status::Pending) {
            auto& [stream, device] = executor.m_devices[stream_index];
            bool all_ready = true;

            for (size_t i = 0; i < requests.size(); i++) {
                if (!executor.m_memory->poll_request(*requests[i], dependencies)) {
                    all_ready = false;
                }
            }

            if (!all_ready) {
                return Poll::Pending;
            }

            auto accessors = std::vector<BufferAccessor>();
            for (size_t i = 0; i < requests.size(); i++) {
                accessors.push_back(executor.m_memory->get_accessor(*requests[i]));
            }

            event = executor.m_streams->with_stream(stream, dependencies, [&](auto s) {
                CudaContextGuard guard {executor.m_streams->get(device->device_id())};
                task->execute(*device, TaskContext {accessors});

                // Make sure to wait on default stream if anything was accidentally submitted on the wrong stream
                KMM_CUDA_CHECK(cuStreamSynchronize(nullptr));
            });

            for (size_t i = 0; i < requests.size(); i++) {
                executor.m_memory->release_request(requests[i], event);
            }

            executor.m_scheduler->set_scheduled(job, event);
            status = Status::Running;
        }

        if (status == Status::Running) {
            if (!executor.m_streams->is_ready(event)) {
                return Poll::Pending;
            }

            executor.m_scheduler->set_complete(job);
            status = Status::Done;
        }

        return Poll::Ready;
    }
};

Executor::Executor(
    std::shared_ptr<CudaStreamManager> streams,
    std::shared_ptr<MemoryManager> memory,
    std::shared_ptr<Scheduler> scheduler) :
    m_streams(streams),
    m_memory(memory),
    m_scheduler(scheduler) {
    KMM_PANIC("TODO: initialize devices");
}

Executor::~Executor() = default;

void Executor::make_progress() {
    size_t new_size = m_operations.size();
    size_t index = 0;

    while (index < new_size) {
        if (m_operations[index]->poll(*this) == Poll::Ready) {
            m_operations[index] = std::move(m_operations[new_size - 1]);
            new_size--;
        } else {
            index++;
        }
    }

    m_operations.resize(new_size);
}

bool Executor::is_idle() const {
    return m_operations.empty();
}

void Executor::submit_host_task(
    std::shared_ptr<TaskNode> job,
    std::shared_ptr<HostTask> task,
    const std::vector<BufferRequirement>& buffers,
    CudaEventSet dependencies) {
    auto trans = m_memory->create_transaction();
    auto reqs = std::vector<MemoryRequest>(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++) {
        reqs[i] = m_memory->create_request(
            buffers[i].buffer_id,
            buffers[i].memory_id,
            buffers[i].access_mode,
            trans);
    }

    m_operations.push_back(std::make_unique<HostOperation>(
        std::move(job),
        std::move(task),
        std::move(reqs),
        std::move(dependencies)));
}

void Executor::submit_device_task(
    std::shared_ptr<TaskNode> job,
    DeviceId device_id,
    std::shared_ptr<DeviceTask> task,
    const std::vector<BufferRequirement>& buffers,
    CudaEventSet dependencies) {
    // TODO: improve stream selection
    auto trans = m_memory->create_transaction();
    auto reqs = std::vector<MemoryRequest>();

    for (size_t i = 0; i < buffers.size(); i++) {
        reqs[i] = m_memory->create_request(
            buffers[i].buffer_id,
            buffers[i].memory_id,
            buffers[i].access_mode,
            trans);
    }

    m_operations.push_back(
        std::make_unique<DeviceOperation>(job, 0, task, std::move(reqs), std::move(dependencies)));
}

class EmptyTask: public HostTask, public DeviceTask {
  public:
    void execute(TaskContext context) override {}
    void execute(CudaDevice& device, TaskContext context) override {}
};

void Executor::submit_prefetch(
    std::shared_ptr<TaskNode> job,
    BufferId buffer_id,
    MemoryId memory_id,
    CudaEventSet dependencies) {
    auto task = std::make_shared<EmptyTask>();
    std::vector<BufferRequirement> buffers = {BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .access_mode = AccessMode::Read}};

    if (memory_id.is_host()) {
        submit_host_task(job, task, buffers, dependencies);
    } else {
        submit_device_task(job, memory_id.as_device(), task, buffers, dependencies);
    }
}

void Executor::submit_copy(
    std::shared_ptr<TaskNode> job,
    BufferId src_id,
    MemoryId src_memory,
    BufferId dst_id,
    MemoryId dst_memory,
    CopyDescription spec,
    CudaEventSet dependencies) {
    KMM_ASSERT(src_id != dst_id || src_memory == dst_memory);

    std::vector<BufferRequirement> buffers = {
        BufferRequirement {
            .buffer_id = src_id,
            .memory_id = src_memory,
            .access_mode = AccessMode::Read},
        BufferRequirement {
            .buffer_id = dst_id,
            .memory_id = dst_memory,
            .access_mode = AccessMode::ReadWrite}};

    if (src_memory.is_host() && dst_memory.is_host()) {
        KMM_TODO();
    } else {
        KMM_TODO();
    }
}

}  // namespace kmm