#include <mutex>

#include "spdlog/spdlog.h"

#include "kmm/internals/executor.hpp"

namespace kmm {

struct OperationQueue {
    mutable std::mutex m_mutex;
    mutable std::vector<std::shared_ptr<Operation>> m_queue;
};

class Operation: public std::enable_shared_from_this<Operation>, public Notify {
  public:
    std::shared_ptr<TaskNode> job;
    std::shared_ptr<OperationQueue> m_queue;
    mutable std::atomic_flag is_queued = true;

    Operation(std::shared_ptr<TaskNode> job) : job(job) {}
    virtual ~Operation() = default;
    virtual Poll poll(Executor&) = 0;

    void notify() const noexcept final;
};

void Operation::notify() const noexcept {
    if (is_queued.test_and_set()) {
        std::unique_lock guard {m_queue->m_mutex};
        m_queue->m_queue.push_back(const_cast<Operation*>(this)->shared_from_this());
    }
}

struct HostOperation: public Operation {
    std::shared_ptr<Task> task;
    std::vector<BufferRequirement> buffers;
    std::vector<MemoryRequest> requests;
    GPUEventSet dependencies;

    HostOperation(
        std::shared_ptr<TaskNode> job,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> buffers,
        GPUEventSet dependencies) :
        Operation(std::move(job)),
        task(std::move(task)),
        buffers(std::move(buffers)),
        dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) {
        try {
            if (buffers.size() != requests.size()) {
                auto trans = executor.m_memory->create_transaction();

                for (size_t i = 0; i < buffers.size(); i++) {
                    auto buffer = executor.m_buffers->get(buffers[i].buffer_id);
                    auto request = executor.m_memory->create_request(
                        buffer,
                        buffers[i].memory_id,
                        buffers[i].access_mode,
                        trans);

                    requests.push_back(std::move(request));
                }
            }

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

            auto context = HostContext {};
            task->execute(context, TaskContext {accessors});

        } catch (const std::exception& error) {
            for (const auto& buffer : buffers) {
                if (buffer.access_mode != AccessMode::Read) {
                    executor.m_buffers->poison(buffer.buffer_id, job->id(), error);
                }
            }
        }

        for (size_t i = 0; i < requests.size(); i++) {
            executor.m_memory->release_request(requests[i]);
        }

        executor.m_scheduler->set_complete(job);
        return Poll::Ready;
    }
};

struct DeviceOperation: public Operation {
    enum struct Status { Init, Pending, Running, Done };
    Status status = Status::Init;

    size_t stream_index;
    std::shared_ptr<Task> task;
    std::vector<BufferRequirement> buffers;
    std::vector<MemoryRequest> requests;
    GPUEventSet dependencies;
    GPUEvent event;

    DeviceOperation(
        std::shared_ptr<TaskNode> job,
        size_t stream_index,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> buffers,
        GPUEventSet dependencies) :
        Operation(std::move(job)),
        stream_index(stream_index),
        task(std::move(task)),
        buffers(std::move(buffers)),
        dependencies(std::move(dependencies)) {}

    Poll poll(Executor& executor) {
        if (status == Status::Init) {
            auto trans = executor.m_memory->create_transaction();
            for (size_t i = 0; i < buffers.size(); i++) {
                auto buffer = executor.m_buffers->get(buffers[i].buffer_id);
                auto request = executor.m_memory->create_request(
                    buffer,
                    buffers[i].memory_id,
                    buffers[i].access_mode,
                    trans);

                requests.push_back(request);
            }

            status = Status::Pending;
        }

        if (status == Status::Pending) {
            bool all_ready = true;

            for (size_t i = 0; i < requests.size(); i++) {
                if (!executor.m_memory->poll_request(*requests[i], dependencies)) {
                    all_ready = false;
                }
            }

            if (!all_ready) {
                return Poll::Pending;
            }

            auto& [stream, device] = executor.m_devices[stream_index];
            auto accessors = std::vector<BufferAccessor>();

            for (size_t i = 0; i < requests.size(); i++) {
                accessors.push_back(executor.m_memory->get_accessor(*requests[i]));
            }

            event = executor.m_streams->with_stream(stream, dependencies, [&](auto s) {
                GPUContextGuard guard {device->context_handle()};
                task->execute(*device, TaskContext {accessors});

                // Make sure to wait on default stream if anything was accidentally submitted on the wrong stream
                KMM_GPU_CHECK(gpuStreamSynchronize(nullptr));
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
    std::vector<GPUContextHandle> contexts,
    std::shared_ptr<GPUStreamManager> streams,
    std::shared_ptr<BufferManager> buffers,
    std::shared_ptr<MemoryManager> memory,
    std::shared_ptr<Scheduler> scheduler) :
    m_streams(streams),
    m_buffers(buffers),
    m_memory(memory),
    m_scheduler(scheduler) {
    for (size_t i = 0; i < contexts.size(); i++) {
        auto device_id = DeviceId(i);

        auto context = contexts[i];
        auto stream = streams->create_stream(context);
        auto device = std::make_unique<GPUDevice>(
            DeviceInfo(device_id, context),
            context,
            streams->get(stream));

        m_devices.emplace_back(stream, std::move(device));
    }
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

void Executor::submit_task(
    std::shared_ptr<TaskNode> job,
    ProcessorId processor_id,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> buffers,
    GPUEventSet dependencies) {
    if (processor_id.is_device()) {
        submit_device_task(
            job,
            processor_id.as_device(),
            std::move(task),
            std::move(buffers),
            std::move(dependencies));
    } else {
        submit_host_task(job, std::move(task), std::move(buffers), std::move(dependencies));
    }
}

void Executor::submit_host_task(
    std::shared_ptr<TaskNode> job,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> buffers,
    GPUEventSet dependencies) {
    spdlog::debug(
        "submit host task {} (buffers={}, dependencies={})",
        job->id(),
        buffers.size(),
        dependencies);

    m_operations.push_back(std::make_unique<HostOperation>(
        std::move(job),
        std::move(task),
        std::move(buffers),
        std::move(dependencies)));
}

void Executor::submit_device_task(
    std::shared_ptr<TaskNode> job,
    DeviceId device_id,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> buffers,
    GPUEventSet dependencies) {
    spdlog::debug(
        "submit device task {} (device={}, buffers={}, dependencies={})",
        job->id(),
        device_id,
        buffers.size(),
        dependencies);

    // TODO: improve stream selection
    size_t stream_index = device_id.get();

    m_operations.push_back(std::make_unique<DeviceOperation>(
        job,
        stream_index,
        task,
        std::move(buffers),
        std::move(dependencies)));
}

class EmptyTask: public Task {
  public:
    void execute(ExecutionContext& device, TaskContext context) override {}
};

void Executor::submit_prefetch(
    std::shared_ptr<TaskNode> job,
    BufferId buffer_id,
    MemoryId memory_id,
    GPUEventSet dependencies) {
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
    GPUEventSet dependencies) {
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