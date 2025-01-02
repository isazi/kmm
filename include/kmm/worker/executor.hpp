#pragma once

#include <future>

#include "kmm/core/execution_context.hpp"
#include "kmm/utils/error_ptr.hpp"
#include "kmm/utils/poll.hpp"
#include "kmm/worker/buffer_registry.hpp"
#include "kmm/worker/device_stream_manager.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/scheduler.hpp"

namespace kmm {

struct DeviceState {
    GPUContextHandle context;
    DeviceStream stream;
    DeviceEvent last_event;
    DeviceContext device;

    DeviceState(DeviceId id, GPUContextHandle context, DeviceStreamManager& stream_manager) :
        context(context),
        stream(stream_manager.create_stream(context)),
        device(DeviceInfo(id, context), context, stream_manager.get(stream)) {}
};

class Executor {
    KMM_NOT_COPYABLE_OR_MOVABLE(Executor)

  public:
    class Job {
        KMM_NOT_COPYABLE_OR_MOVABLE(Job)

      public:
        Job() = default;
        virtual ~Job() = default;
        virtual Poll poll(Executor& executor, Scheduler& scheduler) = 0;

        //      private:
        EventId m_id;
        std::unique_ptr<Job> next = nullptr;
    };

    Executor(
        std::vector<GPUContextHandle> contexts,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<MemorySystem> memory_system
    );

    ~Executor();

    bool is_idle() const;
    void make_progress(Scheduler& scheduler);

    DeviceState& device_state(DeviceId id, const DeviceEventSet& hint_deps = {});

    DeviceStreamManager& stream_manager() {
        return *m_stream_manager;
    }

    MemoryRequestList create_requests(const std::vector<BufferRequirement>& buffers);
    Poll poll_requests(const MemoryRequestList& requests, DeviceEventSet* dependencies);
    std::vector<BufferAccessor> access_requests(const MemoryRequestList& requests);
    void release_requests(MemoryRequestList& requests, DeviceEvent event = {});

    void poison_buffers(
        const std::vector<BufferRequirement>& buffers,
        EventId event_id,
        const std::exception& err
    );

    void execute_command(EventId id, const Command& command, DeviceEventSet dependencies);

  private:
    void insert_job(std::unique_ptr<Job> job);
    void execute_command(EventId id, const CommandExecute& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandCopy& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandReduction& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandFill& command, DeviceEventSet dependencies);

    std::unique_ptr<Job> m_jobs_head = nullptr;
    Job* m_jobs_tail = nullptr;

    std::unique_ptr<BufferRegistry> m_buffer_registry;
    std::unique_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::vector<std::unique_ptr<DeviceState>> m_devices;
};

}  // namespace kmm