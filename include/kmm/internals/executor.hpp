#pragma once

#include <future>

#include "buffer_manager.hpp"
#include "device_stream_manager.hpp"
#include "memory_manager.hpp"
#include "scheduler.hpp"

#include "kmm/core/device_context.hpp"
#include "kmm/utils/poll.hpp"

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
        virtual Poll poll(Executor& executor) = 0;

        //      private:
        std::unique_ptr<Job> next = nullptr;
    };

    Executor(
        std::vector<GPUContextHandle> contexts,
        std::shared_ptr<DeviceStreamManager> stream_manager,
        std::shared_ptr<MemorySystem> memory_system
    );

    ~Executor();

    bool is_idle() const;
    bool is_completed(EventId event_id) const;
    void make_progress();
    void submit_command(EventId id, Command command, EventList deps);

    DeviceState& device_state(DeviceId id, const DeviceEventSet& hint_deps = {});

    DeviceStreamManager& stream_manager() {
        return *m_stream_manager;
    }

    Scheduler& scheduler() {
        return *m_scheduler;
    }

    MemoryRequestList create_requests(const std::vector<BufferRequirement>& buffers);
    Poll poll_requests(const MemoryRequestList& requests, DeviceEventSet* dependencies);
    std::vector<BufferAccessor> access_requests(const MemoryRequestList& requests);
    void release_requests(MemoryRequestList& requests, DeviceEvent event = {});

  private:
    void insert_job(std::unique_ptr<Job> job);
    void execute_command(EventId id, const Command& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandExecute& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandCopy& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandReduction& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandFill& command, DeviceEventSet dependencies);

    std::unique_ptr<Job> m_jobs_head = nullptr;
    Job* m_jobs_tail = nullptr;

    std::unique_ptr<BufferManager> m_buffer_manager;
    std::unique_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<DeviceStreamManager> m_stream_manager;
    std::unique_ptr<Scheduler> m_scheduler;
    std::vector<std::unique_ptr<DeviceState>> m_devices;
};

}  // namespace kmm