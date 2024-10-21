#pragma once

#include <future>

#include "buffer_manager.hpp"
#include "cuda_stream_manager.hpp"
#include "memory_manager.hpp"
#include "scheduler.hpp"

#include "kmm/core/cuda_device.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

struct DeviceState {
    CudaStream stream;
    DeviceEvent last_event;
    CudaDevice device;
};

class Executor {
    KMM_NOT_COPYABLE_OR_MOVABLE(Executor)

  public:
    class Operation {
        KMM_NOT_COPYABLE_OR_MOVABLE(Operation)

      public:
        Operation() = default;
        virtual ~Operation() = default;
        virtual Poll poll(Executor& executor) = 0;

        //      private:
        std::unique_ptr<Operation> next = nullptr;
    };

    Executor(
        std::vector<CudaContextHandle> contexts,
        std::shared_ptr<CudaStreamManager> stream_manager,
        std::shared_ptr<MemoryManager> memory_manager
    );

    ~Executor();

    bool is_idle() const;
    bool is_completed(EventId event_id) const;
    void make_progress();
    void submit_command(EventId id, Command command, EventList deps);

    DeviceState& device_state(DeviceId id, const DeviceEventSet& hint_deps = {});

    CudaStreamManager& stream_manager() {
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
    void insert_operation(std::unique_ptr<Operation> op);
    void execute_command(EventId id, const Command& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandExecute& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandCopy& command, DeviceEventSet dependencies);
    void execute_command(EventId id, const CommandReduction& command, DeviceEventSet dependencies);

    std::unique_ptr<Operation> m_operation_head = nullptr;
    Operation* m_operation_tail = nullptr;

    std::shared_ptr<BufferManager> m_buffer_manager;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<CudaStreamManager> m_stream_manager;
    std::shared_ptr<Scheduler> m_scheduler;
    std::vector<DeviceState> m_devices;
};

}  // namespace kmm