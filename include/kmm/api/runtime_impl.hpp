#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include "kmm/core/copy_specification.hpp"
#include "kmm/core/device_info.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/task.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class Buffer;
class Reduction;
class Scheduler;
class MemoryManager;

struct TaskRequirements {
    std::vector<std::shared_ptr<Buffer>> inputs;
    std::vector<BufferLayout> outputs;
    std::vector<std::shared_ptr<Reduction>> reductions;
};

struct TaskResult {
    EventId event_id;
    std::vector<std::shared_ptr<Buffer>> outputs;
};

class RuntimeImpl: std::enable_shared_from_this<RuntimeImpl> {
    KMM_NOT_COPYABLE_OR_MOVABLE(RuntimeImpl)

  public:
    RuntimeImpl(std::shared_ptr<Scheduler> scheduler, std::shared_ptr<MemoryManager> memory);

    TaskResult enqueue_host_task(std::shared_ptr<HostTask> task, const TaskRequirements& reqs)
        const;

    TaskResult enqueue_device_task(
        DeviceId device_id,
        std::shared_ptr<DeviceTask> task,
        const TaskRequirements& reqs) const;

    std::shared_ptr<Buffer> enqueue_copies(
        MemoryId memory_id,
        BufferLayout descriptor,
        std::vector<CopySpecification> operations,
        std::vector<std::shared_ptr<Buffer>> buffers);

    bool query_event(
        EventId event_id,
        std::chrono::system_clock::time_point deadline =
            std::chrono::system_clock::time_point::max()) const;

    EventId insert_barrier() const;
    EventId join_events(const EventList& events) const;
    void delete_buffer(BufferId id, const EventList& dep) const;

    const std::vector<DeviceInfo>& devices() const;

  private:
    mutable std::mutex m_lock;
    std::vector<DeviceInfo> m_devices;
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<MemoryManager> m_memory;
};

class Buffer {
    KMM_NOT_COPYABLE_OR_MOVABLE(Buffer)

  public:
    Buffer(
        std::shared_ptr<const RuntimeImpl> runtime,
        BufferId id,
        MemoryId owner,
        EventId epoch_event,
        BufferLayout layout);
    ~Buffer();

    BufferId id() const {
        return m_id;
    }

    EventId epoch_event() const {
        return m_epoch;
    }

    std::shared_ptr<const RuntimeImpl> runtime() const {
        return m_runtime;
    }

  private:
    friend class RuntimeImpl;

    std::shared_ptr<const RuntimeImpl> m_runtime;
    BufferId m_id;
    MemoryId m_owner;
    EventId m_epoch;
    EventList m_users;
    BufferLayout m_layout;
};

class Reduction {
    KMM_NOT_COPYABLE_OR_MOVABLE(Reduction)
    struct Fragment {
        BufferId id;
        EventList m_users;
    };

  public:
    Reduction(std::shared_ptr<RuntimeImpl> runtime);
    ~Reduction();

    std::shared_ptr<RuntimeImpl> runtime() const {
        return m_runtime;
    }

  private:
    friend class RuntimeImpl;

    std::shared_ptr<RuntimeImpl> m_runtime;
};

}  // namespace kmm