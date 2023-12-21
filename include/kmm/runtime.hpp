#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmm/executor.hpp"
#include "kmm/types.hpp"
#include "kmm/utils.hpp"

namespace kmm {

class Runtime {
  public:
    Runtime(std::shared_ptr<RuntimeImpl> impl);

    EventId submit_task(
        std::shared_ptr<Task> task,
        TaskRequirements reqs,
        EventList dependencies = {}) const;

    template<typename Device, typename... Args>
    EventId submit(const Device& device, Args&&... args) {
        return device(*m_impl, std::forward<Args>(args)...);
    }

    EventId join(EventList events) const;

    template<typename... Es>
    EventId join(Es... events) const {
        return join(EventList {events...});
    }

    bool is_done(EventId id) const;
    void wait(EventId id) const;
    bool wait_until(EventId id, typename std::chrono::system_clock::time_point deadline) const;
    bool wait_for(EventId id, typename std::chrono::system_clock::duration duration) const;
    EventId submit_barrier() const;
    void synchronize() const;

    std::shared_ptr<RuntimeImpl> inner() const {
        return m_impl;
    }

  private:
    std::shared_ptr<RuntimeImpl> m_impl;
};

class Buffer {
  public:
    Buffer(std::shared_ptr<RuntimeImpl> runtime, BlockId id = BlockId::invalid());
    ~Buffer();

    Buffer(Buffer&&) = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    BlockId id() const {
        return m_id;
    }

    Runtime runtime() const {
        return m_runtime;
    }

    EventId prefetch(MemoryId memory_id, EventList dependencies = {}) const;
    EventId submit_barrier() const;
    void synchronize() const;
    void destroy();

  private:
    BlockId m_id = BlockId::invalid();
    std::shared_ptr<RuntimeImpl> m_runtime;
};

Runtime build_runtime();

}  // namespace kmm