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

class Event {
  public:
    Event(EventId id, std::shared_ptr<RuntimeImpl> runtime) :
        m_id(id),
        m_runtime(std::move(runtime)) {}

    EventId id() const;
    bool is_done() const;
    void wait() const;
    bool wait_until(typename std::chrono::system_clock::time_point) const;
    bool wait_for(typename std::chrono::system_clock::duration) const;

  private:
    EventId m_id;
    std::shared_ptr<RuntimeImpl> m_runtime;
};

class Buffer {
  public:
    Buffer(BlockId id, std::shared_ptr<RuntimeImpl> runtime);
    Buffer(Buffer&&) = delete;
    Buffer(const Buffer&) = default;

    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = default;

    BlockId id() const;
    Runtime runtime() const;
    Event barrier() const;
    void destroy() const;

  private:
    class Lifetime;
    std::shared_ptr<Lifetime> m_lifetime;
};

template<typename T, size_t N = 1>
class Array {
  public:
    Array(std::array<index_t, N> sizes) : m_sizes(sizes) {}
    Array(std::array<index_t, N> sizes, Buffer buffer) : m_sizes(sizes), m_buffer(buffer) {}

    std::array<index_t, N> sizes() const {
        return m_sizes;
    }

    index_t size(size_t axis) const {
        return m_sizes.at(axis);
    }

    index_t size() const {
        return checked_product(m_sizes.begin(), m_sizes.end());
    }

    Buffer buffer() const {
        return m_buffer.value();
    }

    BlockId id() const {
        return m_buffer.value().id();
    }

    Runtime runtime() const;

    ArrayHeader header() const {
        return ArrayHeader::for_type<T>(size());
    }

  private:
    std::optional<Buffer> m_buffer;
    std::array<index_t, N> m_sizes;
};

class Runtime {
  public:
    Runtime(std::shared_ptr<RuntimeImpl> impl);

    Event submit_task(
        std::shared_ptr<Task> task,
        TaskRequirements reqs,
        EventList dependencies = {}) const;

    Event barrier() const;

    template<typename Device, typename... Args>
    Event submit(const Device& device, Args&&... args) {
        EventId id = device(*m_impl, std::forward<Args>(args)...);
        return {id, m_impl};
    }

    Event join(std::vector<Event> events) const;

    template<typename... Es>
    Event join(Es... events) const {
        return join(std::vector<Event> {events...});
    }

    std::shared_ptr<RuntimeImpl> inner() const {
        return m_impl;
    }

  private:
    std::shared_ptr<RuntimeImpl> m_impl;
};

Runtime build_runtime();

}  // namespace kmm