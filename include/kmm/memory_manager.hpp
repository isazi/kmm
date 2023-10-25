#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>

#include "kmm/types.hpp"

namespace kmm {

class Allocation {
  public:
    virtual ~Allocation() = default;
};

class MemoryRequest;

class MemoryManager {
  public:
    void create_buffer(BufferId buffer_id, const BufferDescription&);
    void delete_buffer(BufferId buffer_id);

    std::shared_ptr<MemoryRequest> acquire_buffer(
        BufferId buffer_id,
        DeviceId device_id,
        bool writable,
        std::shared_ptr<void> token);
    std::shared_ptr<Allocation> view_buffer(std::shared_ptr<MemoryRequest>);
    void release_buffer(std::shared_ptr<MemoryRequest>, std::optional<std::string> poison_reason);

    std::optional<std::shared_ptr<MemoryRequest>> poll();

  private:
    static constexpr size_t MAX_DEVICES = 5;

    struct Entry {
        bool is_valid = false;
        bool is_allocated = false;
        std::optional<std::shared_ptr<Allocation>> data = {};
        size_t num_locks = 0;
    };

    struct State {
        BufferId id;
        size_t num_bytes;
        std::deque<std::shared_ptr<MemoryRequest>> waiters;
        std::array<Entry, MAX_DEVICES> entries;
    };

    std::unordered_map<BufferId, State> m_buffers;
};

}  // namespace kmm