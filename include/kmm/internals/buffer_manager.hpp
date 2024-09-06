#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "kmm/internals/memory_manager.hpp"

namespace kmm {

class PoisonException: public std::exception {
  public:
    PoisonException(EventId, const std::string& error);
    const char* what() const noexcept override;

  private:
    std::string m_message;
};

class BufferManager {
  public:
    void add(BufferId id, std::shared_ptr<MemoryManager::Buffer> buffer);
    std::shared_ptr<MemoryManager::Buffer> get(BufferId id);
    void poison(BufferId id, EventId event_id, const std::exception& error);
    std::shared_ptr<MemoryManager::Buffer> remove(BufferId id);

  private:
    struct BufferMeta {
        std::shared_ptr<MemoryManager::Buffer> buffer;
        std::optional<PoisonException> poison_reason;
    };

    std::unordered_map<BufferId, BufferMeta> m_buffers;
};

}  // namespace kmm