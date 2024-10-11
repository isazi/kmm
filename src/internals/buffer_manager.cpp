#include "fmt/format.h"

#include "kmm/internals/buffer_manager.hpp"

namespace kmm {

void BufferManager::add(BufferId id, std::shared_ptr<MemoryManager::Buffer> buffer) {
    m_buffers.emplace(id, BufferMeta {.buffer = buffer});
}

std::shared_ptr<MemoryManager::Buffer> BufferManager::get(BufferId id) {
    auto it = m_buffers.find(id);

    // Buffer not found, ignore
    if (it == m_buffers.end()) {
        throw std::runtime_error(fmt::format("could not retrieve buffer {}: buffer not found", id));
    }

    auto& meta = it->second;

    // If poisoned, throw exception
    if (meta.poison_reason) {
        throw PoisonException(*meta.poison_reason);
    }

    return meta.buffer;
}

void BufferManager::poison(BufferId id, EventId event_id, const std::exception& error) {
    auto it = m_buffers.find(id);

    // Buffer not found, ignore
    if (it == m_buffers.end()) {
        return;
    }

    auto& meta = it->second;

    // Buffer already poisoned, ignore
    if (meta.poison_reason.has_value()) {
        return;
    }

    if (const auto* p = dynamic_cast<const PoisonException*>(&error)) {
        meta.poison_reason = *p;
    } else {
        meta.poison_reason = PoisonException(event_id, error.what());
    }
}

std::shared_ptr<MemoryManager::Buffer> BufferManager::remove(BufferId id) {
    auto it = m_buffers.find(id);

    if (it == m_buffers.end()) {
        throw std::runtime_error(fmt::format("could not remove buffer {}: buffer not found", id));
    }

    auto buffer = std::move(it->second.buffer);
    m_buffers.erase(it);

    return buffer;
}

PoisonException::PoisonException(EventId event_id, const std::string& error) {
    m_message = fmt::format("task {} failed due to error: {}", event_id, error);
}

const char* PoisonException::what() const noexcept {
    return m_message.c_str();
}
}  // namespace kmm
