#pragma once

#include "kmm/memory_manager.hpp"

namespace kmm {

void MemoryManager::create_buffer(BufferId id, BufferLayout layout) {}

void MemoryManager::delete_buffer(BufferId id) {}

std::shared_ptr<BufferRequest> MemoryManager::request_access(BufferId buffer_id, AccessMode mode) {}

void MemoryManager::release_access(std::shared_ptr<BufferRequest> request) {}

}  // namespace kmm