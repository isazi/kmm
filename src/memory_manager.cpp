#include "kmm/memory_manager.hpp"

namespace kmm {

void MemoryManager::create_buffer(BufferId id) {}

void MemoryManager::delete_buffer(BufferId id) {}
std::shared_ptr<MemoryRequest> MemoryManager::acquire_buffer(
    BufferId buffer_id,
    DeviceId device_id,
    bool writable,
    std::shared_ptr<void> token) {
    return std::shared_ptr<MemoryRequest>();
}

std::shared_ptr<Allocation> MemoryManager::view_buffer(std::shared_ptr<MemoryRequest>) {
    return std::shared_ptr<Allocation>();
}

void MemoryManager::release_buffer(
    std::shared_ptr<MemoryRequest>,
    std::optional<std::string> poison_reason) {}

}  // namespace kmm