#pragma once

#include <memory>
#include <vector>

#include "kmm/types.hpp"

namespace kmm {

struct BufferState;

struct MemoryManager {
    std::unordered_map<BufferId, std::shared_ptr<BufferRecord>> buffers;
    std::vector<std::shared_ptr<Memory>> memories;
};

}  // namespace kmm