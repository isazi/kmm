#pragma once

#include <cstddef>

namespace kmm {

enum struct DeviceMemoryKind {
    DefaultPool,
    PrivatePool,
    NoPool,
};

struct WorkerConfig {
    size_t host_memory_limit = std::numeric_limits<size_t>::max();
    size_t host_memory_block_size = 0;
    DeviceMemoryKind device_memory_kind = DeviceMemoryKind::DefaultPool;
    size_t device_memory_limit = std::numeric_limits<size_t>::max();
    size_t device_memory_block_size = 0;
};

WorkerConfig default_config_from_environment();

}  // namespace kmm