#pragma once

#include <variant>

#include "kmm/core/buffer.hpp"
#include "kmm/core/copy_description.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

struct CommandBufferCreate {
    BufferId id;
    BufferLayout layout;
};

struct CommandBufferDelete {
    BufferId id;
};

struct CommandPrefetch {
    BufferId buffer_id;
    MemoryId memory_id;
};

struct CommandCopy {
    BufferId src_buffer;
    MemoryId src_memory;
    BufferId dst_buffer;
    MemoryId dst_memory;
    CopyDescription spec;
};

struct CommandExecuteHost {
    std::shared_ptr<HostTask> task;
    std::vector<BufferRequirement> buffers;
};

struct CommandExecuteDevice {
    DeviceId device_id;
    std::shared_ptr<DeviceTask> task;
    std::vector<BufferRequirement> buffers;
};

struct CommandEmpty {};

using Command = std::variant<
    CommandEmpty,
    CommandBufferCreate,
    CommandBufferDelete,
    CommandPrefetch,
    CommandCopy,
    CommandExecuteHost,
    CommandExecuteDevice>;

}  // namespace kmm