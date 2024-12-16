#pragma once

#include <variant>

#include "fmt/ostream.h"

#include "kmm/core/buffer.hpp"
#include "kmm/core/copy_def.hpp"
#include "kmm/core/fill_def.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/reduction.hpp"
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
    CopyDef definition;
};

struct CommandExecute {
    ProcessorId processor_id;
    std::shared_ptr<Task> task;
    std::vector<BufferRequirement> buffers;
};

struct CommandReduction {
    BufferId src_buffer;
    BufferId dst_buffer;
    MemoryId memory_id;
    ReductionDef definition;
};

struct CommandFill {
    BufferId dst_buffer;
    MemoryId memory_id;
    FillDef definition;
};

struct CommandEmpty {};

using Command = std::variant<
    CommandEmpty,
    CommandBufferCreate,
    CommandBufferDelete,
    CommandPrefetch,
    CommandCopy,
    CommandExecute,
    CommandReduction,
    CommandFill>;

inline const char* command_name(const Command& cmd) {
    static constexpr const char* names[] = {
        "CommandEmpty",
        "CommandBufferCreate",
        "CommandBufferDelete",
        "CommandPrefetch",
        "CommandCopy",
        "CommandExecute",
        "CommandReduction",
        "CommandFill"};

    return names[cmd.index()];
}

inline std::ostream& operator<<(std::ostream& f, const Command& cmd) {
    return f << command_name(cmd);
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::Command>: fmt::ostream_formatter {};