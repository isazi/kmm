#pragma once

#include <variant>

#include "kmm/executor.hpp"
#include "kmm/identifiers.hpp"

namespace kmm {

struct EmptyCommand {};

struct ExecuteCommand {
    ExecutorId executor_id;
    std::shared_ptr<Task> task;
    std::vector<TaskInput> inputs;
    std::vector<TaskOutput> outputs;
};

struct BlockDeleteCommand {
    BlockId id;
};

struct BlockPrefetchCommand {
    MemoryId memory_id;
    BlockId block_id;
};

using Command =
    std::variant<EmptyCommand, ExecuteCommand, BlockDeleteCommand, BlockPrefetchCommand>;

inline const char* format_as(const Command& cmd) {
    switch (cmd.index()) {
        case 0:
            return "EmptyCommand";
        case 1:
            return "ExecuteCommand";
        case 2:
            return "BlockDeleteCommand";
        case 3:
            return "BlockPrefetchCommand";
        default:
            return "???";
    }
}
}  // namespace kmm