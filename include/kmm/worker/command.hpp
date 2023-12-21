#pragma once

#include <functional>
#include <future>
#include <variant>

#include "fmt/format.h"

#include "kmm/executor.hpp"
#include "kmm/types.hpp"
#include "kmm/worker/block_manager.hpp"

namespace kmm {

struct ExecuteCommand {
    ExecutorId executor_id;
    std::shared_ptr<Task> task;
    std::vector<TaskInput> inputs;
    std::vector<TaskOutput> outputs;
};

struct EmptyCommand {};

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
            return "CommandNoop";
        case 1:
            return "CommandExecute";
        case 2:
            return "CommandBlockDelete";
        case 3:
            return "CommandPrefetch";
        default:
            return "???";
    }
}
}  // namespace kmm