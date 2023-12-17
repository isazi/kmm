#pragma once

#include <functional>
#include <future>
#include <variant>

#include "fmt/format.h"

#include "kmm/block_manager.hpp"
#include "kmm/executor.hpp"
#include "kmm/types.hpp"

namespace kmm {

struct CommandExecute {
    DeviceId device_id;
    std::shared_ptr<Task> task;
    std::vector<TaskInput> inputs;
    std::vector<TaskOutput> outputs;
};

struct CommandNoop {};

struct CommandBlockDelete {
    BlockId id;
};

using Command = std::variant<CommandNoop, CommandExecute, CommandBlockDelete>;

static const char* format_as(const Command& cmd) {
    switch (cmd.index()) {
        case 0:
            return "CommandNoop";
        case 1:
            return "CommandExecute";
        case 2:
            return "CommandBlockDelete";
        default:
            return "???";
    }
}
}  // namespace kmm