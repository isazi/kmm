#include <functional>
#include <future>
#include <variant>

#include "fmt/format.h"
#include "kmm/executor.hpp"
#include "kmm/object_manager.hpp"
#include "kmm/types.hpp"

namespace kmm {

struct BufferRequirement {
    PhysicalBufferId buffer_id;
    DeviceId memory_id;
    bool is_write;
};

struct CommandExecute {
    std::optional<ObjectId> output_object_id;
    DeviceId device_id;
    std::shared_ptr<Task> task;
    std::vector<BufferRequirement> buffers;
};

struct CommandNoop {};

struct CommandBufferCreate {
    PhysicalBufferId id;
    BufferLayout description;
};

struct CommandBufferDelete {
    PhysicalBufferId id;
};

struct CommandObjectCreate {
    ObjectId id;
    ObjectHandle object;
};

struct CommandObjectDelete {
    ObjectId id;
};

struct CommandPromise {
    mutable std::promise<void> promise;
};

using Command = std::variant<
    CommandNoop,
    CommandPromise,
    CommandExecute,
    CommandBufferCreate,
    CommandBufferDelete,
    CommandObjectCreate,
    CommandObjectDelete>;

static const char* format_as(const Command& cmd) {
    switch (cmd.index()) {
        case 0:
            return "CommandNoop";
        case 1:
            return "CommandPromise";
        case 2:
            return "CommandExecute";
        case 3:
            return "CommandBufferCreate";
        case 4:
            return "CommandBufferDelete";
        case 5:
            return "CommandObjectCreate";
        case 6:
            return "CommandObjectDelete";
        default:
            return "???";
    }
}

struct CommandPacket {
    CommandPacket(OperationId id, Command command, std::vector<OperationId> dependencies = {}) :
        id(id),
        command(std::move(command)),
        dependencies(std::move(dependencies)) {}

    OperationId id;
    Command command;
    std::vector<OperationId> dependencies;
};
}  // namespace kmm