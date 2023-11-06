#include <gtest/gtest.h>

#include "kmm/dag_builder.hpp"

using namespace kmm;

class MockTask: public Task {
    TaskResult execute(ExecutorContext&, TaskContext&) override {
        return {};
    }
};

TEST(TestDAGBuilder, create_delete_buffer) {
    auto builder = DAGBuilder();
    auto layout =
        BufferLayout {.num_bytes = 1, .alignment = 1, .home = DeviceId(42), .name = "test"};

    auto id = builder.create_buffer(layout);

    builder.delete_buffer(id);

    auto commands = builder.flush();
    ASSERT_EQ(commands.size(), 2);

    auto create_cmd = std::get<CommandBufferCreate>(commands.at(0).command);
    ASSERT_EQ(create_cmd.description.name, layout.name);

    auto delete_cmd = std::get<CommandBufferDelete>(commands.at(1).command);
    ASSERT_EQ(commands.at(1).dependencies, std::vector {commands.at(0).id});
    ASSERT_EQ(delete_cmd.id, create_cmd.id);
}

TEST(TestDAGBuilder, submit_task) {
    auto builder = DAGBuilder();
    auto task = std::make_shared<MockTask>();
    auto layout =
        BufferLayout {.num_bytes = 1, .alignment = 1, .home = DeviceId(42), .name = "test"};

    auto buffer_id = builder.create_buffer(layout);

    auto reqs = TaskRequirements {
        .device_id = DeviceId(3),
        .buffers = {VirtualBufferRequirement {
            .buffer_id = buffer_id,
            .mode = AccessMode::Write,
        }}};
    builder.submit_task(task, reqs, {});

    auto reqs2 = TaskRequirements {
        .device_id = DeviceId(4),
        .buffers = {VirtualBufferRequirement {
            .buffer_id = buffer_id,
            .mode = AccessMode::Read,
        }}};
    builder.submit_task(task, reqs2, {});
    builder.delete_buffer(buffer_id);

    auto commands = builder.flush();

    ASSERT_EQ(commands.size(), 4);
    ASSERT_EQ(commands[0].dependencies.size(), 0);
    ASSERT_EQ(commands[1].dependencies, std::vector {commands[0].id});
    ASSERT_EQ(commands[2].dependencies, std::vector {commands[1].id});
    ASSERT_EQ(commands[3].dependencies, (std::vector {commands[2].id, commands[1].id}));

    auto create_cmd = std::get<CommandBufferCreate>(commands.at(0).command);
    ASSERT_EQ(create_cmd.description.name, "test");

    auto exe_write = std::get<CommandExecute>(commands.at(1).command);
    ASSERT_EQ(exe_write.output_object_id.has_value(), false);
    ASSERT_EQ(exe_write.device_id, DeviceId(3));
    ASSERT_EQ(exe_write.task, task);
    ASSERT_EQ(exe_write.buffers.size(), 1);
    ASSERT_EQ(exe_write.buffers[0].buffer_id, create_cmd.id);
    ASSERT_EQ(exe_write.buffers[0].memory_id, DeviceId(3));
    ASSERT_EQ(exe_write.buffers[0].is_write, true);

    auto exe_read = std::get<CommandExecute>(commands.at(2).command);
    ASSERT_EQ(exe_read.output_object_id.has_value(), false);
    ASSERT_EQ(exe_read.device_id, DeviceId(4));
    ASSERT_EQ(exe_read.task, task);
    ASSERT_EQ(exe_read.buffers.size(), 1);
    ASSERT_EQ(exe_read.buffers[0].buffer_id, create_cmd.id);
    ASSERT_EQ(exe_read.buffers[0].memory_id, DeviceId(4));
    ASSERT_EQ(exe_read.buffers[0].is_write, false);

    auto delete_cmd = std::get<CommandBufferDelete>(commands.at(3).command);
    ASSERT_EQ(delete_cmd.id, create_cmd.id);
}

TEST(TestDAGBuilder, join) {
    auto builder = DAGBuilder();

    auto a = builder.join({});
    ASSERT_EQ(a, OperationId::invalid());

    auto b = builder.join({OperationId::invalid()});
    ASSERT_EQ(b, OperationId::invalid());

    auto c = builder.join({
        OperationId(4),
        OperationId(4),
    });
    ASSERT_EQ(c, OperationId(4));

    auto d = builder.join({
        OperationId(4),
        OperationId(5),
    });
    ASSERT_EQ(d, OperationId(1));

    auto commands = builder.flush();
    ASSERT_EQ(commands.size(), 1);
    ASSERT_EQ(commands[0].dependencies, (std::vector {OperationId(4), OperationId(5)}));
}