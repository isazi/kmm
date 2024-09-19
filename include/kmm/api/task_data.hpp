#pragma once

#include "kmm/api/access.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Cuda };

template<typename T>
struct TaskDataProcessor;

template<ExecutionSpace, typename T>
struct TaskDataDeserialize;

struct TaskBuilder {
    TaskGraph& graph;
    Worker& worker;
    MemoryId memory_id;
    std::vector<BufferRequirement> buffers;
    EventList dependencies;
};

struct TaskResult {
    TaskGraph& graph;
    Worker& worker;
    EventList events;
};

template<typename T, typename = void>
struct TaskData {
    TaskData(T value) : m_value(std::move(value)) {}

    static TaskData pack(TaskBuilder& builder, T value) {
        return {std::move(value)};
    }

    template<ExecutionSpace Space>
    T unpack(TaskContext& context) {
        return m_value;
    }

  private:
    T m_value;
};

template<typename T>
struct TaskDataProcessor {
    using type = TaskData<T>;

    TaskDataProcessor(T value) : m_value(value) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        return TaskData<T>::pack(builder, m_value);
    }

    void finalize(const TaskResult& result) {
        // Nothing to do
    }

  private:
    T m_value;
};

template<ExecutionSpace Space, typename T>
struct TaskDataDeserialize<Space, TaskData<T>> {
    static T unpack(TaskContext& context, TaskData<T>& data) {
        return data.template unpack<Space>(context);
    }
};

}  // namespace kmm