#pragma once

#include <cstddef>
#include <vector>

#include "kmm/core/domain.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

struct WorkChunk {
    ProcessorId owner_id;
    NDIndex offset;
    NDSize size;
};

struct WorkPartition {
    std::vector<WorkChunk> chunks;
};

template<typename P>
struct IntoWorkPartition {
    static WorkPartition call(
        P partitioner,
        NDRange index_space,
        const SystemInfo& info,
        ExecutionSpace space
    ) {
        return (partitioner)(index_space, info, space);
    }
};

struct ChunkPartitioner {
    ChunkPartitioner(NDSize chunk_size) : m_chunk_size(chunk_size) {}
    ChunkPartitioner(
        int64_t x,
        int64_t y = std::numeric_limits<int64_t>::max(),
        int64_t z = std::numeric_limits<int64_t>::max()
    ) :
        m_chunk_size(x, y, z) {}

    WorkPartition operator()(NDRange index_space, const SystemInfo& info, ExecutionSpace space)
        const;

  private:
    NDSize m_chunk_size;
};

}  // namespace kmm
