#pragma once

#include <cstddef>
#include <vector>

#include "kmm/core/identifiers.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/core/work_range.hpp"

namespace kmm {

struct TaskChunk {
    ProcessorId owner_id;
    WorkIndex offset;
    WorkDim size;
};

struct TaskPartition {
    std::vector<TaskChunk> chunks;
};

struct TaskPartitioner {
    TaskPartitioner(WorkDim chunk_size) : m_chunk_size(chunk_size) {}
    TaskPartitioner(
        int64_t x,
        int64_t y = std::numeric_limits<int64_t>::max(),
        int64_t z = std::numeric_limits<int64_t>::max()
    ) :
        m_chunk_size(x, y, z) {}

    TaskPartition operator()(WorkDim index_space, const SystemInfo& info) const;

  private:
    WorkDim m_chunk_size;
};

}  // namespace kmm