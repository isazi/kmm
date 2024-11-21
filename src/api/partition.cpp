#include "spdlog/spdlog.h"

#include "kmm/api/partition.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

TaskPartition TaskPartitioner::operator()(WorkDim index_space, const SystemInfo& info) const {
    std::vector<TaskChunk> chunks;
    size_t num_devices = info.num_devices();

    if (index_space.is_empty()) {
        return {chunks};
    }

    if (num_devices == 0) {
        throw std::runtime_error("no GPU devices found, cannot partition work");
    }

    if (m_chunk_size.is_empty()) {
        throw std::runtime_error(fmt::format("invalid chunk size: {}", m_chunk_size));
    }

    WorkDim chunk_size;
    std::array<int64_t, WORK_DIMS> num_chunks;

    for (size_t i = 0; i < WORK_DIMS; i++) {
        if (m_chunk_size[i] < index_space[i]) {
            chunk_size[i] = m_chunk_size[i];
            num_chunks[i] = div_ceil(index_space[i], chunk_size[i]);
        } else {
            chunk_size[i] = index_space[i];
            num_chunks[i] = 1;
        }
    }

    size_t owner_id = 0;
    auto offset = WorkIndex {};
    auto size = WorkDim {};

    for (int64_t z = 0; z < num_chunks[2]; z++) {
        for (int64_t y = 0; y < num_chunks[1]; y++) {
            for (int64_t x = 0; x < num_chunks[0]; x++) {
                auto current = Point<3> {x, y, z};

                for (size_t i = 0; i < WORK_DIMS; i++) {
                    offset[i] = current[i] * chunk_size[i];
                    size[i] = std::min(chunk_size[i], index_space[i] - offset[i]);
                }

                chunks.push_back({DeviceId(owner_id), offset, size});
                owner_id = (owner_id + 1) % num_devices;
            }
        }
    }

    return {std::move(chunks)};
}

}  // namespace kmm