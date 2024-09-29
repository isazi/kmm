#include "spdlog/spdlog.h"

#include "kmm/api/partition.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

Partition ChunkPartitioner::operator()(WorkDim index_space, const SystemInfo& info) const {
    std::vector<Chunk> chunks;
    size_t num_devices = info.num_devices();

    if (num_devices == 0) {
        throw std::runtime_error("no CUDA devices found, cannot partition work");
    }

    WorkDim chunk_size;
    std::array<size_t, WORK_DIMS> num_chunks;
    WorkIndex current;

    for (size_t i = 0; i < WORK_DIMS; i++) {
        if (m_chunk_size[i] < index_space[i]) {
            chunk_size[i] = m_chunk_size[i];
            num_chunks[i] = div_ceil(index_space[i], chunk_size[i]);
        } else {
            chunk_size[i] = index_space[i];
            num_chunks[i] = 1;
        }
    }

    bool is_done = false;
    size_t owner_id = 0;

    while (!is_done) {
        auto offset = WorkIndex {};
        auto size = WorkDim {};

        for (size_t i = 0; i < WORK_DIMS; i++) {
            offset[i] = current[i] * chunk_size[i];
            size[i] = std::min(chunk_size[i], index_space[i] - offset[i]);
        }

        chunks.push_back(Chunk {DeviceId(owner_id), offset, size});
        owner_id = (owner_id + 1) % num_devices;

        is_done = true;

        for (size_t i = 0; i < WORK_DIMS; i++) {
            current[i] += 1;

            if (current[i] < num_chunks[i]) {
                is_done = false;
                break;
            }

            current[i] = 0;
        }
    }

    return {std::move(chunks)};
}

}  // namespace kmm