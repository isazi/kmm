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
    size_t num_total_chunks = 1;

    for (size_t i = 0; i < WORK_DIMS; i++) {
        if (m_chunk_size[i] < index_space[i]) {
            chunk_size[i] = m_chunk_size[i];
            num_chunks[i] = div_ceil(index_space[i], chunk_size[i]);
            num_total_chunks *= checked_cast<size_t>(num_chunks[i]);
        } else {
            chunk_size[i] = index_space[i];
            num_chunks[i] = 1;
        }
    }

    size_t owner_id = 0;

    for (size_t it = 0; it < num_total_chunks; it++) {
        auto offset = WorkIndex {};
        auto size = WorkDim {};

        for (size_t i = 0; i < WORK_DIMS; i++) {
            offset[i] = current[i] * chunk_size[i];
            size[i] = std::min(chunk_size[i], index_space[i] - offset[i]);
        }

        chunks.push_back(Chunk {DeviceId(owner_id), offset, size});
        owner_id = (owner_id + 1) % num_devices;

        for (size_t i = 0; i < WORK_DIMS; i++) {
            current[i] += 1;

            if (current[i] >= num_chunks[i]) {
                current[i] = 0;
            } else {
                break;
            }
        }
    }

    return {std::move(chunks)};
}

}  // namespace kmm