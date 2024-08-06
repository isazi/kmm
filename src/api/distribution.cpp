#include "kmm/api/distribution.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template<size_t N>
Partition<N> ChunkPartition<N>::operator()(RuntimeImpl* runtime) const {
    const auto& devices = runtime->devices();
    auto result = std::vector<Chunk<N>> {};
    auto current = point<N>::zero();

    if (global_size.is_empty()) {
        return {global_size, result};
    }

    if (devices.empty()) {
        throw std::runtime_error("no CUDA devices found");
    }

    if (chunk_size.is_empty()) {
        throw std::runtime_error("chunk size must be non-zero");
    }

    dim<N> num_chunks;
    size_t num_total_chunks = 1;

    for (size_t i = 0; i < N; i++) {
        num_chunks[i] = div_ceil(global_size[i], chunk_size[i]);
        num_total_chunks *= checked_cast<size_t>(num_chunks[i]);
    }

    point<N> index = {};

    for (size_t i = 0; i < num_total_chunks; i++) {
        auto region =
            rect(global_size).intersection(rect<N> {index * chunk_size.to_point(), chunk_size});

        auto device_id = devices[result.size() % devices.size()].device_id();
        result.push_back({global_size, region, device_id});

        for (size_t j = 0; j < N; j++) {
            index[j]++;

            if (index[j] >= num_chunks[j]) {
                index[j] = 0;
            } else {
                break;
            }
        }
    }

    return {global_size, result};
}

template struct ChunkPartition<0>;
template struct ChunkPartition<1>;
template struct ChunkPartition<2>;
template struct ChunkPartition<3>;
template struct ChunkPartition<4>;

}  // namespace kmm