#include "fmt/format.h"

#include "kmm/api/array_base.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

bool ArrayBase::is_empty() const {
    for (size_t i = 0; i < rank(); i++) {
        if (size(i) <= 0) {
            return true;
        }
    }

    return false;
}

void ArrayBase::synchronize() const {
    for (size_t i = 0; i < num_chunks(); i++) {
        const auto& buffer = this->chunk(i);
        buffer->runtime()->query_event(buffer->epoch_event());
    }
}

template<size_t N>
ArrayImpl<N>::ArrayImpl(dim<N> shape, std::vector<ArrayChunk<N>> chunks) {
    auto chunk_size = dim<N>::zero();

    for (const auto& chunk : chunks) {
        if (chunk.offset == point<N>::zero()) {
            chunk_size = chunk.shape;
        }
    }

    if (chunk_size.is_empty()) {
        throw std::runtime_error(
            fmt::format("chunk size cannot be zero, given chunk size {}", chunk_size));
    }

    dim<N> num_chunks;
    size_t total_num_chunks = 1;

    for (size_t i = 0; i < N; i++) {
        num_chunks[i] = div_ceil(shape[i], chunk_size[i]);
        total_num_chunks *= size_t(num_chunks[i]);
    }

    std::vector<std::shared_ptr<Buffer>> buffers {total_num_chunks};

    for (auto& chunk : chunks) {
        point<N> chunk_offset;
        dim<N> chunk_shape;
        size_t index = 0;
        bool is_valid = true;

        for (size_t i = 0; i < N; i++) {
            if (chunk.offset[i] < 0 || chunk.offset[i] >= shape[i]
                || chunk.offset[i] % chunk_size[i] != 0) {
                is_valid = false;
                break;
            }

            int64_t expected_size = std::min(chunk_size[i], shape[i] - chunk.offset[i]);

            if (chunk.shape[i] != expected_size) {
                is_valid = false;
                break;
            }

            int64_t k = chunk.offset[i] / chunk_size[i];
            index = index * size_t(m_num_chunks[i]) + size_t(k);
        }

        if (!is_valid) {
            throw std::runtime_error(fmt::format(
                "invalid chunk specification: data chunk at offset {} with dimensions {} should not exist, it should have aligned to the chunk size {}",
                chunk.offset,
                chunk.shape,
                chunk_size));
        }

        if (buffers[index] != nullptr) {
            throw std::runtime_error(fmt::format(
                "invalid chunk specification: data chunk at offset {} with dimensions {} was written by multiple tasks"));
        }

        buffers[index] = std::move(chunk.buffer);
    }

    for (size_t index = 0; index < buffers.size(); index++) {
        if (buffers[index] != nullptr) {
            continue;
        }

        point<N> offset;
        dim<N> chunk_shape;

        for (size_t i = N; i > 0; i--) {
            size_t k = index % num_chunks[i - 1];
            index /= num_chunks[i - 1];

            offset[i - 1] = k * chunk_size[i - 1];
            chunk_shape[i - 1] = std::min(m_chunk_size[i], shape[i] - offset[i]);
        }

        throw std::runtime_error(fmt::format(
            "invalid chunk specification: data chunk at offset {} with dimensions {} was not written to by any task",
            offset,
            chunk_shape));
    }

    m_buffers = std::move(buffers);
    m_num_chunks = num_chunks;
    m_chunk_size = chunk_size;
    m_global_size = shape;
}

template<size_t N>
ArrayChunk<N> ArrayImpl<N>::find_chunk(const rect<N>& region) const {
    point<N> offset;
    dim<N> shape;
    size_t index = 0;

    for (size_t i = 0; i < N; i++) {
        int64_t k = div_floor(region.begin(i), m_chunk_size[i]);

        if (k < 0 || k >= m_num_chunks[i]) {
            throw std::runtime_error(fmt::format(
                "The requested access region {} is out of bounds, array has dimensions {}",
                region,
                m_global_size));
        }

        int64_t begin = k * m_chunk_size[i];
        int64_t end = std::min((k + 1) * m_chunk_size[i], m_global_size[i]);

        if (region.size(i) > end - region.begin(i)) {
            throw std::runtime_error(fmt::format(
                "The requested access region {} does not lie within a single chunk, array has chunk size {}",
                region,
                m_chunk_size));
        }

        offset[i] = begin;
        shape[i] = end - offset[i];
        index = index * size_t(m_num_chunks[i]) + size_t(k);
    }

    return {.offset = offset, .shape = shape, .buffer = m_buffers[index]};
}

// Instantiate for arrays up to 6 dimensions
template class ArrayImpl<0>;
template class ArrayImpl<1>;
template class ArrayImpl<2>;
template class ArrayImpl<3>;
template class ArrayImpl<4>;
template class ArrayImpl<5>;
template class ArrayImpl<6>;
}  // namespace kmm