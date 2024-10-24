
#include "spdlog/spdlog.h"

#include "kmm/memops/cuda_fill.hpp"
#include "kmm/utils/cuda.hpp"
#include "kmm/utils/integer_fun.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

template<typename T, uint32_t block_size>
__global__ void fill_kernel(size_t nelements, T* dest_buffer, T fill_value) {
    size_t i = blockIdx.x * size_t(block_size) + threadIdx.x;

    while (i < nelements) {
        dest_buffer[i] = fill_value;
        i += size_t(block_size) * gridDim.x;
    }
}

template<typename T>
void submit_fill_kernel(
    CUstream stream,
    CUdeviceptr dest_buffer,
    size_t nelements,
    const void* fill_pattern
) {
    static constexpr uint32_t max_grid_size = 512;
    static constexpr uint32_t block_size = 256;

    T fill_value;
    ::memcpy(&fill_value, fill_pattern, sizeof(T));

    uint32_t grid_size = nelements < max_grid_size * size_t(block_size)
        ? div_ceil(nelements, size_t(block_size))
        : max_grid_size;

    fill_kernel<T, block_size>
        <<<grid_size, block_size, 0, stream>>>(nelements, (T*)dest_buffer, fill_value);
}

template<size_t N>
bool is_fill_pattern_repetitive(const void* fill_pattern, size_t fill_pattern_size) {
    if (fill_pattern_size % N != 0) {
        return false;
    }

    for (size_t i = 1; i < fill_pattern_size / N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (static_cast<const uint8_t*>(fill_pattern)[i * N + j]
                != static_cast<const uint8_t*>(fill_pattern)[j]) {
                return false;
            }
        }
    }

    return true;
}

void execute_cuda_fill_async(
    CUstream stream,
    CUdeviceptr dest_buffer,
    size_t nbytes,
    const void* fill_pattern,
    size_t fill_pattern_size
) {
    if (nbytes == 0 || fill_pattern_size == 0) {
        return;
    }

    size_t remainder = nbytes % fill_pattern_size;
    if (remainder != 0) {
        execute_cuda_fill_async(
            stream,
            dest_buffer + (nbytes - remainder),
            remainder,
            fill_pattern,
            remainder
        );

        nbytes -= remainder;
    }

    if (is_fill_pattern_repetitive<1>(fill_pattern, fill_pattern_size)) {
        uint8_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint8_t));
        KMM_CUDA_CHECK(cuMemsetD8Async(  //
            CUdeviceptr(dest_buffer),
            pattern,
            nbytes,
            stream
        ));

    } else if (is_fill_pattern_repetitive<2>(fill_pattern, fill_pattern_size)) {
        uint16_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint16_t));
        KMM_CUDA_CHECK(cuMemsetD16Async(  //
            CUdeviceptr(dest_buffer),
            pattern,
            nbytes / sizeof(uint16_t),
            stream
        ));

    } else if (is_fill_pattern_repetitive<4>(fill_pattern, fill_pattern_size)) {
        uint32_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint32_t));
        KMM_CUDA_CHECK(cuMemsetD32Async(  //
            CUdeviceptr(dest_buffer),
            pattern,
            nbytes / sizeof(uint32_t),
            stream
        ));

    } else if (is_fill_pattern_repetitive<8>(fill_pattern, fill_pattern_size)) {
        KMM_ASSERT(dest_buffer % 8 == 0);  // must be aligned?
        submit_fill_kernel<uint64_t>(stream, dest_buffer, nbytes / sizeof(uint64_t), fill_pattern);
    } else {
        throw CudaException(fmt::format(
            "could not fill buffer, value is {} bits, but only 8, 16, 32 or 64 bit is supported",
            fill_pattern_size * 8
        ));
    }
}

}  // namespace kmm
