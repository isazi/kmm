
#include "spdlog/spdlog.h"

#include "kmm/memops/cuda_fill.hpp"
#include "kmm/utils/cuda.hpp"

namespace kmm {

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
    size_t fill_pattern_size) {
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
            remainder);

        nbytes -= remainder;
    }

    if (is_fill_pattern_repetitive<1>(fill_pattern, fill_pattern_size)) {
        uint8_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint8_t));
        KMM_CUDA_CHECK(cuMemsetD8Async(  //
            CUdeviceptr(dest_buffer),
            pattern,
            nbytes,
            stream));

    } else if (is_fill_pattern_repetitive<2>(fill_pattern, fill_pattern_size)) {
        uint16_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint16_t));
        KMM_CUDA_CHECK(cuMemsetD16Async(  //
            CUdeviceptr(dest_buffer),
            pattern,
            nbytes / sizeof(uint16_t),
            stream));

    } else if (is_fill_pattern_repetitive<4>(fill_pattern, fill_pattern_size)) {
        uint32_t pattern;
        ::memcpy(&pattern, fill_pattern, sizeof(uint32_t));
        KMM_CUDA_CHECK(cuMemsetD32Async(  //
            CUdeviceptr(dest_buffer),
            pattern,
            nbytes / sizeof(uint32_t),
            stream));

    } else {
        throw CudaException(fmt::format(
            "could not fill buffer, value is {} bit, but only 8, 16, or 32 bit is supported",
            fill_pattern_size * 8));
    }
}

}  // namespace kmm
