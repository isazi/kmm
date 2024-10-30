#include "kmm/api/mapper.hpp"
#include "kmm/api/runtime.hpp"

__global__ void initialize_range(
    kmm::WorkRange chunk,
    kmm::cuda_subview_mut<float> output
) {
    int64_t i = blockIdx.x * blockDim.x +  threadIdx.x + chunk.begin();
    if (i >= chunk.end()) {
        return;
    }

    output[i] = float(i);
}

__global__ void fill_range(
    kmm::WorkRange chunk,
    float value,
    kmm::cuda_subview_mut<float> output
) {
    int64_t i = blockIdx.x * blockDim.x +  threadIdx.x + chunk.begin();
    if (i >= chunk.end()) {
        return;
    }

    output[i] = value;
}

__global__ void vector_add(
    kmm::WorkRange chunk,
    kmm::cuda_subview_mut<float> output,
    kmm::cuda_subview<float> left,
    kmm::cuda_subview<float> right
) {
    int64_t i = blockIdx.x * blockDim.x +  threadIdx.x + chunk.begin();
    if (i >= chunk.end()) {
        return;
    }

    output[i] = left[i] + right[i];
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int n = 2'000'000'000;
    int chunk_size = n / 10;
    dim3 block_size = 256;

    auto A = kmm::Array<float>{n};
    auto B = kmm::Array<float>{n};
    auto C = kmm::Array<float>{n};

    rt.parallel_submit(
        {n},
        {chunk_size},
        kmm::CudaKernel(initialize_range, block_size),
        write(A, access(_x))
    );

    rt.parallel_submit(
        {n},
        {chunk_size},
        kmm::CudaKernel(fill_range, block_size),
        float(M_PI),
        write(B, access(_x))
    );

    rt.parallel_submit(
        {n},
        {chunk_size},
        kmm::CudaKernel(vector_add, block_size),
        write(C, access(_x)),
        read(A, access(_x)),
        read(B, access(_x))
    );

    std::vector<float> result(n);
    C.copy_to(result.data());

    return 0;
}
