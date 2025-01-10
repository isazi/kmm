#include "kmm/kmm.hpp"

__global__ void initialize_matrix_kernel(
    kmm::NDRange chunk,
    kmm::gpu_subview_mut<float, 2> matrix
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        matrix[i][j] = 1.0f;
    }
}

__global__ void sum_total_kernel(
    kmm::NDRange chunk,
    kmm::gpu_subview<float, 2> matrix,
    kmm::gpu_subview_mut<float, 2> sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        sum[i][j] += matrix[i][j];
    }
}

__global__ void sum_rows_kernel(
    kmm::NDRange chunk,
    kmm::gpu_subview<float, 2> matrix,
    kmm::gpu_subview_mut<float, 2> rows_sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        rows_sum[i][j] += matrix[i][j];
    }
}

__global__ void sum_cols_kernel(
    kmm::NDRange chunk,
    kmm::gpu_subview<float, 2> matrix,
    kmm::gpu_subview_mut<float, 2> cols_sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        cols_sum[j][i] += matrix[i][j];
    }
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    int width = 32768;
    int height = 32768;
    int chunk_width = width / 8;
    int chunk_height = height / 8;

    auto rt = kmm::make_runtime();
    auto matrix = kmm::Array<float, 2> {{height, width}};

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::GPUKernel(initialize_matrix_kernel, {16, 16}),
        write(matrix, slice(_y, _x))
    );

    rt.synchronize();

    auto total_sum = kmm::Scalar<float>();
    auto rows_sum = kmm::Array<float>(height);
    auto cols_sum = kmm::Array<float>(width);

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::GPUKernel(sum_total_kernel, {16, 16}),
        matrix(_y, _x),
        reduce(kmm::Reduction::Sum, privatize(_y, _x), total_sum)
    );

    rt.synchronize();

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::GPUKernel(sum_rows_kernel, {16, 16}),
        matrix(_y, _x),
        reduce(kmm::Reduction::Sum, privatize(_y), rows_sum(_x))
    );

    rt.synchronize();

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::GPUKernel(sum_cols_kernel, {16, 16}),
        matrix(_y, _x),
        reduce(kmm::Reduction::Sum, privatize(_x), cols_sum(_y))
    );

    rt.synchronize();

    return EXIT_SUCCESS;
}