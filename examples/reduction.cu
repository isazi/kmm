#include "kmm/kmm.hpp"

__global__ void initialize_matrix_kernel(
    kmm::WorkRange chunk,
    kmm::cuda_subview_mut<float, 2> matrix
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        matrix[i][j] = 1.0f;
    }
}

__global__ void sum_total_kernel(
    kmm::WorkRange chunk,
    kmm::cuda_subview<float, 2> matrix,
    kmm::cuda_subview_mut<float, 2> sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        sum[i][j] += matrix[i][j];
    }
}

__global__ void sum_rows_kernel(
    kmm::WorkRange chunk,
    kmm::cuda_subview<float, 2> matrix,
    kmm::cuda_subview_mut<float, 2> rows_sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        rows_sum[i][j] += matrix[i][j];
    }
}

__global__ void sum_cols_kernel(
    kmm::WorkRange chunk,
    kmm::cuda_subview<float, 2> matrix,
    kmm::cuda_subview_mut<float, 2> cols_sum
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
    auto matrix = kmm::Array<float, 2>{{height, width}};

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(initialize_matrix_kernel, {16, 16}),
        write(matrix, access(_y, _x))
    );

    rt.synchronize();

    auto total_sum = kmm::Scalar<float>();
    auto rows_sum = kmm::Array<float>(height);
    auto cols_sum = kmm::Array<float>(width);

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(sum_total_kernel, {16, 16}),
        read(matrix, access(_y, _x)),
        reduce(total_sum, kmm::ReductionOp::Sum, privatize(_y, _x))
    );

    rt.synchronize();

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(sum_rows_kernel, {16, 16}),
        read(matrix, access(_y, _x)),
        reduce(rows_sum, kmm::ReductionOp::Sum, privatize(_y), access(_x))
    );

    rt.synchronize();

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(sum_cols_kernel, {16, 16}),
        read(matrix, access(_y, _x)),
        reduce(cols_sum, kmm::ReductionOp::Sum, privatize(_x), access(_y))
    );

    rt.synchronize();

    return EXIT_SUCCESS;
}