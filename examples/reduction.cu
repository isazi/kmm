#include "kmm/kmm.hpp"

__global__ void initialize_matrix_kernel(
    kmm::WorkChunk chunk,
    kmm::cuda_subview_mut<float, 2> matrix
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.begin.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.begin.x;

    if (i < chunk.end.y && j < chunk.end.x) {
        matrix[i][j] = 1.0f;
    }
}

__global__ void sum_total_kernel(
    kmm::WorkChunk chunk,
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
    kmm::WorkChunk chunk,
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
    kmm::WorkChunk chunk,
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

    int width = 4096;
    int height = 4096;
    int chunk_width = 1024;
    int chunk_height = 1024;

    auto rt = kmm::make_runtime();
    auto matrix = kmm::Array<float, 2>{{height, width}};

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(initialize_matrix_kernel, {16, 16}),
        kmm::write(matrix, slice(_y, _x))
    );

    matrix.synchronize();

    auto total_sum = kmm::Array<float, 0>();
    auto rows_sum = kmm::Array<float>(height);
    auto cols_sum = kmm::Array<float>(width);

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(sum_total_kernel, {16, 16}),
        read(matrix, slice(_y, _x)),
        reduce(total_sum, kmm::ReductionOp::Sum, privatize(_y, _x))
    );

    rt.synchronize();

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(sum_rows_kernel, {16, 16}),
        read(matrix, slice(_y, _x)),
        reduce(rows_sum, kmm::ReductionOp::Sum, privatize(_y), slice(_x))
    );

    rt.synchronize();

    rt.parallel_submit(
        {width, height},
        {chunk_width, chunk_height},
        kmm::CudaKernel(sum_cols_kernel, {16, 16}),
        read(matrix, slice(_y, _x)),
        reduce(cols_sum, kmm::ReductionOp::Sum, privatize(_x), slice(_y))
    );

    rt.synchronize();

    return EXIT_SUCCESS;
}