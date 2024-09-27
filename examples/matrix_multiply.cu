#include "kmm/kmm.hpp"

void fill_array(
    kmm::WorkChunk region,
    kmm::subview_mut<float, 2> array,
    float value
) {
    for (auto i = region.begin(0); i < region.end(0); i++) {
        for (auto j = region.begin(1); j < region.end(1); j++) {
            array[i][j] = value;
        }
    }
}

void matrix_multiply(
    kmm::CudaDevice& device,
    kmm::WorkChunk chunk,
    int n,
    int m,
    int k,
    kmm::cuda_subview_mut<float, 2> C,
    kmm::cuda_subview<float, 2> A,
    kmm::cuda_subview<float, 2> B
) {
    float alpha = 1.0;
    float beta = 0.0;

    const float* A_ptr = A.data_at({chunk.begin.x, chunk.begin.z});
    const float* B_ptr = B.data_at({chunk.begin.z, chunk.begin.y});
    float* C_ptr = C.data_at({chunk.begin.x, chunk.begin.y});

    KMM_CUDA_CHECK(cublasGemmEx(
        device.cublas(),
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        chunk.size(0),
        chunk.size(1),
        chunk.size(2),
        &alpha,
        A_ptr,
        CUDA_R_32F,
        A.stride(0),
        B_ptr,
        CUDA_R_32F,
        B.stride(0),
        &beta,
        C_ptr,
        CUDA_R_32F,
        C.stride(0),
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

int main() {
    using namespace kmm::placeholders;

    auto rt = kmm::make_runtime();
    int n = 500;
    int m = 500;
    int k = 500;
    int chunk_size = 100;

    auto A = kmm::Array<float, 2>{{n, k}};
    auto B = kmm::Array<float, 2>{{k, m}};
    auto C = kmm::Array<float, 2>{{n, m}};

    rt.parallel_submit(
        {n, k},
        {chunk_size, chunk_size},
        kmm::Host(fill_array),
        write(A, slice(_x, _y)),
        1.0F
    );

    rt.parallel_submit(
        {k, m},
        {chunk_size, chunk_size},
        kmm::Host(fill_array),
        write(B, slice(_x, _y)),
        1.0F
    );

    rt.parallel_submit(
        {n, m, k},
        {chunk_size, chunk_size, chunk_size},
        kmm::Cuda(matrix_multiply),
        n, m, k,
        reduce(C, kmm::ReductionOp::Sum, slice(_x, _y)),
        read(A, slice(_x, _z)),
        read(B, slice(_z, _y))
    );
}