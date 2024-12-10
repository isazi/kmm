#include "kmm/kmm.hpp"

void fill_array(
    kmm::NDRange region,
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
    kmm::DeviceContext& device,
    kmm::NDRange region,
    int n,
    int m,
    int k,
    kmm::gpu_subview_mut<float, 2> C,
    kmm::gpu_subview<float, 2> A,
    kmm::gpu_subview<float, 2> B
) {
    using kmm::checked_cast;

    float alpha = 1.0;
    float beta = 0.0;

    const float* A_ptr = A.data_at({region.begin.x, region.begin.z});
    const float* B_ptr = B.data_at({region.begin.z, region.begin.y});
    float* C_ptr = C.data_at({region.begin.x, region.begin.y});

    KMM_GPU_CHECK(cublasGemmEx(
        device.blas(),
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        checked_cast<int>(region.sizes().x),
        checked_cast<int>(region.sizes().y),
        checked_cast<int>(region.sizes().z),
        &alpha,
        A_ptr,
        CUDA_R_32F,
        checked_cast<int>(A.stride()),
        B_ptr,
        CUDA_R_32F,
        checked_cast<int>(B.stride()),
        &beta,
        C_ptr,
        CUDA_R_32F,
        checked_cast<int>(C.stride()),
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int n = 5000;
    int m = 5000;
    int k = 5000;
    int chunk_size = n / 5;

    auto A = kmm::Array<float, 2>{{n, k}};
    auto B = kmm::Array<float, 2>{{k, m}};
    auto C = kmm::Array<float, 2>{{n, m}};

    rt.parallel_submit(
        {n, k},
        {chunk_size, chunk_size},
        kmm::Host(fill_array),
        write(A, access(_x, _y)),
        1.0F
    );

    rt.parallel_submit(
        {k, m},
        {chunk_size, chunk_size},
        kmm::Host(fill_array),
        write(B, access(_x, _y)),
        1.0F
    );

    for (size_t repeat = 0; repeat < 10; repeat++) {
        C.reset();

        rt.parallel_submit(
            {n, m, k},
            {chunk_size, chunk_size, chunk_size},
            kmm::GPU(matrix_multiply),
            n,
            m,
            k,
            reduce(C, kmm::ReductionOp::Sum, access(_x, _y)),
            read(A, access(_x, _z)),
            read(B, access(_z, _y))
        );

        rt.synchronize();
    }

    return EXIT_SUCCESS;
}
