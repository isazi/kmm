#include "vector_add.h"
#include "vector_add.cuh"


int main(void) {
    unsigned int threads_per_block = 256;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    int n = SIZE;

    // Allocate memory on the host
    auto A = new float[n];
    auto B = new float[n];
    auto C = new float[n];

    // Initialize array A and B on the host
    initialize(A, B);

    // Allocate memory on the device
    void *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, n * sizeof(float ));
    cudaMalloc(&B_d, n * sizeof(float ));
    cudaMalloc(&C_d, n * sizeof(float ));

    // Copy A and B from host to device
    cudaMemcpy(A_d, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n * sizeof(float), cudaMemcpyHostToDevice);

    // Execute the function on the device
    vector_add<<<n_blocks, threads_per_block>>>(reinterpret_cast<float *>(A_d), reinterpret_cast<float *>(B_d), reinterpret_cast<float *>(C_d), n);

    // Copy C from device to host
    cudaMemcpy(C, C_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result on the host.
    cudaDeviceSynchronize();
    verify(C);

    std::cout << "done\n";

    return 0;
}
