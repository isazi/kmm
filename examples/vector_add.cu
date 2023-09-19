#include <iostream>

#include "kmm.hpp"

#define SIZE 65536

// __global__ void vector_add(float* A, float* B, float* C, unsigned int size) {
//     unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if (item < size) {
//         C[item] = A[item] + B[item];
//     }
// }

void initialize(void* A, void* B, unsigned int size) {
    for (unsigned int item = 0; item < size; item++) {
        reinterpret_cast<float*>(A)[item] = 1.0;
        reinterpret_cast<float*>(B)[item] = 2.0;
    }
}

void execute() {}

void verify(float* C, unsigned int size) {
    for (unsigned int item = 0; item < size; item++) {
        if ((C[item] - 3.0) > 1.0e-9) {
            std::cout << "ERROR" << std::endl;
            break;
        }
    }
    std::cout << "SUCCESS" << std::endl;
}

int main(void) {
    unsigned int threads_per_block = 1024;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    std::size_t n = SIZE * sizeof(float);

    // Create memory manager
    auto manager = kmm::Manager();
    // Create devices
    auto cpu = kmm::CPU();
    auto gpu = kmm::CUDA();
    // Allocate memory on the host
    auto A_h = manager.create<kmm::FP_Single>(n);
    auto B_h = manager.create<kmm::FP_Single>(n);
    auto C_h = manager.create<kmm::FP_Single>(n);
    // TODO: run initialization
    // Allocate memory on the GPU
    auto A_d = manager.create<kmm::FP_Single>(n);
    auto B_d = manager.create<kmm::FP_Single>(n);
    auto C_d = manager.create<kmm::FP_Single>(n);
    // Copy data to the GPU
    manager.copy_to(gpu, A_d, A_h);
    manager.copy_to(gpu, B_d, B_h);
    // TODO: run kernel
    // Free GPU memory and copy data back
    manager.release(A_d);
    manager.release(B_d);
    manager.release(gpu, C_d, C_h);
    // TODO: run verify
    // Free host memory
    manager.release(A_h);
    manager.release(B_h);
    manager.release(C_h);
    return 0;
}
