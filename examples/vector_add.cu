#include <iostream>

#include "kmm.hpp"

#define SIZE 65536

// __global__ void vector_add(float* A, float* B, float* C, unsigned int size) {
//     unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if (item < size) {
//         C[item] = A[item] + B[item];
//     }
// }

void initialize(float* A, float* B, unsigned int size) {
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
    // Request 3 memory areas of a certain size
    auto A = manager.create<float>(n);
    auto B = manager.create<float>(n);
    auto C = manager.create<float>(n);
    // TODO: run initialization
    // Copy data to the GPU
    manager.move_to(gpu, A);
    manager.move_to(gpu, B);
    // TODO: run kernel
    // Free GPU memory and copy data back
    manager.release(A);
    manager.release(B);
    float* C_h = reinterpret_cast<float*>(malloc(n));
    manager.copy_release(C, C_h);
    // TODO: run verify
    return 0;
}
