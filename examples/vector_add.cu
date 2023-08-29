#include <iostream>

#include "kmm.hpp"

#define SIZE 65536

__global__ void vector_add(float* A, float* B, float* C, unsigned int size) {
    unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (item < size) {
        C[item] = A[item] + B[item];
    }
}

int main(void) {
    unsigned int threads_per_block = 1024;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    unsigned int A_d, B_d, C_d;
    float* A_h;
    float* B_h;
    float* C_h;
    std::size_t n = SIZE * sizeof(float);
    auto manager = kmm::MemoryManager();

    A_h = reinterpret_cast<float*>(malloc(n));
    B_h = reinterpret_cast<float*>(malloc(n));
    C_h = reinterpret_cast<float*>(malloc(n));
    for (unsigned int item = 0; item < SIZE; item++) {
        A_h[item] = 1.0;
        B_h[item] = 2.0;
    }
    A_d = manager.allocate(n, reinterpret_cast<void*>(A_h));
    B_d = manager.allocate(n, reinterpret_cast<void*>(B_h));
    C_d = manager.allocate(n);
    vector_add<<<threads_per_block, n_blocks, 0, manager.getStream()>>>(
        reinterpret_cast<float*>(manager.getPointer(A_d)),
        reinterpret_cast<float*>(manager.getPointer(B_d)),
        reinterpret_cast<float*>(manager.getPointer(C_d)),
        SIZE);
    manager.release(A_d);
    manager.release(B_d);
    manager.release(C_d, n, reinterpret_cast<void*>(C_h));
    for (unsigned int item = 0; item < SIZE; item++) {
        if ((C_h[item] - 3.0) > 1.0e-9) {
            std::cout << "ERROR" << std::endl;
            return -1;
        }
    }
    std::cout << "SUCCESS" << std::endl;
    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}
