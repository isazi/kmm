#include <iostream>

#include "kmm.hpp"

#define SIZE 65536

__global__ void vector_add(float* A, float* B, float* C, unsigned int size) {
    unsigned int item = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (item < size) {
        C[item] = A[item] + B[item];
    }
}

void initialize(void* A, void* B, unsigned int size) {
    for (unsigned int item = 0; item < size; item++) {
        reinterpret_cast<float*>(A)[item] = 1.0;
        reinterpret_cast<float*>(B)[item] = 2.0;
    }
}

int main(void) {
    unsigned int threads_per_block = 1024;
    unsigned int n_blocks = ceil((1.0 * SIZE) / threads_per_block);
    unsigned int A_h, B_h, C_h;
    unsigned int A_d, B_d, C_d;
    std::size_t n = SIZE * sizeof(float);
    auto manager = kmm::Manager();

    A_h = manager.create(kmm::DeviceType::CPU, n);
    B_h = manager.create(kmm::DeviceType::CPU, n);
    C_h = manager.create(kmm::DeviceType::CPU, n);
    manager.run(&initialize, kmm::DeviceType::CPU, 0, A_h, B_h, SIZE);
    A_d = manager.create(kmm::DeviceType::CUDA, 0, n);
    B_d = manager.create(kmm::DeviceType::CUDA, 0, n);
    C_d = manager.create(kmm::DeviceType::CUDA, 0, n);
    manager.copy_to(kmm::DeviceType::CUDA, A_d, n, A_h, 0);
    manager.copy_to(kmm::DeviceType::CUDA, B_d, n, B_h, 0);
    // TODO: reimplement execution
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
    manager.release(A_h);
    manager.release(B_h);
    manager.release(C_h);
    return 0;
}
