#include <iostream>
#include <cmath>

#define SIZE 65536000

void initialize(float* A, float* B) {
#pragma omp parallel for
    for (unsigned int item = 0; item < SIZE; item++) {
        A[item] = 1.0;
        B[item] = 2.0;
    }

    std::cout << "initialize\n";
}

void verify(const float* C) {
#pragma omp parallel for
    for (unsigned int item = 0; item < SIZE; item++) {
        if (fabsf(C[item] - 3.0f) > 1.0e-9) {
            std::cout << "ERROR" << std::endl;
            break;
        }
    }

    std::cout << "SUCCESS" << std::endl;
}