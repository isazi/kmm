
__global__ void vector_add(const float* A, const float* B, float* C, int size) {
    int item = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (item < size) {
        C[item] = A[item] + B[item];
    }
}