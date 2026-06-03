/**
 * Exercise 2: Vector Addition — Your First Kernel
 *
 * Implement a CUDA kernel that performs element-wise addition of two
 * vectors: C[i] = A[i] + B[i].
 */

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>

#define N 1024

// TODO 1: Write the vector addition kernel.
//   Each thread computes one element: C[i] = A[i] + B[i].
//   Compute the global thread index from blockIdx, blockDim, threadIdx.
//   Guard against out-of-bounds access (i >= n).
__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            C[i] = A[i] + B[i];
        }
}

bool verify(const float *A, const float *B, const float *C, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(C[i] - (A[i] + B[i])) > 1e-5f) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, A[i] + B[i], C[i]);
            return false;
        }
    }
    return true;
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    float *d_A, *d_B, *d_C;

    // TODO 2: Allocate device memory for d_A, d_B, d_C.
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // TODO 3: Copy h_A and h_B from host to device.
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // TODO 4: Launch the vecAdd kernel with N threads per block and 1 block.
    vecAdd<<<1, N>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    // TODO 5: Copy the result (d_C) back to host (h_C).
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    if (verify(h_A, h_B, h_C, N)) {
        printf("SUCCESS\n");
    } else {
        printf("FAILURE\n");
    }

    // TODO 6: Free device and host memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
