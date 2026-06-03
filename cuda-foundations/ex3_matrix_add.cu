/**
 * Exercise 3: Matrix Addition — 2D Grid and Block Indexing
 *
 * Implement a CUDA kernel that performs element-wise matrix addition:
 *   C[row][col] = A[row][col] + B[row][col]
 *
 * Matrices are stored in row-major order in a 1D array:
 *   element (row, col) is at index [row * cols + col]
 */

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>

#define ROWS 512
#define COLS 1024

// TODO 1: Write the matrix addition kernel.
//   Use 2D thread indexing to compute (row, col):
//     row = blockIdx.y * blockDim.y + threadIdx.y
//     col = blockIdx.x * blockDim.x + threadIdx.x
//   Linearize to a flat index: idx = row * cols + col
//   Guard: row < rows && col < cols
__global__ void matAdd(const float *A, const float *B, float *C, int rows, int cols) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

bool verify(const float *A, const float *B, const float *C, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(C[i] - (A[i] + B[i])) > 1e-5f) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, A[i] + B[i], C[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int totalElements = ROWS * COLS;
    size_t bytes = totalElements * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < totalElements; i++) {
        h_A[i] = (float)(rand() % 100);
        h_B[i] = (float)(rand() % 100);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // TODO 2: Define 2D block and grid dimensions using dim3.
    //   Use a block size of (16, 16).
    //   Compute the grid size so that every matrix element is covered.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((COLS + 15) / 16, (ROWS + 15) / 16);

    // TODO 3: Launch the matAdd kernel.
    matAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    if (verify(h_A, h_B, h_C, ROWS, COLS)) {
        printf("SUCCESS\n");
    } else {
        printf("FAILURE\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
