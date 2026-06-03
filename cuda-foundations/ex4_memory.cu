/**
 * Exercise 4: Memory Operations — Global and Shared Memory
 *
 * Implement array reversal two ways:
 *   Part A — Using global memory only.
 *   Part B — Using shared memory for block-level reversal.
 */

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

/* ------------------------------------------------------------------ */
/*  Part A: Global Memory Reversal                                     */
/* ------------------------------------------------------------------ */

// TODO 1: Write a kernel that reverses the array using global memory.
__global__ void reverseGlobal(const float *in, float *out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        out[n - 1 - i] = in[i];
    }
}

/* ------------------------------------------------------------------ */
/*  Part B: Shared Memory Block-Level Reversal                         */
/* ------------------------------------------------------------------ */

// TODO 2: Write a kernel that reverses elements *within each block*
//   using shared memory.
//   Steps:
//     1. Declare a __shared__ float array of size BLOCK_SIZE.
//     2. Each thread loads in[globalIdx] into shared[threadIdx.x].
//     3. __syncthreads()
//     4. Each thread reads shared[BLOCK_SIZE - 1 - threadIdx.x] and
//        writes it to the corresponding global output position.
__global__ void reverseShared(const float *in, float *out, int n) {
    __shared__ float s_data[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x; 
    
    if (i < n) {
        s_data[tid] = in[i];
    }
    
    __syncthreads();

    if (i < n) {
        out[i] = s_data[BLOCK_SIZE - 1 - tid];
    }
}

/* ------------------------------------------------------------------ */
/*  Verification                                                       */
/* ------------------------------------------------------------------ */

bool verifyReverse(const float *original, const float *reversed, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(reversed[i] - original[n - 1 - i]) > 1e-5f) {
            printf("  Mismatch at %d: expected %f, got %f\n", i,
                   original[n - 1 - i], reversed[i]);
            return false;
        }
    }
    return true;
}

bool verifyBlockReverse(const float *original, const float *reversed, int n, int blockSize) {
    for (int b = 0; b < n / blockSize; b++) {
        for (int t = 0; t < blockSize; t++) {
            int inIdx  = b * blockSize + t;
            int outIdx = b * blockSize + (blockSize - 1 - t);
            if (fabsf(reversed[outIdx] - original[inIdx]) > 1e-5f) {
                printf("  Block %d mismatch at local %d: expected %f, got %f\n",
                       b, t, original[inIdx], reversed[outIdx]);
                return false;
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main() {
    size_t bytes = N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* ====== Part A: Global Memory ====== */
    {
        printf("Part A: Global Memory Reversal\n");
        float *h_out = (float *)malloc(bytes);
        float *d_in, *d_out;

        // TODO 3: Allocate d_in and d_out, copy h_in to device,
        //         launch reverseGlobal, copy result back to h_out.
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        
        reverseGlobal<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, N);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        if (verifyReverse(h_in, h_out, N))
            printf("  SUCCESS\n");
        else
            printf("  FAILURE\n");

        // TODO 4: Free d_in, d_out, h_out.
        cudaFree(d_in);
        cudaFree(d_out);
        free(h_out);
    }

    /* ====== Part B: Shared Memory ====== */
    {
        printf("Part B: Shared Memory Block-Level Reversal\n");
        float *h_out = (float *)malloc(bytes);
        float *d_in, *d_out;

        // TODO 5: Allocate, copy, launch reverseShared, copy back.
        // What should the block dimension be for your kernel implementation?
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
        
        reverseShared<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, N);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        if (verifyBlockReverse(h_in, h_out, N, BLOCK_SIZE))
            printf("  SUCCESS\n");
        else
            printf("  FAILURE\n");

        // TODO 6: Free d_in, d_out, h_out.
        cudaFree(d_in); 
        cudaFree(d_out);
        free(h_out);
    }

    free(h_in);
    return 0;
}
