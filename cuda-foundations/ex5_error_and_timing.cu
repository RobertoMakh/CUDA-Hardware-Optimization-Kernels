/**
 * Exercise 5: Matrix Transpose — Shared Memory Speedup
 *
 *   Part A — Understand the CUDA_CHECK error-checking macro.
 *   Part B — Implement a naive matrix transpose kernel.
 *            (reads are coalesced, but writes are strided - slow)
 *   Part C — Implement a tiled transpose using shared memory.
 *            (both reads and writes from global memory are coalesced - fast)
 *   Part D — Time both kernels and observe the speedup.
 *
 * Background:
 *   Transposing means B[col][row] = A[row][col].
 *   In row-major memory: B[col * N + row] = A[row * N + col].
 *
 *   The naive kernel reads consecutive elements (coalesced) but writes
 *   with a stride of N (uncoalesced = many memory transactions).
 *
 *   The shared-memory kernel loads a TILE_DIM x TILE_DIM tile into
 *   shared memory, then writes it out transposed — making both the
 *   global read AND write coalesced.
 */

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>

#define N 4096
#define TILE_DIM 32
#define MEASURE_ITERATIONS 100

/* ------------------------------------------------------------------ */
/*  Part A: Understand the error-checking macro                        */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);  \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Part B: Naive transpose                                            */
/* ------------------------------------------------------------------ */

// TODO 1: Implement the naive transpose kernel.
//   Each thread transposes one element: out[col * n + row] = in[row * n + col]
//   Use 2D indexing:
//     row = blockIdx.y * blockDim.y + threadIdx.y
//     col = blockIdx.x * blockDim.x + threadIdx.x
//   Guard: row < n && col < n
__global__ void transposeNaive(const float *in, float *out, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}

/* ------------------------------------------------------------------ */
/*  Part C: Shared-memory tiled transpose                              */
/* ------------------------------------------------------------------ */

// TODO 2: Implement the tiled transpose kernel using shared memory.
//   Steps:
//     1. Declare __shared__ float tile[TILE_DIM][TILE_DIM];
//     2. Compute input (row, col) and load: tile[threadIdx.y][threadIdx.x] = in[row * n + col]
//     3. __syncthreads()
//     4. Compute the OUTPUT tile position (the tile is at a transposed block location, notice kernel launch dimensions):
//        int outCol = blockIdx.y * TILE_DIM + threadIdx.x;
//        int outRow = blockIdx.x * TILE_DIM + threadIdx.y;
//     5. Write: out[outRow * n + outCol] = tile[threadIdx.x][threadIdx.y]
//        (swapped indices in tile — that's the transpose)
__global__ void transposeShared(const float *in, float *out, int n) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = in[row * n + col];
    }

    __syncthreads();

    int outCol = blockIdx.y * TILE_DIM + threadIdx.x;
    int outRow = blockIdx.x * TILE_DIM + threadIdx.y;

    if (outRow < n && outCol < n) {
        out[outRow * n + outCol] = tile[threadIdx.x][threadIdx.y];
    }
}

/* ------------------------------------------------------------------ */
/*  Verification                                                       */
/* ------------------------------------------------------------------ */

bool verify(const float *original, const float *transposed, int n, const char *name) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float expected = original[row * n + col];
            float actual   = transposed[col * n + row];
            if (fabsf(actual - expected) > 1e-5f) {
                printf("[%s transpose] mismatch at (%d,%d): expected %f, got %f\n", name, row, col, expected, actual);
                return false;
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  Part D: Timing                                                     */
/* ------------------------------------------------------------------ */

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N * N; i++)
        h_in[i] = (float)(rand() % 1000) / 100.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(N / TILE_DIM, N / TILE_DIM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    


    // TODO 3: Measure the execution time of transposeNaive using events.
    //         Run it MEASURE_ITERATIONS times and store the total time in naiveMs.
    float naiveMs = 0;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < MEASURE_ITERATIONS; i++) {
        transposeNaive<<<gridDim, blockDim>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); 
    CUDA_CHECK(cudaEventElapsedTime(&naiveMs, start, stop));

    /* Verify naive result */
    float *h_out = (float *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool naiveCorrect = verify(h_in, h_out, N, "Naive");

    // TODO 4: Measure the execution time of transposeShared using events.
    //         Run it MEASURE_ITERATIONS times and store the total time in sharedMs.
    float sharedMs = 0;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0; i < MEASURE_ITERATIONS; i++) {
        transposeShared<<<gridDim, blockDim>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for GPU to finish
    CUDA_CHECK(cudaEventElapsedTime(&sharedMs, start, stop));

    /* Verify shared result */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool sharedCorrect = verify(h_in, h_out, N, "Shared");

    /* --- Report --- */
    printf("Matrix transpose %dx%d (%d iterations)\n", N, N, MEASURE_ITERATIONS);
    printf("  Naive:  %8.3f ms  [%s]\n", naiveMs, naiveCorrect ? "CORRECT" : "WRONG");
    printf("  Shared: %8.3f ms  [%s]\n", sharedMs, sharedCorrect ? "CORRECT" : "WRONG");
    printf("  Speedup: %.2fx\n", naiveMs / sharedMs);

    // TODO 5: Clean up — destroy events, free device and host memory.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
