/**
 * Exercise 1: CUDA Device Query
 *
 * Query the GPU device properties using the CUDA runtime API.
 * This is the first thing you should do when working with a new GPU —
 * understand what hardware you're running on.
 */

#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

void queryDevice() {
    int deviceCount = 0;

    // TODO 1: Get the number of CUDA-capable devices and store in deviceCount.
    cudaGetDeviceCount(&deviceCount);
    

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;

        // TODO 2: Get properties for device i into `prop`.
        cudaGetDeviceProperties(&prop, i);
        

        printf("===== Device %d =====\n", i);

        // TODO 3: Print the following properties:
        printf("Device name: %s \n", prop.name);
        printf("Compute capability: (%d.%d) \n", prop.major, prop.minor);
        printf("Total global memory: %lu MB \n", prop.totalGlobalMem / (1024 * 1024));
        printf("Number of SMs: %d \n", prop.multiProcessorCount);
        printf("Max threads per block: %d \n", prop.maxThreadsPerBlock);
        printf("Max block dimensions: (%d, %d, %d) \n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d) \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Warp size: %d \n", prop.warpSize);
        //   - Device name                       (prop.name)
        //   - Compute capability (major.minor)  (prop.major, prop.minor)
        //   - Total global memory in MB         (prop.totalGlobalMem)
        //   - Number of SMs                     (prop.multiProcessorCount)
        //   - Max threads per block             (prop.maxThreadsPerBlock)
        //   - Max block dimensions (x, y, z)    (prop.maxThreadsDim[0..2])
        //   - Max grid dimensions  (x, y, z)    (prop.maxGridSize[0..2])
        //   - Warp size                         (prop.warpSize)
    }
}

int main() {
    queryDevice();
    return 0;
}
