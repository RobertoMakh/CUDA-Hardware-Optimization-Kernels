# 2D Heat Diffusion & Stencil Kernel Optimization

## Overview
A native C++/CUDA project simulating 2D heat diffusion across a grid. The simulation applies a large weighted stencil at each timestep to compute temperature changes influenced by a moving heat source. The core focus is hardware-level optimization using native CUDA memory types, streams, and kernel tuning to achieve maximum speedup over a baseline implementation without relying on external libraries or autotuners.

## Implemented Kernels
* **Reference Kernel:** A baseline implementation (`heat_cuda_reference.cu`) applying the heat equation and stencil weighting using standard global memory access patterns.
* **Optimized Stencil Kernel:** A highly tuned implementation (`heat_cuda_optimized.cu`) designed to maximize throughput and minimize execution time. It leverages native CUDA optimizations to efficiently handle the memory-bound nature of calculating the large weighted stencil across the simulation grid.

## Performance Profiling & Validation
* **Benchmarking:** A custom `benchmark` utility rigorously validates the numerical correctness of the optimized kernel against the reference across multiple simulations.
* **Hardware Timing:** Execution time is profiled to measure the mean, minimum, maximum, and standard deviation, demonstrating raw speedup on an NVIDIA GeForce RTX 2080 Ti.
* **Visualization:** A `plot` utility is included to generate visual BMP frame sequences of the grid, allowing for physical observation of the heat spread and moving source over time.