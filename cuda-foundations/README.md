\# CUDA Foundations \& Hardware Optimization



\## Overview

A collection of native C++/CUDA kernels demonstrating core GPU programming mechanics, memory hierarchy management, and hardware-level performance optimization. Developed to explore execution configurations, memory dataflow, and the performance impact of coalesced access patterns.



\## Implemented Kernels

\* \*\*Device Querying:\*\* Interrogating GPU hardware limits, multiprocessor count, and warp sizes.

\* \*\*Execution Configuration:\*\* Mapping 1D and 2D grids/blocks to data for vector and matrix arithmetic.

\* \*\*Shared Memory Utilization:\*\* Reversing arrays using both global memory and block-level shared memory to bypass global memory latency.

\* \*\*Memory Coalescing \& Tiling:\*\* Optimizing a matrix transpose kernel by loading tiles into `\_\_shared\_\_` memory, ensuring both read and write operations to global memory are fully coalesced. 



\## Performance Profiling

The matrix transpose implementation measures kernel execution time using CUDA events, demonstrating the raw hardware speedup achieved by eliminating uncoalesced memory strides.

