# FPGA Quantized Neural Network Accelerator

## Overview
A Python and Jupyter-based project focusing on Quantization-Aware Training (QAT) and hardware synthesis of neural networks for FPGA deployment. Utilizing Brevitas for training and Xilinx FINN, the project compiles ONNX graphs directly into custom streaming dataflow hardware architectures on a PYNQ-Z2 (Zynq-7020) board. The core focus is balancing model accuracy with hardware constraints by tuning spatial folding (SIMD/PE) and analyzing resource utilization.

## Implemented Models
* **W16A8 FC Network:** A baseline quantized fully connected network using 16-bit weights and 8-bit activations, heavily utilizing BRAM for weight storage.
* **W2A1 (Binary) Network:** An aggressively quantized model utilizing 2-bit ternary weights and 1-bit bipolar activations. It replaces standard DSP multipliers with highly efficient XNOR and popcount logic mapped directly to LUTs.
* **Custom W4A8 CNN:** A custom-designed Convolutional Neural Network optimized to balance accuracy, throughput, and latency. It leverages data augmentation and tuned folding configurations to achieve over 56% accuracy on CIFAR-10 while fitting within the strict physical resource constraints of the FPGA.

## Performance Profiling & Validation
* **Hardware Synthesis:** Models are pushed through FINN's 19-step compilation pipeline, converting ONNX representations into Verilog via Vitis HLS, and finally generating bitstreams using Vivado out-of-context synthesis.
* **Resource Utilization:** Detailed profiling of LUT, Flip-Flop (FF), BRAM_36K, and DSP block consumption across different quantization bit-widths and parallelism strategies. 
* **On-Board Inference:** Execution of the generated `.bit` files on a physical PYNQ-Z2 board, actively measuring end-to-end pipeline latency (ms/image), frames-per-second (FPS) throughput, and validating empirical hardware accuracy against the software QAT baseline.
