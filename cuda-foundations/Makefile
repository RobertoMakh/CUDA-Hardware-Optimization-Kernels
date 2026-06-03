NVCC = nvcc

TARGETS = ex1 ex2 ex3 ex4 ex5
FLAGS = -allow-unsupported-compiler
all: $(TARGETS)

ex1: ex1_device_query.cu
	$(NVCC) $(FLAGS) -o $@ $<

ex2: ex2_vector_add.cu
	$(NVCC) $(FLAGS) -o $@ $<

ex3: ex3_matrix_add.cu
	$(NVCC) $(FLAGS) -o $@ $<

ex4: ex4_memory.cu
	$(NVCC) $(FLAGS) -o $@ $<

ex5: ex5_error_and_timing.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
