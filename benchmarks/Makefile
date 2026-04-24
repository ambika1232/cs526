CUDA_HOME ?= /usr/local/cuda

bench_prefetch: bench_prefetch.c
	gcc $< \
	    -I$(CUDA_HOME)/include \
	    -L$(CUDA_HOME)/lib64 -lcuda \
	    -o $@

clean:
	rm -f bench_prefetch
