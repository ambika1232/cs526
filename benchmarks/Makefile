CUDA_HOME ?= /usr/local/cuda
CC        := gcc
CFLAGS    := -I$(CUDA_HOME)/include
LDFLAGS   := -L$(CUDA_HOME)/lib64 -lcuda

TARGETS := bench_prefetch bench_gemm bench_2mm bench_3mm bench_atax bench_bicg

all: $(TARGETS)

%: %.c
	$(CC) $< $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f $(TARGETS)
