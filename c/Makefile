# Copyright (C) 2024, Rupert Nash, The University of Edinburgh.
#
# All rights reserved.
#
# This file is provided to you to complete an assessment and for
# subsequent private study. It may not be shared and, in particular,
# may not be posted on the internet. Sharing this or any modified
# version may constitute academic misconduct under the University's
# regulations.

CFLAGS ?= -O3
CUDAFLAGS = -arch=sm_70
LDFLAGS ?= -lm

COMMON = hist_cpu.o timer.o io.o
PROGS = test bench

OBJECTS = $(COMMON) $(PROGS:=.o)

all : $(PROGS)

test : test.o $(COMMON)

bench : bench.cu hist_gpu.cu $(COMMON)
	nvcc $(CUDAFLAGS) $(CFLAGS) $(NVCCFLAGS) $^ -o $@

.PHONY : clean
clean :
	rm -f $(PROGS) $(OBJECTS)
