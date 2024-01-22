# Copyright (C) 2024, Rupert Nash, The University of Edinburgh.
#
# All rights reserved.
#
# This file is provided to you to complete an assessment and for
# subsequent private study. It may not be shared and, in particular,
# may not be posted on the internet. Sharing this or any modified
# version may constitute academic misconduct under the University's
# regulations.

# Disable all of make's built-in rules (similar to Fortran's implicit none)
MAKEFLAGS += --no-builtin-rules --no-builtin-variables

# configuration
FC = nvfortran
CC = nvc
FCFLAGS = -O3
CFLAGS = -O3
LD = $(FC)
RM = rm -f

# list of all source files
COMMON_SRCS = hist_cpu.f95 io.f95 ../c/io.c

PROGS = test bench

COMMON_OBJS = $(addsuffix .o, $(notdir $(COMMON_SRCS)))

.PHONY: all clean
all: $(PROGS)

test: $(COMMON_OBJS) test.F95.o
	$(LD) -o $@ $^

bench: $(COMMON_OBJS) hist_gpu.cuf.o bench.cuf.o
	$(LD) -cuda -o $@ $^

%.f95.o: %.f95
	$(FC) $(FCFLAGS) -c -o $@ $<
%.F95.o: %.F95
	$(FC) $(FCFLAGS) -c -o $@ $<
%.cuf.o: %.cuf
	$(FC) $(FCFLAGS) -cuda -c -o $@ $<
%.c.o : ../c/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# define dependencies between object files (cos modules)
test.F95.o : hist_cpu.f95.o


clean:
	$(RM) $(COMMON_OBJS) test.F95.o bench.cuf.o hist_gpu.cuf.o $(wildcard *.mod) $(PROGS)
