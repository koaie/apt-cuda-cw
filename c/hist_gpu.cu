/* -*- mode: C; -*- */
/*
 * Copyright (C) 2024, Rupert Nash, The University of Edinburgh.
 *
 * All rights reserved.
 *
 * This file is provided to you to complete an assessment and for
 * subsequent private study. It may not be shared and, in particular,
 * may not be posted on the internet. Sharing this or any modified
 * version may constitute academic misconduct under the University's
 * regulations.
 */

/* YOU MAY MODIFY THIS FILE FREELY (WITHIN THE CONSTRAINTS DESCRIBED
 * IN THE README). */

#include "hist_gpu.h"

#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 256
#define MAX_BLOCKS 2560 // 32 max blocks per SM * 80 SMs

#define NUM_BLOCKS(ARRAY_SIZE, THREADS_PER_BLOCK) ((ARRAY_SIZE)-1) / THREADS_PER_BLOCK + 1

/*
 * Return index of first element in range greater than value, or len
 * if not found.
 */
__device__ int upper_bound(int len, double const *data, double value)
{
  int begin = 0;
  while (len > 0)
  {
    int half = len / 2;
    int mid = begin + half;
    if (value < data[mid])
    {
      len = half;
    }
    else
    {
      begin = mid + 1;
      len -= half + 1;
    }
  }
  return begin;
}

__global__ void hist_kernel_serial(int const nbins, double const *bin_edges, int const ndata, double const *data, int *ans)
{
  /* Zero result array */
  for (int i = 0; i < nbins; ++i)
  {
    ans[i] = 0;
  }

  for (int i = 0; i < ndata; ++i)
  {
    int ub = upper_bound(nbins + 1, bin_edges, data[i]);
    if (ub == 0)
    {
      /* value below all bins */
    }
    else if (ub == nbins + 1)
    {
      /* value above all bins */
    }
    else
    {
      /* in a bin! */
      ans[ub - 1] += 1;
    }
  }
}

__global__ void hist_kernel_parallel(int const nbins, double const *bin_edges, int const ndata, double const *data, int *ans)
{
  // shared memory
  extern __shared__ int local_ans[];

  // Our thread index in the grid
  // i = block id * block size + thread id
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Zero result array */
  // Increase by one as long as nthread+num_blocks is smaller than nbins (allowing us to deal with bins bigger than nblocks or nthreads)
  for (int j = threadIdx.x; j < nbins; j += blockDim.x)
  {
    local_ans[j] = 0;
  }
  __syncthreads();

  /*
  Skip to the next data index by the size of the x dimension 

  index, i = block id * block size + thread id
  grid size = block size * number of blocks
  +1 = to increment our index
  
  steps:
  | i | i + block size * nblocks + 1 | ... until i < ndata
  */
  int grid_size = blockDim.x * gridDim.x;

  for(int j = i; j < ndata; j += grid_size)
  {
    int ub = upper_bound(nbins + 1, bin_edges, data[j]);
    // simplify boundries
    if (ub > 0 && ub < nbins + 1)
    {
      /* in a shared bin! */
      atomicAdd(&local_ans[ub - 1], 1);
    }
  }
  __syncthreads();


  // Copy from shared (onchip) to global, i.e. reduction
  for (int j = threadIdx.x; j < nbins; j += blockDim.x)
  {
    atomicAdd(&ans[j], local_ans[j]);
  }
}

/*
 * Compute the counts of data in the bins.
 *
 * ASSUMES THAT DATA ARRAYS ARE ON GPU
 *
 * int nbins - the number of bins
 *
 * double bin_edges[nbins+1] - array of length nbins + 1 holding the
 * bin edges must be sorted and no zero-size bins, i.e.:
 *
 *   bin_edges[i] < bin_edges[i+1]
 *
 * int ndata - number of input data points
 *
 * double data[ndata] - data to be histogrammed, size == ndata
 *
 * int counts[nbins] - array of int, size nbins. Holds the result and
 * must be allocated by caller. Element i holds the count of elements
 * in data >= bin_edges[i] and < bin_edges[i+1]
 */
void compute_histogram_gpu(int const nbins, double const *bin_edges, int const ndata, double const *data, int *counts)
{
  /* Skip checks as the data lives in GPU memory - for this exercise
   * we will be using the same inputs as for the CPU version which has
   * checked already.
   */

  size_t bin_size = nbins * sizeof(int); // Size of bins shared array

  int nblocks = NUM_BLOCKS(ndata, NUM_THREADS); // Dynmically allocate number of blocks

  // if nblocks is higher than the maxmimum possible block count, then use the maximum.
  if(nblocks > MAX_BLOCKS)
  {
    nblocks = MAX_BLOCKS;
  }

  // Useful debugging information
  // printf("nbins %d, ndata %d, nthreads %d, nblocks %d\n", nbins, ndata, NUM_THREADS, nblocks);

  // Zero array
  cudaMemset(counts,0,sizeof(int) * nbins);

  // Launch kerne with nblocks, nthreads, and allocate bin_size for the shared array
  hist_kernel_parallel<<<nblocks, NUM_THREADS, bin_size>>>(nbins, bin_edges, ndata, data, counts);

  // Serial, kept for benchmarking and speedup gathering
  // hist_kernel_serial<<<1,1>>>(nbins, bin_edges, ndata, data, counts);

  /* REMEBMER TO ENSURE YOUR KERNEL ARE FINISHED! */
  CUDA_CHECK(cudaDeviceSynchronize());
}
