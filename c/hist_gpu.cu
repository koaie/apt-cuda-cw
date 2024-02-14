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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    atomicAdd(&ans[ub - 1], 1);
  }
}

__global__ void hist_zero_array(int const nbins, int *ans)
{
  /* Zero result array */
  for (int i = 0; i < nbins; ++i)
  {
    ans[i] = 0;
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
  // hist_kernel_serial<<<1,1>>>(nbins, bin_edges, ndata, data, counts);

  int nthreads = 100;
  int nblocks = NUM_BLOCKS(ndata,nthreads);
  
  int size = nthreads * nblocks; // total threads
  int local_ndata = ndata / size; // Iterations per thread
  // int diff = ndata - local_ndata * size;

  printf("ndata %d, nblocks %d, local_ndata %d, diff %d\n", ndata, nblocks, local_ndata, diff);
  hist_zero_array<<<1, 1>>>(nbins, counts);
  hist_kernel_parallel<<<nblocks, nthreads>>>(nbins, bin_edges, ndata, data, counts);
  /* REMEBMER TO ENSURE YOUR KERNEL ARE FINISHED! */
  CUDA_CHECK(cudaDeviceSynchronize());
}
