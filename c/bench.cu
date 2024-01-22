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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"
#include "io.h"
#include "hist_cpu.h"
#include "hist_gpu.h"

static char const* usage =
  "Compute a histogram, measuring performance.\n"
  "Usage:\n"
  "    bench [-n NRUNS] bins_file data_file hist_file\n"
  "\n"
  "optional arg -n sets number of runs for stats\n"
  "bins_file contains the bin edges\n"
  "data_file contains the data to be histogrammed\n"
  "\n"
  "Output data to `hist_file`\n"
  "Timing data to standard output\n";

typedef struct run_stats {
  double min;
  double max;
  double mean;
  double std;
  int N;
} run_stats;

void calc_stats(int const N, double const* data, run_stats* ans) {
  /* Compute stats */
  ans->min = INFINITY;
  ans->max = -INFINITY;
  double tsum = 0.0, tsumsq = 0.0;
  for (int i = 0; i < N; ++i) {
    double const t = data[i];
    tsum += t;
    tsumsq += t * t;
    ans->min = (t < ans->min) ? t : ans->min;
    ans->max = (t > ans->max) ? t : ans->max;
  }
  ans->mean = tsum / N;
  double tvar = (tsumsq - tsum*tsum / N) / (N - 1);
  ans->std = sqrt(tvar);
  ans->N = N;
}

void print_stats(FILE* stream, char const* where, run_stats* ans) {
  printf("\nSummary for %s (all in s):\nmin = %e, max = %e, mean = %e, std = %e\n",
	 where,
	 ans->min, ans->max, ans->mean, ans->std);
}

int main(int argc, char* argv[]) {
  
  int nruns = 10;
  char const* edges_fn = NULL;
  char const* data_fn = NULL;
  char const* hist_fn = NULL;

  {
    /* Parse command line arguments */
    if (argc < 4) {
      fprintf(stderr, "Too few arguments\n");
      fprintf(stderr, usage);
      return 1;
    }

    int iarg = 1;
    if (strcmp(argv[iarg], "-n") == 0) {
      char* end;
      nruns = strtol(argv[iarg + 1], &end, 0);
      if (end == argv[iarg + 1]) {
	fprintf(stderr, "Problem with argument to option '-n'\n");
	fprintf(stderr, usage);
	return 1;
      }
      iarg += 2;
    }
    if (iarg < argc) {
      edges_fn = argv[iarg++];
    } else {
      fprintf(stderr, "No bins filename supplied\n");
      fprintf(stderr, usage);
      return 1;
    }

    if (iarg < argc) {
      data_fn = argv[iarg++];
    } else {
      fprintf(stderr, "No data filename supplied\n");
      fprintf(stderr, usage);
      return 1;
    }

    if (iarg < argc) {
      hist_fn = argv[iarg++];
    } else {
      fprintf(stderr, "No hist filename supplied\n");
      fprintf(stderr, usage);
      return 1;
    }

    if (iarg != argc) {
      fprintf(stderr, "Too many arguments supplied\n");
      fprintf(stderr, usage);
      return 1;
    }
  }

  int nedges;
  double* edges;
  if (read_column_data(edges_fn, &nedges, &edges)) {
    fprintf(stderr, "Error reading data from '%s'\n", edges_fn);
    return 1;
  } else {
    printf("Read %d bin edges from '%s'\n", nedges, edges_fn);
  }
  int const nbins = nedges - 1;
  check_bin_self_consistency(nbins, edges);

  int ndata;
  double* data;
  if (read_column_data(data_fn, &ndata, &data)) {
    fprintf(stderr, "Error reading data from '%s'\n", data_fn);
    return 1;
  } else {
    printf("Read %d data points from '%s'\n", ndata, data_fn);
  }

  /* We'll do the runs and print stats at the end */
  double cpu_runtime_s[nruns];
  int* cpu_counts = (int*)malloc(nbins * sizeof(int));

  printf("Beginning %d CPU runs\n", nruns);

  for (int i = 0; i < nruns; ++i) {
    Timer t;
    timer_start(&t);
    compute_histogram_cpu(nbins, edges, ndata, data, cpu_counts);
    timer_stop(&t);
    cpu_runtime_s[i] = timer_check(&t);
    printf("Run %d: %e s\n", i, cpu_runtime_s[i]);
  }

  double gpu_runtime_s[nruns];

  int const counts_size = nbins*sizeof(int);
  int const edges_size = nedges*sizeof(double);
  int const data_size = ndata*sizeof(double);

  int* gpu_counts = (int*)malloc(counts_size);
  int* dev_gpu_counts;
  CUDA_CHECK(cudaMalloc(&dev_gpu_counts, counts_size));

  double* dev_edges;
  CUDA_CHECK(cudaMalloc(&dev_edges, edges_size));
  CUDA_CHECK(cudaMemcpy(dev_edges, edges, edges_size, cudaMemcpyHostToDevice));

  double* dev_data;
  CUDA_CHECK(cudaMalloc(&dev_data, data_size));
  CUDA_CHECK(cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice));

  printf("Beginning %d GPU runs\n", nruns);

  for (int i = 0; i < nruns; ++i) {
    Timer t;
    timer_start(&t);
    compute_histogram_gpu(nbins, dev_edges, ndata, dev_data, dev_gpu_counts);
    timer_stop(&t);
    gpu_runtime_s[i] = timer_check(&t);
    printf("Run %d: %e s\n", i, gpu_runtime_s[i]);
  }

  CUDA_CHECK(cudaMemcpy(gpu_counts, dev_gpu_counts, counts_size, cudaMemcpyDeviceToHost));

  /* Check results match CPU */
  int diff = 0;
  for (int i = 0; i < nbins; ++i) {
    diff += abs(cpu_counts[i] - gpu_counts[i]);
  }

  if (diff > 0) {
    fprintf(stderr, "L1 norm of difference in counts (%d) too large\n", diff);
    return 1;
  } else {
    printf("CPU and GPU histograms match\n");
  }

  {
    /* Write histogram */
    printf("Writing histogram to '%s'\n", hist_fn);
    if (write_hist(hist_fn, nbins, edges, cpu_counts)) {
      fprintf(stderr, "Error writing output\n");
      return 1;
    }
  }
  /* Compute stats */
  run_stats cpu_stats;
  calc_stats(nruns, cpu_runtime_s, &cpu_stats);
  print_stats(stdout, "CPU", &cpu_stats);

  run_stats gpu_stats;
  calc_stats(nruns, gpu_runtime_s, &gpu_stats);
  print_stats(stdout, "GPU", &gpu_stats);
  
  free(gpu_counts);
  CUDA_CHECK(cudaFree(dev_gpu_counts));
  CUDA_CHECK(cudaFree(dev_edges));
  CUDA_CHECK(cudaFree(dev_data));
  free(cpu_counts);
}
