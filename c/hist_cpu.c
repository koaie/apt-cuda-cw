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
#include "hist_cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer.h"

void check_bin_self_consistency(int const nbins, double const* bin_edges) {
  /* Sanity check */
  if (nbins <= 0) {
    fprintf(stderr, "Require at least one bin, have %d\n", nbins);
    exit(1);
  }

  for (int i = 0; i < nbins; ++i) {
    if (bin_edges[i] >= bin_edges[i+1]) {
      fprintf(stderr, "Bin edges not increasing at index %d", i);
      exit(1);
    }
  }
}

/*
 * Compute the counts of data in the bins.
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
void compute_histogram_cpu(int const nbins, double const* bin_edges, int const ndata, double const* data, int* counts) {
  /* Zero result array */
  memset(counts, 0, nbins * sizeof(int));

  /* Main calculation */
  for (int i = 0; i < ndata; ++i) {
    int ub = upper_bound(nbins + 1, bin_edges, data[i]);
    if (ub == 0) {
      /* value below all bins */
    } else if (ub == nbins + 1) {
      /* value above all bins */
    } else {
      /* in a bin! */
      counts[ub - 1] += 1;
    }
  }
}

/* 
 * Return index of first element in range greater than value, or len
 * if not found.
 */
int upper_bound(int len, double const* data, double value) {
  int begin = 0;
  while (len > 0) {
    int half = len / 2;
    int mid = begin + half;
    if (value < data[mid]) {
      len = half;
    } else {
      begin = mid + 1;
      len -= half + 1;
    }
  }
  return begin;
}
