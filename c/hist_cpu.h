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
#ifndef APT_CUDACW_HIST_CPU_H
#define APT_CUDACW_HIST_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

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
void compute_histogram_cpu(
			   int const nbins, double const* bin_edges,
			   int const ndata, double const* data,
			   int* counts
			   );

/*
 * Simple self-consistency checks (args as above).
 */
void check_bin_self_consistency(int const nbins, double const* bin_edges);

/* 
 * Return index of first element in range greater than value, or len
 * if not found.
 */
int upper_bound(int len, double const* data, double value);

#ifdef __cplusplus
}
#endif
#endif
