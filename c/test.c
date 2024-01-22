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

#include "hist_cpu.h"
#include "io.h"

/* 
 * A tiny test running harness.
 */
typedef void (*TestFunc)(FILE*);

static FILE* TEST_OUTPUT = NULL;
static int test_error_code = 0;
static int test_n_assertions = 0;

void TEST_INIT() {
  TEST_OUTPUT = stderr;
}

#define TEST_CASE(name) void TEST_CASE_ ## name(FILE* TEST_LOG)
#define REQUIRE(expr) do {					\
    test_n_assertions++;					\
    if (!(expr)) {						\
      fprintf(TEST_LOG, "  line %d: %s", __LINE__, #expr);	\
      test_error_code = 1;					\
      return;							\
    }								\
  } while (0)

#define REQUIRE_APPROX(actual, expected, margin)	\
  REQUIRE(fabs(actual - expected) < margin)

void testrunner(char const* name, TestFunc fn) {
  char* test_log_txt;
  size_t test_log_size;
  FILE* test_log_handle = open_memstream(&test_log_txt, &test_log_size);
  test_error_code = 0;
  test_n_assertions = 0;
  fn(test_log_handle);
  fclose(test_log_handle);
  if (test_error_code == 0) {
    /* OK */
    fprintf(TEST_OUTPUT, "Test '%s': PASS with %d assertions\n", name, test_n_assertions);
  } else {
    fprintf(TEST_OUTPUT, "Test '%s': FAIL\n", name);
    fprintf(TEST_OUTPUT, test_log_txt);
    if (test_log_txt[test_log_size-1] != '\n') {
      fputc('\n', TEST_OUTPUT);
    }
  }
  free(test_log_txt);
}

#define RUN_TEST(name) testrunner(#name, TEST_CASE_ ## name)

/*
 * End of test harness.
 */


TEST_CASE(upper_bound) {
  double const edges[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  int const nedges = sizeof(edges)/ sizeof(double);

  double const data[] = {0.5, 1.5, 1.0, 4.5, 5.0, 6.0};
  int const ndata = sizeof(data) / sizeof(double);

  int const expected[] = {1, 2, 2, 5, 6, 6};
  int const nexpected = sizeof(expected) / sizeof(int);
  REQUIRE(nexpected == ndata);

  for (int i = 0; i < ndata; ++i) {
    int ub = upper_bound(nedges, edges, data[i]);
    REQUIRE(ub == expected[i]);
  }
}

TEST_CASE(hist) {
  double const edges[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  int const nedges = sizeof(edges)/ sizeof(double);
  int const nbins = nedges - 1;

  double const data[] = {-0.5, 0.0, 0.5, 1.5, 1.0, 4.5, 5.0, 6.0};
  int const ndata = sizeof(data) / sizeof(double);

  int h[nbins];
  compute_histogram_cpu(nbins, edges, ndata, data, h);

  int const expected[] = {2, 2, 0, 0, 1};
  REQUIRE(sizeof(expected)/sizeof(int) == nbins);
  for (int i = 0; i < nbins; ++i) {
    REQUIRE(h[i] == expected[i]);
  }
}

TEST_CASE(read_data) {
  int ndata;
  double* data;
  int err = read_column_data("../sample_data/uni_100k.dat", &ndata, &data);
  REQUIRE(!err);
  REQUIRE(ndata == 100000);
  REQUIRE_APPROX(data[0], 0.631278952142, 1e-12);
  REQUIRE_APPROX(data[ndata - 1], 0.303617733625, 1e-12);
}

int main() {
  TEST_INIT();
  RUN_TEST(upper_bound);
  RUN_TEST(hist);
  RUN_TEST(read_data);
}
