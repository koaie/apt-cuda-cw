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

#include "io.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define CHUNK_SIZE 1024
typedef struct chunk {
  int pos;
  struct chunk* next;
  double data[CHUNK_SIZE];
} chunk;

chunk* chunk_new() {
  chunk* ans = (chunk*)malloc(sizeof(chunk));
  ans->pos = 0;
  ans->next = NULL;
  return ans;
}

void chunk_free(chunk* c) {
  if (c->next) {
    chunk_free(c->next);
  }
  free(c);
}

/*
 * Read all the data from the file (one floating point value per line)
 * into a buffer.
 *
 * Return the size and buffer in the locations pointed
 * to by nread and data respectively. Caller is responsible for
 * freeing data. 
 *
 * Return value is error code (0 == success).
 */
int read_column_data(char const* fn, int* nread, double** data) {
  FILE* fh = fopen(fn, "r");
  if (fh == NULL) {
    fprintf(stderr, "Can't open file '%s'", fn);
    return 1;
  }

  chunk* const HEAD = chunk_new();
  int nchunks = 1;

  chunk* last = HEAD;
  while(1) {
    double tmp = 0.0;
    int nread = fscanf(fh, "%le ", &tmp);
    if (nread == 1) {
      /* OK - store it */
      /* Maybe we need to allocate a new chunk */
      if (last->pos == CHUNK_SIZE) {
	chunk* next = chunk_new();
	nchunks++;
	last->next = next;
	last = next;
      }
      last->data[last->pos] = tmp;
      last->pos++;
    } else {
      /* We're done */
      break;
    }
  }

  /* Done with the file. */
  fclose(fh);

  /* First, compute size and allocate. */
  *nread = CHUNK_SIZE * (nchunks - 1) + last->pos;
  *data = (double*)malloc(*nread * sizeof(double));

  /* Now copy chunks into a contiguous buffer. */
  double* p = *data;
  for(
      chunk* current = HEAD;
      current != NULL;
      current = current->next
      ) {
    memcpy(p, current->data, current->pos * sizeof(double));
    p += current->pos;
  }

  chunk_free(HEAD);
  return 0;
}

/* For the Fortran version to clean up with. */
void free_column_data(double** data) {
  free(*data);
}

int write_hist(char const* fn, int const nbins, double const* edges, int const* counts) {
  FILE* fh = fopen(fn, "w");
  fprintf(fh, "# low, high, count\n");
  
  for (int i = 0; i < nbins; ++i) {
    fprintf(fh, "%e,%e,%d\n", edges[i], edges[i+1], counts[i]);
  }

  fclose(fh);

  return 0;
}
