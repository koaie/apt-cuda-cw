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

#ifndef APT_CUDACW_IO_H
#define APT_CUDACW_IO_H

#ifdef __cplusplus
extern "C" {
#endif

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
int read_column_data(char const* fn, int* nread, double** data);

/*
 * Write histogram data to the specified file.
 */
int write_hist(char const* fn, int const nbins, double const* edges, int const* counts);

#ifdef __cplusplus
}
#endif

#endif
