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

#ifndef APT_CUDACW_TIMER_H
#define APT_CUDACW_TIMER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <time.h>

typedef struct Timer {
  struct timespec start;
  struct timespec stop;
} Timer;

/* Start a timer */
void timer_start(Timer*);
/* Stop a timer */
void timer_stop(Timer*);
/* Return the number of seconds between timer start and stop. */
double timer_check(Timer*);

#ifdef __cplusplus
}
#endif

#endif
