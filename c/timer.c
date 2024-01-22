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

#include "timer.h"
#include <stdint.h>

void timer_start(Timer* t) {
  timespec_get(&t->start, TIME_UTC);
}
void timer_stop(Timer* t) {
  timespec_get(&t->stop, TIME_UTC);
}

double timer_check(Timer* t) {
  int64_t dt_s = (int64_t)t->stop.tv_sec - (int64_t)t->start.tv_sec;
  int64_t ans = dt_s * 1000000000;
  ans += t->stop.tv_nsec - t->start.tv_nsec;
  return ans * 1e-9;
}
