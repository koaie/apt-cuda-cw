# APT CUDA programming coursework

Expected time to complete: 5 hours

Remember what you submit should be your own work. Anything not your
own work which you have accessed should be correctly referenced and
cited. You must not share this assessment's source code nor your
solutions. Please see further information and guidance from the School
of Informatics Academic Misconduct Officer:
https://web.inf.ed.ac.uk/infweb/admin/policies/academic-misconduct

## Summary

Your goal here is to take a naïve, serial code that computes the
histogram of some data, and make this run correctly and efficiently
using a single GPU on Cirrus (i.e. one NVIDIA V100).

You must also prepare a brief report (maximum 1 page) explaining, with
reference to the algorithm and hardware, how the changes you have made
to the code achieve this performance improvement.

As part of marking, your code will be compiled and run on Cirrus using
the `gpu` partition, so please use that for any tuning and
profiling you do.

## Problem description

The histogram function must work as follows. Given:

- `B` bins, defined by an array of `B + 1` floating point values
  giving the bin edges, this array being sorted into ascending order;

- an array of `N` floating point data values to be histogrammed;

- an allocated but uninitialised output array of `B` integers to hold
  the result;

the function must store in `output[i]` the number of values in the
data array that are within the bin, i.e. `bin_edges[i] <= x < bin_edges[i]`.

The code includes a serial, CPU-only implementation of this algorithm
for your reference as well as a direct port of this to CUDA (i.e.
purely serial using a single thread and block).

There are two versions of the software available, written in C and
Fortran. **Please choose one version to work with as you prefer -
there is no difference in marks available.**

## Set up

1. Clone the code repository into your home directory
`git clone /work/z04/shared/apt-cuda-cw/apt-cuda-cw`

2. Sample input data for your testing is available on a shared folder
on Cirrus:
```
ls -lh /work/z04/shared/apt-cuda-cw/sample_data
total 5.2G
-rw-r--r-- 1 rnashz04 z04 2.4M Jan 19 16:14 exp_100k.dat
-rw-r--r-- 1 rnashz04 z04 2.4G Jan 19 16:18 exp_100M.dat
-rw-r--r-- 1 rnashz04 z04 239M Jan 19 16:14 exp_10M.dat
-rw-r--r-- 1 rnashz04 z04  24M Jan 19 16:14 exp_1M.dat
-rw-r--r-- 1 rnashz04 z04  124 Jan 19 16:20 exp.bins
-rw-r--r-- 1 rnashz04 z04 2.4M Jan 19 16:09 uni_100k.dat
-rw-r--r-- 1 rnashz04 z04 2.4G Jan 19 16:14 uni_100M.dat
-rw-r--r-- 1 rnashz04 z04 239M Jan 19 16:10 uni_10M.dat
-rw-r--r-- 1 rnashz04 z04  24M Jan 19 16:09 uni_1M.dat
-rw-r--r-- 1 rnashz04 z04   44 Jan 19 16:20 uniform.bins
```

3. Choose C (`cd apt-cuda-cw/c`)or Fortran (`cd apt-cuda-cw/fortran`)
versions of the assessment.

4. Load the required NVIDIA module (`module load nvidia/nvhpc/22.11`).

5. Compile the code: `make`

This will produce two executables: `test` and `bench`.

6. Confirm the code works by running the basic (CPU) unit tests on the login node:

```
[user@cirrus-login2 c]$ ./test 
Test 'upper_bound': PASS with 7 assertions
Test 'hist': PASS with 6 assertions
Test 'read_data': PASS with 4 assertions
```

6. Confirm you can run on the GPUs. First, edit the `run_bench.sh`
submission script to specify your budget code. Then submit to SLURM
for execution with `sbatch run_bench.sh`. An example output for the
unmodified code (C version, Fortran is similar) is:

```
Read 11 bin edges from '/lustre/home/shared/apt-cuda-cw/sample_data/uniform.bins'
Read 100000 data points from '/lustre/home/shared/apt-cuda-cw/sample_data/uni_100k.dat'
Beginning 3 CPU runs
Run 0: 2.137458e-03 s
Run 1: 2.139035e-03 s
Run 2: 2.133959e-03 s
Beginning 3 GPU runs
Run 0: 3.725760e-02 s
Run 1: 3.718957e-02 s
Run 2: 3.577288e-02 s
CPU and GPU histograms match
Writing histogram to 'uni_100k.dat.hist'

Summary for CPU (all in s):
min = 2.133959e-03, max = 2.139035e-03, mean = 2.136817e-03, std = 2.597938e-06

Summary for GPU (all in s):
min = 3.577288e-02, max = 3.725760e-02, mean = 3.674002e-02, std = 8.382561e-04
```

## Requirements for your code

The code will be marked on correctness [10%], clarity [10%], and
performance [30%].

Your code will form part of the submission and is worth 50% of the
marks. You may only change either `c/hist_gpu.cu` or
`fortran/hist_gpu.cuf`. This submitted file will be added to a copy of
the supplied repository for benchmarking.

Your code must compile (with only the modules given above loaded)
simply by running `make` and run with the existing command line
options. Note that a code that does not compile is, by definition
incorrect, and that a code that produces incorrect results scores no
points for performance.

You must ensure that you wait for kernels to complete *inside* the
`compute_histogram_gpu` function, in order that the timings are
correct. If you remove the synchronisation or otherwise try to trick
the timing code this will be treated as academic misconduct through
the usual processes. If in doubt, please check with me.

You may use functions and types from the standard library and base
CUDA library, but from no other sources.

**To be sure your code works, please test it on Cirrus and submit that
version!**

I have provided a directory with 5 GB of sample data in various sizes
you can use for testing and tuning. The path for this is
`/work/z04/shared/apt-cuda-cw/sample_data` on Cirrus or you can
generate this using the script (`sample_data/generate.sh`) if you have
access to another GPU and wish to work offline. The `run_bench.sh`
script runs one small test while `run_bench_long.sh` runs on all the
supplied data sets and may take a while with an unoptimised code (so
please wait until you've made some progress before trying this!). You
can, of course, use these as templates for your own tests.

**Correctness**: the benchmark code compares the GPU results to those from
the (unmodified) serial CPU version.

**Clarity**: your modifications will be marked for usual good practice
in programming. Are variables/functions sensibly named? Code well
formatted? Are comments present, where necessary, to explain what is
not obvious? (To avoid doubt, your choice of decomposition of the
problem over blocks and threads is not obvious.)

**Performance**: I will use three unknown data sets for testing. Each
will consist of data sizes between one million and one billion points
(inclusive), with bin counts less than 256. At least one test case
will have non-uniform bins. Each test case will be given equal weight
(i.e. 10% of total) and scored based on the best measured run time.


## Requirements for report

This must be a PDF with a maximum of one A4 page and text at 12pt or
greater.

You must include:

0. A clear statement of whether you are using the C or Fortran version

1. A brief description (2-3 sentences) of why computing the histogram
   in parallel is not a trivial task on an NVIDIA GPU. [10 %]

2. A description of the approach you have chosen, which features of
   the hardware/software it uses and how this gives good
   performance. You might choose to refer to your code (e.g. "see
   comments on lines 90-95 of hist_gpu.cuf for full details.") [30%]

3. A brief discussion of your process. You may wish to include: approaches
   tried but discounted; description of any debugging/profiling/tuning;
   potential improvements considered but not attempted. [10%]

## Submission

Please see the instructions on Learn for full details, but you will
have to submit the code and report to separate queues.

**It is very important that you use your exam number for both the code
and report filenames so we can match them up!** E.g., the report should
be `B0123456.pdf` and the code either `B0123456.cu` or `B0123456.cuf`.

## Hints

You have been provided the code in a version controlled git
repository - I suggest that you make use of it to track your progress.

I highly recommend starting with a look at the existing code and
thinking about point 1 from the report. Consider how you will get data
from GPU global memory into the SMs and back again. Consider how you
will decompose the index space between blocks and threads.

You may wish to get interactive access to a GPU if you are doing
debugging/performance tuning. Use this SLURM command:
```
srun --time=0:10:0 --partition=gpu --qos=short --gres=gpu:1 --account=$BUDGETCODE --pty /usr/bin/bash
```
(replacing `$BUDGETCODE` with project budget).

Warp divergence is not a major issue in this case.
