#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:1:00
#SBATCH --account=m23oc-s2617564

NVHPC_VERSION=22.11
module load nvidia/nvhpc/$NVHPC_VERSION

DATADIR=/work/z04/shared/apt-cuda-cw/sample_data

cmd=(
    ncu
    -o cuda-cw-${SLURM_JOB_ID} # save to file
    --kernel-name 'hist_kernel_parallel' # specify kernel to allow skip and count (regex so it works for C and Fortran)
    --launch-skip 1 # skip first one as cold
    --launch-count 10 # collect only ten runs (as collecting detailed info)
    --set detailed # collect 
    bench -n 3 $DATADIR/exp.bins $DATADIR/uni_10M.dat uni_10M.hist # application
)
"${cmd[@]}"