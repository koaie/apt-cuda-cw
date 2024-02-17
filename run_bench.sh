#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:1:00
#SBATCH --account=m23oc-s2617564

DATADIR=/work/z04/shared/apt-cuda-cw/sample_data

srun ./bench -n 3 $DATADIR/exp.bins $DATADIR/uni_10M.dat uni_10M.hist
