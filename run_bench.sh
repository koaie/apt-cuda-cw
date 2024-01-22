#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:1:00

# Optionally set your budget code here
# #SBATCH --account=[budget code]

DATADIR=/work/z04/shared/apt-cuda-cw/sample_data

srun ./bench -n 3 $DATADIR/uniform.bins $DATADIR/uni_100k.dat uni_100k.hist
