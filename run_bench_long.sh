#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:5:00

# Optionally set your budget code here
# #SBATCH --account=[budget code]


echo "Only run this after doing some shorter tests or you will waste your allocation."
exit 1
# Remove the above lines once you've done some shorter tests :)

DATADIR=/work/z04/shared/apt-cuda-cw/sample_data

srun ./bench -n 10 $DATADIR/uniform.bins $DATADIR/uni_100k.dat uni_100k.hist
srun ./bench -n 10 $DATADIR/uniform.bins $DATADIR/uni_1M.dat uni_1M.hist
srun ./bench -n 10 $DATADIR/uniform.bins $DATADIR/uni_10M.dat uni_10M.hist
srun ./bench -n 10 $DATADIR/uniform.bins $DATADIR/uni_100M.dat uni_100M.hist

srun ./bench -n 10 $DATADIR/exp.bins $DATADIR/exp_100k.dat exp_100k.hist
srun ./bench -n 10 $DATADIR/exp.bins $DATADIR/exp_1M.dat exp_1M.hist
srun ./bench -n 10 $DATADIR/exp.bins $DATADIR/exp_10M.dat exp_10M.hist
srun ./bench -n 10 $DATADIR/exp.bins $DATADIR/exp_100M.dat exp_100M.hist
