#!/bin/bash

if [[ "$HOSTNAME" =~ "cirrus-login" ]]; then
    # On Cirrus need to load conda for numpy
    module load anaconda/python3
fi

echo "Generate uniform data"
echo "Generating 100k..."
python3 generate_data.py --n     100,000 --seed 423542 uni_100k.dat
echo "Generating 1M..."
python3 generate_data.py --n   1,000,000 --seed 423543 uni_1M.dat
echo "Generating 10M..."
python3 generate_data.py --n  10,000,000 --seed 423544 uni_10M.dat
echo "Generating 100M..."
python3 generate_data.py --n 100,000,000 --seed 423545 uni_100M.dat

echo "Generate exponential data"
echo "Generating 100k..."
python3 generate_data.py --n     100,000 --seed 523542 --dist exponential exp_100k.dat
echo "Generating 1M..."
python3 generate_data.py --n   1,000,000 --seed 523543 --dist exponential exp_1M.dat
echo "Generating 10M..."
python3 generate_data.py --n  10,000,000 --seed 523544 --dist exponential exp_10M.dat
echo "Generating 100M..."
python3 generate_data.py --n 100,000,000 --seed 523545 --dist exponential exp_100M.dat

