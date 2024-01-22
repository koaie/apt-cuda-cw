import argparse
import sys

import numpy as np

def comma_int(s):
    return int(s.replace(",", ""))

def uniform_dist(rng):
    return rng.uniform
def gaussian_dist(rng):
    return rng.normal
def exponential_dist(rng):
    return rng.exponential

distributions = {
    "uniform": uniform_dist,
    "gaussian": gaussian_dist,
    "exponential": exponential_dist
}

p = argparse.ArgumentParser()

p.add_argument("--seed", default=9876, type=int, help="Random seed")
p.add_argument("--n", default=1000, type=comma_int, help="Number of data points to generate")
p.add_argument("--dist", default="uniform", choices=distributions.keys())
p.add_argument("output", type=argparse.FileType("w"), nargs="?", default=sys.stdout)

args = p.parse_args()

rng = np.random.default_rng(args.seed)
dist = distributions[args.dist](rng)

data = dist(size=args.n)
np.savetxt(args.output, data)
