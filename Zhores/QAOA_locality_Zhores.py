"""Seeded command-line locality experiment using the validated shared kernel.

Legacy positional CLI and CSV row layout are preserved. Beta denotes the cost
angle; gamma denotes the mixer angle. Importing this module performs no work.
"""
import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

# This entry point must also work when invoked from outside the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from QAOA_locality import (get_binary_strings, get_circuit_operators,
                           generate_random_QAOA_operator, make_grid,
                           count_solutions_XI as _count_solutions_XI)


def generate_QAOA_operator(nqubits, locality, number_of_terms,
                           number_of_layers=1, beta_angle=None,
                           gamma_angle=None, *, seed=None):
    return generate_random_QAOA_operator(
        nqubits, locality, number_of_terms, number_of_layers,
        None if beta_angle is None or len(beta_angle) == 0 else beta_angle,
        None if gamma_angle is None or len(gamma_angle) == 0 else gamma_angle,
        seed=seed)


def count_solutions_XI(nqubits, Operator, tol=1e-10):
    if np.shape(Operator) != (2**nqubits, 2**nqubits):
        raise ValueError('Operator dimensions do not match nqubits.')
    return _count_solutions_XI(Operator, tol)


def compiler(n_qubits, n_layers, n_terms, n_locality, *, seed=None):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    beta = rng.random(n_layers)*np.pi/3
    gamma = rng.random(n_layers)*np.pi/6
    grid = make_grid(n_qubits, n_locality, n_terms, n_layers, beta, gamma,
                     seed=rng)
    return grid, time.perf_counter()-start, beta, gamma


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('n_qubits', 'n_layers', 'n_terms', 'n_locality'):
        parser.add_argument(name, type=int)
    parser.add_argument('path', type=Path)
    parser.add_argument('index', type=int)
    args = parser.parse_args(argv)
    if min(args.n_qubits, args.n_layers, args.n_terms, args.n_locality) < 1:
        parser.error('Dimensions, layers, terms, and locality must be positive.')
    seed = args.index % 200
    grid, elapsed, beta, gamma = compiler(
        args.n_qubits, args.n_layers, args.n_terms, args.n_locality, seed=seed)
    name = (f'{args.n_qubits}n_{args.n_layers}p_{args.n_terms}t_'
            f'{args.n_locality}l_{seed}seed.csv')
    args.path.mkdir(parents=True, exist_ok=True)
    with (args.path/name).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([args.n_qubits, args.n_layers, args.n_terms, args.n_locality])
        writer.writerow([elapsed])
        writer.writerow(beta)
        writer.writerow(gamma)
        np.savetxt(stream, grid.reshape(grid.shape[0], -1), delimiter=',', fmt='%.17g')
    return args.path/name


if __name__ == '__main__':
    main()
