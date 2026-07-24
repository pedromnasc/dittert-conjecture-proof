#!/usr/bin/env python3
"""LP search using group-summed cubic binomial squares."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_degree4_column_n6 as degree4
import search_pair_sos_n6 as linear
import search_sos_n6 as base


def cubic_slack_row(
    representative: np.ndarray,
    group: np.ndarray,
    orbit_size: int,
) -> sparse.csr_array:
    images = np.unique(np.sort(group[:, representative], axis=1), axis=0)
    pairs = set()
    for image in images:
        edge = tuple(int(value) for value in image)
        for chosen_tail in itertools.combinations(range(1, 6), 2):
            chosen = (0,) + chosen_tail
            left = tuple(edge[position] for position in chosen)
            right = tuple(
                edge[position] for position in range(6) if position not in chosen
            )
            left_index = int(base.rank_multisets(np.asarray([left]), base.NV)[0])
            right_index = int(base.rank_multisets(np.asarray([right]), base.NV)[0])
            pairs.add(tuple(sorted((left_index, right_index))))
    scale = len(group) / orbit_size
    rows = []
    columns = []
    data = []
    for left, right in pairs:
        rows.extend((left, right))
        columns.extend((right, left))
        data.extend((scale, scale))
    count = base.multiset_count(3)
    return sparse.coo_array(
        (data, (rows, columns)), shape=(count, count)
    ).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("degree4_seed", type=Path)
    parser.add_argument("--output", type=Path, default=Path("n6_cubic_binomial.npz"))
    args = parser.parse_args()

    group = base.variable_group()
    degree_six_orbits, representatives, sizes = base.monomial_orbits(6, group)
    target = base.target_coefficients(representatives)
    cubic_monomials = base.multiset_array(3)

    atoms: list[tuple[int, int]] = []
    atom_keys = set()
    atom_columns = []
    for row in np.flatnonzero(target < 0):
        edge = tuple(int(value) for value in representatives[int(row)])
        if len(set(edge)) != 6:
            raise AssertionError("negative target row is not square-free")
        for chosen_tail in itertools.combinations(range(1, 6), 2):
            chosen = (0,) + chosen_tail
            left = tuple(edge[position] for position in chosen)
            right = tuple(
                edge[position] for position in range(6) if position not in chosen
            )
            left_index = int(base.rank_multisets(np.asarray([left]), base.NV)[0])
            right_index = int(base.rank_multisets(np.asarray([right]), base.NV)[0])
            left_square = tuple(sorted(left + left))
            right_square = tuple(sorted(right + right))
            left_row = int(
                degree_six_orbits[
                    int(base.rank_multisets(np.asarray([left_square]), base.NV)[0])
                ]
            )
            right_row = int(
                degree_six_orbits[
                    int(base.rank_multisets(np.asarray([right_square]), base.NV)[0])
                ]
            )
            key = (min(left_row, right_row), max(left_row, right_row), int(row))
            if key in atom_keys:
                continue
            atom_keys.add(key)
            column = np.zeros(len(target))
            column[left_row] += len(group) / int(sizes[left_row])
            column[right_row] += len(group) / int(sizes[right_row])
            column[int(row)] -= 2 * len(group) / int(sizes[int(row)])
            atoms.append((left_index, right_index))
            atom_columns.append(column)

    seed = np.load(args.degree4_seed, allow_pickle=False)
    multipliers = np.asarray(seed["multipliers"], dtype=np.int16)
    coordinates = np.asarray(seed["gram_coordinates"], dtype=float)
    lookup = {
        (int(i), int(j)): index
        for index, (i, j) in enumerate(linear.LINEAR_UPPER)
    }
    full_index = np.asarray(
        [
            [lookup[tuple(sorted((i, j)))] for j in range(base.NV)]
            for i in range(base.NV)
        ]
    )
    fixed_grams = []
    fixed_columns = []
    for multiplier, gram in zip(multipliers, coordinates[:, full_index]):
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        trace = float(np.trace(projected))
        normalized = projected / trace
        matrix = degree4.multiplier_matrix(
            len(group), multiplier, degree_six_orbits, sizes
        )
        fixed_grams.append(normalized)
        fixed_columns.append(
            np.asarray(matrix @ degree4.pair_vector(normalized)).ravel()
        )

    coefficient = np.column_stack(fixed_columns + atom_columns)
    inequalities = np.column_stack((coefficient, np.ones(len(target))))
    objective = np.zeros(coefficient.shape[1] + 1)
    objective[-1] = -1
    print(
        f"degree-four-rays={len(fixed_columns)} cubic-atoms={len(atoms)}",
        flush=True,
    )
    result = optimize.linprog(
        objective,
        A_ub=inequalities,
        b_ub=target,
        bounds=[(0, None)] * coefficient.shape[1] + [(None, None)],
        method="highs-ipm",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
        },
    )
    if not result.success:
        raise SystemExit(result.message)
    weights = np.maximum(result.x[:-1], 0)
    residual = target - coefficient @ weights
    print("margin", float(result.x[-1]), flush=True)
    print("minimum residual", float(residual.min()), flush=True)
    print("active cubic atoms", int(np.count_nonzero(weights[302:] > 1e-12)), flush=True)

    degree4_factor_blocks = []
    degree4_factors = []
    for block, (weight, gram) in enumerate(zip(weights[:302], fixed_grams)):
        if weight <= 1e-12:
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(weight * gram)
        for value, vector in zip(eigenvalues, eigenvectors.T):
            if value > 1e-12:
                degree4_factor_blocks.append(block)
                degree4_factors.append(np.sqrt(value) * vector)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        allowed=np.asarray(base.ALLOWED, dtype=np.int8),
        group=group,
        multipliers=multipliers,
        target=target,
        residual=residual,
        margin=np.asarray(float(result.x[-1])),
        degree4_factor_blocks=np.asarray(degree4_factor_blocks, dtype=np.int16),
        degree4_factors=np.asarray(degree4_factors, dtype=float),
        cubic_monomials=cubic_monomials,
        cubic_atoms=np.asarray(atoms, dtype=np.int16),
        cubic_weights=np.asarray(weights[302:]),
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
