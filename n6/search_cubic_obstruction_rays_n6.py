#!/usr/bin/env python3
"""LP search with dense cubic-square rays from hyperedge obstruction matrices."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_cubic_binomial_n6 as cubic
import search_degree4_column_n6 as degree4
import search_pair_sos_n6 as linear
import search_sos_n6 as base


def cubic_ray_column(
    vector: np.ndarray,
    cubic_monomials: np.ndarray,
    degree_six_orbits: np.ndarray,
    degree_six_sizes: np.ndarray,
    group_size: int,
) -> np.ndarray:
    support = np.flatnonzero(np.abs(vector) > 1e-9)
    left, right = np.triu_indices(len(support))
    left_variables = support[left]
    right_variables = support[right]
    monomials = np.concatenate(
        (cubic_monomials[left_variables], cubic_monomials[right_variables]), axis=1
    )
    monomials.sort(axis=1)
    rows = degree_six_orbits[base.rank_multisets(monomials, base.NV)]
    factors = np.where(left == right, 1.0, 2.0)
    values = (
        factors
        * vector[left_variables]
        * vector[right_variables]
        * group_size
        / degree_six_sizes[rows]
    )
    return np.bincount(
        rows, weights=values, minlength=len(degree_six_sizes)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("degree4_seed", type=Path)
    parser.add_argument("--output", type=Path, default=Path("n6_cubic_rays.npz"))
    args = parser.parse_args()

    group = base.variable_group()
    degree_six_orbits, representatives, sizes = base.monomial_orbits(6, group)
    target = base.target_coefficients(representatives)
    cubic_monomials = base.multiset_array(3)
    cubic_vectors = []
    cubic_columns = []
    for count, row in enumerate(np.flatnonzero(target < 0), 1):
        slack = cubic.cubic_slack_row(
            representatives[int(row)], group, int(sizes[int(row)])
        )
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            slack, k=1, which="SA", tol=1e-9
        )
        vector = eigenvectors[:, 0]
        cubic_vectors.append(vector)
        cubic_columns.append(
            cubic_ray_column(
                vector, cubic_monomials, degree_six_orbits, sizes, len(group)
            )
        )
        if count % 20 == 0 or count == np.count_nonzero(target < 0):
            print(f"built cubic rays {count}/{np.count_nonzero(target < 0)}", flush=True)

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
    coefficient = np.column_stack(fixed_columns + cubic_columns)
    inequalities = np.column_stack((coefficient, np.ones(len(target))))
    objective = np.zeros(coefficient.shape[1] + 1)
    objective[-1] = -1
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
    print("active cubic rays", int(np.count_nonzero(weights[302:] > 1e-12)), flush=True)

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
        cubic_vectors=np.asarray(cubic_vectors),
        cubic_weights=np.asarray(weights[302:]),
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
