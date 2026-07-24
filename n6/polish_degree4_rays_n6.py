#!/usr/bin/env python3
"""Polish an approximate SDP solution by LP-reweighting its PSD eigen-rays."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_degree4_column_n6 as column
import search_pair_sos_n6 as pair_search
import search_sos_n6 as base


def full_coordinate_index() -> np.ndarray:
    lookup = {
        (int(i), int(j)): index
        for index, (i, j) in enumerate(pair_search.LINEAR_UPPER)
    }
    return np.asarray(
        [
            [lookup[tuple(sorted((i, j)))] for j in range(base.NV)]
            for i in range(base.NV)
        ],
        dtype=np.int32,
    )


def pair_vector(vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [vector[int(i)] * vector[int(j)] for i, j in pair_search.LINEAR_UPPER]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("search_result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="highs-ipm", choices=("highs", "highs-ipm", "highs-ds"))
    args = parser.parse_args()

    source = np.load(args.search_result, allow_pickle=False)
    group = np.asarray(source["group"], dtype=np.int16)
    multipliers = np.asarray(source["multipliers"], dtype=np.int16)
    coordinates = np.asarray(source["gram_coordinates"], dtype=float)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(
        6, group
    )
    target = base.target_coefficients(degree_six_reps)
    matrices = [
        column.multiplier_matrix(
            len(group), multiplier, degree_six_orbits, degree_six_sizes
        )
        for multiplier in multipliers
    ]

    grams = coordinates[:, full_coordinate_index()]
    ray_vectors = []
    ray_blocks = []
    columns = []
    for block, (gram, matrix) in enumerate(zip(grams, matrices)):
        _, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        for ray in range(base.NV):
            vector = eigenvectors[:, ray]
            value = np.asarray(matrix @ pair_vector(vector)).reshape(-1, 1)
            columns.append(sparse.csc_array(value))
            ray_vectors.append(vector)
            ray_blocks.append(block)
    coefficient = sparse.hstack(columns, format="csc")
    inequalities = sparse.hstack(
        (coefficient, np.ones((len(target), 1))), format="csc"
    )
    objective = np.zeros(coefficient.shape[1] + 1)
    objective[-1] = -1
    print(
        f"rays={coefficient.shape[1]} coefficient-shape={coefficient.shape} "
        f"nnz={coefficient.nnz}",
        flush=True,
    )
    result = optimize.linprog(
        objective,
        A_ub=inequalities,
        b_ub=target,
        bounds=[(0.0, None)] * coefficient.shape[1] + [(None, None)],
        method=args.method,
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
        },
    )
    if not result.success:
        raise SystemExit(result.message)
    weights = np.maximum(np.asarray(result.x[:-1]), 0)
    residual = target - coefficient @ weights
    print("margin", float(result.x[-1]), flush=True)
    print("minimum residual", float(residual.min()), flush=True)
    print("active rays", int(np.count_nonzero(weights > 1e-12)), flush=True)

    polished_grams = np.zeros((len(multipliers), base.NV, base.NV))
    for weight, block, vector in zip(weights, ray_blocks, ray_vectors):
        if weight:
            polished_grams[block] += weight * np.multiply.outer(vector, vector)
    polished_coordinates = np.asarray(
        [
            [gram[int(i), int(j)] for i, j in pair_search.LINEAR_UPPER]
            for gram in polished_grams
        ]
    )
    minimum_eigenvalues = np.linalg.eigvalsh(polished_grams)[:, 0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        allowed=np.asarray(base.ALLOWED, dtype=np.int8),
        group=group,
        multipliers=multipliers,
        degree_six_representatives=degree_six_reps,
        degree_six_sizes=degree_six_sizes,
        target=target,
        residual=np.asarray(residual),
        margin=np.asarray(float(result.x[-1])),
        minimum_eigenvalues=minimum_eigenvalues,
        gram_coordinates=polished_coordinates,
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
