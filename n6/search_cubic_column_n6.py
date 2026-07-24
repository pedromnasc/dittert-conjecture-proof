#!/usr/bin/env python3
"""Column generation for the full group-summed cubic SOS cone."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_cubic_obstruction_rays_n6 as cubic_rays
import search_degree4_column_n6 as degree4
import search_pair_sos_n6 as linear
import search_sos_n6 as base


def cubic_dual_slack(
    dual: np.ndarray,
    representatives: np.ndarray,
    sizes: np.ndarray,
    group: np.ndarray,
) -> sparse.csr_array:
    entries: dict[tuple[int, int], float] = {}
    active = np.flatnonzero(dual > 1e-10)
    print(f"  cubic dual support={len(active)}", flush=True)
    for count, row in enumerate(active, 1):
        images = np.unique(np.sort(group[:, representatives[int(row)]], axis=1), axis=0)
        pairs = set()
        for image in images:
            monomial = tuple(int(value) for value in image)
            for chosen in itertools.combinations(range(6), 3):
                chosen_set = set(chosen)
                left = tuple(monomial[position] for position in chosen)
                right = tuple(
                    monomial[position]
                    for position in range(6)
                    if position not in chosen_set
                )
                left_index = int(
                    base.rank_multisets(np.asarray([left]), base.NV)[0]
                )
                right_index = int(
                    base.rank_multisets(np.asarray([right]), base.NV)[0]
                )
                pairs.add(tuple(sorted((left_index, right_index))))
        value = float(dual[int(row)]) * len(group) / int(sizes[int(row)])
        for pair in pairs:
            entries[pair] = entries.get(pair, 0.0) + value
        if count % 100 == 0:
            print(f"  assembled dual rows {count}/{len(active)}", flush=True)
    rows = []
    columns = []
    data = []
    for (left, right), value in entries.items():
        rows.append(left)
        columns.append(right)
        data.append(value)
        if left != right:
            rows.append(right)
            columns.append(left)
            data.append(value)
    count = base.multiset_count(3)
    result = sparse.coo_array(
        (data, (rows, columns)), shape=(count, count)
    ).tocsr()
    print(f"  cubic slack nnz={result.nnz}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("degree4_seed", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rays-per-iteration", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("n6_cubic_column.npz"))
    args = parser.parse_args()

    group = base.variable_group()
    degree_six_orbits, representatives, sizes = base.monomial_orbits(6, group)
    target = base.target_coefficients(representatives)
    cubic_monomials = base.multiset_array(3)
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

    cubic_vectors: list[np.ndarray] = []
    cubic_columns: list[np.ndarray] = []
    accepted_weights = None
    accepted_residual = None
    accepted_margin = None
    for iteration in range(args.iterations):
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
        dual = -np.asarray(result.ineqlin.marginals)
        print(
            f"iteration={iteration} rays={len(cubic_vectors)} "
            f"margin={float(result.x[-1]):.12g} residual={float(residual.min()):.12g}",
            flush=True,
        )
        if residual.min() > 1e-7:
            accepted_weights = weights
            accepted_residual = residual
            accepted_margin = float(result.x[-1])
            break
        slack = cubic_dual_slack(dual, representatives, sizes, group)
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            slack,
            k=args.rays_per_iteration,
            which="SA",
            tol=1e-8,
        )
        order = np.argsort(eigenvalues)
        print("  cubic minimum eigenvalues", eigenvalues[order], flush=True)
        added = 0
        for index in order:
            if eigenvalues[index] >= -1e-9:
                continue
            vector = eigenvectors[:, index]
            cubic_vectors.append(vector)
            cubic_columns.append(
                cubic_rays.cubic_ray_column(
                    vector, cubic_monomials, degree_six_orbits, sizes, len(group)
                )
            )
            added += 1
        if not added:
            raise SystemExit("cubic dual slack is PSD before a positive margin")

    if accepted_weights is None or accepted_residual is None or accepted_margin is None:
        raise SystemExit("no positive margin within the iteration limit")
    degree4_factor_blocks = []
    degree4_factors = []
    for block, (weight, gram) in enumerate(zip(accepted_weights[:302], fixed_grams)):
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
        residual=accepted_residual,
        margin=np.asarray(accepted_margin),
        degree4_factor_blocks=np.asarray(degree4_factor_blocks, dtype=np.int16),
        degree4_factors=np.asarray(degree4_factors, dtype=float),
        cubic_monomials=cubic_monomials,
        cubic_vectors=np.asarray(cubic_vectors),
        cubic_weights=np.asarray(accepted_weights[302:]),
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
