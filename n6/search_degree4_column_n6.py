#!/usr/bin/env python3
"""Column generation for all degree-four monomial/linear-SOS multipliers."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_pair_sos_n6 as pair_search
import search_sos_n6 as base


def multiplier_matrix(
    group_size: int,
    multiplier: np.ndarray,
    degree_six_orbits: np.ndarray,
    degree_six_sizes: np.ndarray,
) -> sparse.csr_array:
    bases = np.concatenate(
        (
            np.repeat(multiplier.reshape(1, 4), len(pair_search.LINEAR_UPPER), axis=0),
            pair_search.LINEAR_UPPER,
        ),
        axis=1,
    )
    bases.sort(axis=1)
    indices = base.rank_multisets(bases, base.NV)
    rows = degree_six_orbits[indices]
    columns = np.arange(len(pair_search.LINEAR_UPPER), dtype=np.int32)
    gram_factor = np.where(
        pair_search.LINEAR_UPPER[:, 0] == pair_search.LINEAR_UPPER[:, 1], 1, 2
    ).astype(np.int64)
    divisors = degree_six_sizes[rows]
    numerators = gram_factor * group_size
    if np.any(numerators % divisors):
        raise AssertionError("degree-six group averaging is not integral")
    return sparse.coo_array(
        (numerators // divisors, (rows, columns)),
        shape=(len(degree_six_sizes), len(pair_search.LINEAR_UPPER)),
        dtype=np.int64,
    ).tocsr()


def pair_vector(gram: np.ndarray) -> np.ndarray:
    return np.asarray(
        [gram[int(i), int(j)] for i, j in pair_search.LINEAR_UPPER], dtype=float
    )


def slack_matrix(functional: np.ndarray) -> np.ndarray:
    result = np.zeros((base.NV, base.NV), dtype=float)
    for value, (i, j) in zip(functional, pair_search.LINEAR_UPPER):
        if i == j:
            result[int(i), int(j)] = value
        else:
            result[int(i), int(j)] = value / 2
            result[int(j), int(i)] = value / 2
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--rays-per-block", type=int, default=2)
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    parser.add_argument("--method", choices=("highs", "highs-ipm", "highs-ds"), default="highs-ipm")
    parser.add_argument("--seed-negative-rows", action="store_true")
    parser.add_argument("--prune-tolerance", type=float, default=1e-10)
    parser.add_argument("--history-per-block", type=int, default=8)
    parser.add_argument(
        "--initial-search-result",
        type=Path,
        help="seed each block from Gram coordinates in a prior search archive",
    )
    parser.add_argument("--output", type=Path, default=Path("n6_degree4_column_search.npz"))
    args = parser.parse_args()

    base.verify_ranking()
    group = base.variable_group()
    degree_four_orbits, multipliers, degree_four_sizes = base.monomial_orbits(4, group)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(6, group)
    if len(multipliers) != 302 or len(degree_six_reps) != 5605:
        raise AssertionError("unexpected orbit count")
    target = base.target_coefficients(degree_six_reps)
    matrices = [
        multiplier_matrix(len(group), multiplier, degree_six_orbits, degree_six_sizes)
        for multiplier in multipliers
    ]
    ray_grams: list[list[np.ndarray]] = [[] for _ in multipliers]
    ray_columns: list[list[np.ndarray]] = [[] for _ in multipliers]
    print(f"multiplier-blocks={len(multipliers)}", flush=True)
    if args.initial_search_result:
        initial = np.load(args.initial_search_result, allow_pickle=False)
        coordinates = np.asarray(initial["gram_coordinates"], dtype=float)
        if coordinates.shape != (len(multipliers), len(pair_search.LINEAR_UPPER)):
            raise ValueError("unexpected initial Gram-coordinate shape")
        lookup = {
            (int(i), int(j)): index
            for index, (i, j) in enumerate(pair_search.LINEAR_UPPER)
        }
        full_index = np.asarray(
            [
                [lookup[tuple(sorted((i, j)))] for j in range(base.NV)]
                for i in range(base.NV)
            ]
        )
        seeded = 0
        for block, gram in enumerate(coordinates[:, full_index]):
            eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
            projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
            trace = float(np.trace(projected))
            if trace <= args.prune_tolerance:
                continue
            normalized = projected / trace
            ray_grams[block].append(normalized)
            ray_columns[block].append(
                np.asarray(matrices[block] @ pair_vector(normalized)).ravel()
            )
            seeded += 1
        print(f"seeded prior Gram blocks={seeded}", flush=True)
    if args.seed_negative_rows:
        negative_rows = np.flatnonzero(target < 0)
        seeded = 0
        for count, row in enumerate(negative_rows, 1):
            for block, matrix in enumerate(matrices):
                functional = np.asarray(matrix[int(row), :].toarray()).ravel()
                if not np.any(functional):
                    continue
                eigenvalues, eigenvectors = np.linalg.eigh(slack_matrix(functional))
                if eigenvalues[0] >= -args.reduced_cost_tolerance:
                    continue
                vector = eigenvectors[:, 0]
                gram = np.multiply.outer(vector, vector)
                if any(float(np.sum(existing * gram)) > 1 - 1e-8 for existing in ray_grams[block]):
                    continue
                ray_grams[block].append(gram)
                ray_columns[block].append(np.asarray(matrix @ pair_vector(gram)).ravel())
                seeded += 1
            if count % 25 == 0 or count == len(negative_rows):
                print(
                    f"seeded negative rows {count}/{len(negative_rows)} rays={seeded}",
                    flush=True,
                )

    accepted_grams: list[np.ndarray] | None = None
    accepted_residual: np.ndarray | None = None
    final_margin: float | None = None
    for iteration in range(args.iterations):
        block_sizes = [len(columns) for columns in ray_columns]
        all_columns = [column for columns in ray_columns for column in columns]
        if all_columns:
            produced = np.column_stack(all_columns)
            inequalities = sparse.csr_array(
                np.concatenate((produced, np.ones((len(target), 1))), axis=1)
            )
            number_of_rays = produced.shape[1]
        else:
            inequalities = sparse.csr_array(np.ones((len(target), 1)))
            number_of_rays = 0
        objective = np.zeros(number_of_rays + 1)
        objective[-1] = -1.0
        result = optimize.linprog(
            objective,
            A_ub=inequalities,
            b_ub=target,
            bounds=[(0.0, None)] * number_of_rays + [(None, None)],
            method=args.method,
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            print(result.message, flush=True)
            raise SystemExit(1)
        weights = np.maximum(result.x[:-1], 0.0)
        if all_columns:
            residual = target - produced @ weights
        else:
            residual = target.copy()
        dual = -np.asarray(result.ineqlin.marginals)
        if dual.min() < -2e-7 or abs(dual.sum() - 1) > 2e-6:
            raise AssertionError(f"unexpected dual min={dual.min()} sum={dual.sum()}")

        reduced_minimum = float("inf")
        candidate_rays: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
        negative_blocks = 0
        for matrix in matrices:
            functional = np.asarray(matrix.T @ dual).ravel()
            eigenvalues, eigenvectors = np.linalg.eigh(slack_matrix(functional))
            block_candidates = []
            for index in range(min(args.rays_per_block, base.NV)):
                vector = eigenvectors[:, index]
                gram = np.multiply.outer(vector, vector)
                column = np.asarray(matrix @ pair_vector(gram)).ravel()
                reduced = float(dual @ column)
                if abs(reduced - eigenvalues[index]) > 5e-7 * (1 + abs(eigenvalues[index])):
                    raise AssertionError("reduced-cost mismatch")
                block_candidates.append((reduced, gram, column))
            if block_candidates[0][0] < -args.reduced_cost_tolerance:
                negative_blocks += 1
            reduced_minimum = min(reduced_minimum, block_candidates[0][0])
            candidate_rays.append(block_candidates)
        active = int(np.count_nonzero(weights > 1e-10))
        print(
            f"iteration={iteration} rays={number_of_rays} active={active} "
            f"margin={float(result.x[-1]):.12g} residual={float(residual.min()):.12g} "
            f"negative-blocks={negative_blocks} reduced-min={reduced_minimum:.12g}",
            flush=True,
        )
        if residual.min() > 1e-7:
            accepted_grams = []
            cursor = 0
            for block, size in enumerate(block_sizes):
                block_weights = weights[cursor : cursor + size]
                accepted_grams.append(
                    sum(
                        (
                            weight * gram
                            for weight, gram in zip(block_weights, ray_grams[block])
                        ),
                        np.zeros((base.NV, base.NV)),
                    )
                )
                cursor += size
            accepted_residual = residual
            final_margin = float(result.x[-1])
            break
        added = 0
        cursor = 0
        for block, size in enumerate(block_sizes):
            block_weights = weights[cursor : cursor + size]
            old_grams = ray_grams[block]
            aggregate = sum(
                (
                    weight * gram
                    for weight, gram in zip(block_weights, ray_grams[block])
                ),
                np.zeros((base.NV, base.NV)),
            )
            trace = float(np.trace(aggregate))
            retained: list[np.ndarray] = []
            if trace > args.prune_tolerance:
                normalized = aggregate / trace
                retained.append(normalized)
            if args.history_per_block:
                retained.extend(old_grams[-args.history_per_block :])
            ray_grams[block] = retained
            ray_columns[block] = [
                np.asarray(matrices[block] @ pair_vector(gram)).ravel()
                for gram in retained
            ]
            cursor += size
        if cursor != len(weights):
            raise AssertionError("pruning weight partition failed")
        for block, candidates in enumerate(candidate_rays):
            for reduced, gram, column in candidates:
                if reduced < -args.reduced_cost_tolerance:
                    ray_grams[block].append(gram)
                    ray_columns[block].append(column)
                    added += 1
        if not added:
            raise SystemExit("dual PSD feasible without a positive certificate margin")

    if (
        accepted_grams is None
        or accepted_residual is None
        or final_margin is None
    ):
        raise SystemExit("no positive certificate margin within the iteration limit")

    factor_blocks = []
    factors = []
    for block, gram in enumerate(accepted_grams):
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        for value, vector in zip(eigenvalues, eigenvectors.T):
            if value <= 1e-12:
                continue
            factor_blocks.append(block)
            factors.append(np.sqrt(value) * vector)
    print("ACCEPTED numerical certificate", flush=True)
    print("minimum residual", float(accepted_residual.min()), flush=True)
    print("nonzero factors", len(factors), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        allowed=np.asarray(base.ALLOWED, dtype=np.int8),
        group=group,
        multipliers=multipliers,
        degree_six_representatives=degree_six_reps,
        degree_six_sizes=degree_six_sizes,
        target=target,
        residual=accepted_residual,
        margin=np.asarray(final_margin),
        factor_blocks=np.asarray(factor_blocks, dtype=np.int16),
        factors=np.asarray(factors, dtype=float),
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
