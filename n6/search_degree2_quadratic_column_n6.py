#!/usr/bin/env python3
"""Column generation for degree-two multipliers times quadratic SOS forms."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_sos_n6 as base
import search_degree4_column_n6 as degree4
import search_pair_sos_n6 as linear


QUADRATIC_UPPER = np.column_stack(np.triu_indices(len(base.M2))).astype(np.int32)


def multiplier_matrix(
    group_size: int,
    multiplier: np.ndarray,
    degree_six_orbits: np.ndarray,
    degree_six_sizes: np.ndarray,
) -> sparse.csr_array:
    left = QUADRATIC_UPPER[:, 0]
    right = QUADRATIC_UPPER[:, 1]
    bases = np.concatenate(
        (
            np.repeat(multiplier.reshape(1, 2), len(QUADRATIC_UPPER), axis=0),
            base.M2[left],
            base.M2[right],
        ),
        axis=1,
    )
    bases.sort(axis=1)
    ranks = base.rank_multisets(bases, base.NV)
    rows = degree_six_orbits[ranks]
    columns = np.arange(len(QUADRATIC_UPPER), dtype=np.int32)
    gram_factor = np.where(left == right, 1, 2).astype(np.int64)
    numerators = gram_factor * group_size
    divisors = degree_six_sizes[rows]
    if np.any(numerators % divisors):
        raise AssertionError("degree-six orbit averaging is not integral")
    return sparse.coo_array(
        (numerators // divisors, (rows, columns)),
        shape=(len(degree_six_sizes), len(QUADRATIC_UPPER)),
        dtype=np.int64,
    ).tocsr()


def upper_vector(gram: np.ndarray) -> np.ndarray:
    return gram[QUADRATIC_UPPER[:, 0], QUADRATIC_UPPER[:, 1]]


def ray_gram(ray: np.ndarray) -> np.ndarray:
    return np.multiply.outer(ray, ray) if ray.ndim == 1 else ray


def ray_upper(ray: np.ndarray) -> np.ndarray:
    if ray.ndim == 1:
        return ray[QUADRATIC_UPPER[:, 0]] * ray[QUADRATIC_UPPER[:, 1]]
    return upper_vector(ray)


def slack_matrix(functional: np.ndarray) -> np.ndarray:
    size = len(base.M2)
    result = np.zeros((size, size), dtype=float)
    left = QUADRATIC_UPPER[:, 0]
    right = QUADRATIC_UPPER[:, 1]
    result[left, right] = functional
    off_diagonal = left != right
    result[left[off_diagonal], right[off_diagonal]] /= 2
    result[right[off_diagonal], left[off_diagonal]] = result[
        left[off_diagonal], right[off_diagonal]
    ]
    return result


def sparse_slack_row(matrix: sparse.csr_array, row: int) -> sparse.csr_array:
    values = matrix[[row], :].tocoo()
    coordinates = values.col
    left = QUADRATIC_UPPER[coordinates, 0]
    right = QUADRATIC_UPPER[coordinates, 1]
    diagonal = left == right
    rows = np.concatenate((left, right[~diagonal]))
    columns = np.concatenate((right, left[~diagonal]))
    data = np.concatenate(
        (np.where(diagonal, values.data, values.data / 2), values.data[~diagonal] / 2)
    )
    return sparse.coo_array(
        (data, (rows, columns)), shape=(len(base.M2), len(base.M2))
    ).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rays-per-block", type=int, default=3)
    parser.add_argument("--history-per-block", type=int, default=8)
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    parser.add_argument("--prune-tolerance", type=float, default=1e-12)
    parser.add_argument("--seed-negative-rows", action="store_true")
    parser.add_argument(
        "--degree4-seed",
        type=Path,
        help="add fixed PSD degree-four/linear Gram rays from a search archive",
    )
    parser.add_argument("--price-degree4", action="store_true")
    parser.add_argument("--degree4-rays-per-block", type=int, default=1)
    parser.add_argument(
        "--method", choices=("highs", "highs-ipm", "highs-ds"), default="highs-ipm"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("n6_degree2_quadratic_search.npz")
    )
    args = parser.parse_args()

    base.verify_ranking()
    group = base.variable_group()
    _, multipliers, _ = base.monomial_orbits(2, group)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(
        6, group
    )
    if len(multipliers) != 16 or len(degree_six_reps) != 5605:
        raise AssertionError("unexpected orbit count")
    target = base.target_coefficients(degree_six_reps)
    matrices = [
        multiplier_matrix(
            len(group), multiplier, degree_six_orbits, degree_six_sizes
        )
        for multiplier in multipliers
    ]
    print(
        f"multiplier-blocks={len(multipliers)} m2={len(base.M2)} "
        f"upper-coordinates={len(QUADRATIC_UPPER)}",
        flush=True,
    )

    ray_grams: list[list[np.ndarray]] = [[] for _ in multipliers]
    ray_columns: list[list[np.ndarray]] = [[] for _ in multipliers]
    fixed_columns: list[np.ndarray] = []
    fixed_grams: list[np.ndarray] = []
    fixed_gram_blocks: list[int] = []
    degree4_matrices: list[sparse.csr_array] = []
    degree4_multipliers = np.empty((0, 4), dtype=np.int16)
    if args.degree4_seed:
        seed = np.load(args.degree4_seed, allow_pickle=False)
        degree4_multipliers = np.asarray(seed["multipliers"], dtype=np.int16)
        coordinates = np.asarray(seed["gram_coordinates"], dtype=float)
        if degree4_multipliers.shape != (302, 4) or coordinates.shape != (
            302,
            len(linear.LINEAR_UPPER),
        ):
            raise ValueError("unexpected degree-four seed shape")
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
        for block, (multiplier, gram) in enumerate(
            zip(degree4_multipliers, coordinates[:, full_index])
        ):
            eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
            projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
            trace = float(np.trace(projected))
            if trace <= args.prune_tolerance:
                fixed_grams.append(np.zeros_like(projected))
                fixed_gram_blocks.append(block)
                fixed_columns.append(np.zeros(len(target)))
                continue
            normalized = projected / trace
            matrix = degree4.multiplier_matrix(
                len(group), multiplier, degree_six_orbits, degree_six_sizes
            )
            degree4_matrices.append(matrix)
            fixed_grams.append(normalized)
            fixed_gram_blocks.append(block)
            fixed_columns.append(
                np.asarray(matrix @ degree4.pair_vector(normalized)).ravel()
            )
        print(f"fixed degree-four rays={len(fixed_columns)}", flush=True)
        if len(degree4_matrices) != 302:
            degree4_matrices = [
                degree4.multiplier_matrix(
                    len(group), multiplier, degree_six_orbits, degree_six_sizes
                )
                for multiplier in degree4_multipliers
            ]
    if args.seed_negative_rows:
        negative_rows = np.flatnonzero(target < 0)
        seeded = 0
        for count, row in enumerate(negative_rows, 1):
            for block, matrix in enumerate(matrices):
                slack = sparse_slack_row(matrix, int(row))
                if slack.nnz == 0:
                    continue
                eigenvalues, eigenvectors = sparse.linalg.eigsh(
                    slack, k=1, which="SA", tol=1e-9
                )
                if eigenvalues[0] >= -args.reduced_cost_tolerance:
                    continue
                vector = eigenvectors[:, 0]
                ray_grams[block].append(vector)
                ray_columns[block].append(
                    np.asarray(matrix @ ray_upper(vector)).ravel()
                )
                seeded += 1
            if count % 10 == 0 or count == len(negative_rows):
                print(
                    f"seeded negative rows {count}/{len(negative_rows)} rays={seeded}",
                    flush=True,
                )
    accepted_grams: list[np.ndarray] | None = None
    accepted_fixed_grams: list[np.ndarray] | None = None
    accepted_residual: np.ndarray | None = None
    final_margin: float | None = None
    for iteration in range(args.iterations):
        block_sizes = [len(values) for values in ray_columns]
        all_columns = fixed_columns + [
            column for values in ray_columns for column in values
        ]
        if all_columns:
            produced = np.column_stack(all_columns)
            inequalities = sparse.csr_array(
                np.column_stack((produced, np.ones(len(target))))
            )
            number_of_rays = produced.shape[1]
        else:
            produced = None
            inequalities = sparse.csr_array(np.ones((len(target), 1)))
            number_of_rays = 0
        objective = np.zeros(number_of_rays + 1)
        objective[-1] = -1
        result = optimize.linprog(
            objective,
            A_ub=inequalities,
            b_ub=target,
            bounds=[(0.0, None)] * number_of_rays + [(None, None)],
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
        dynamic_weights = weights[len(fixed_columns) :]
        residual = target.copy() if produced is None else target - produced @ weights
        dual = -np.asarray(result.ineqlin.marginals)
        if dual.min() < -2e-7 or abs(dual.sum() - 1) > 2e-6:
            raise AssertionError("unexpected LP dual")

        candidates = []
        reduced_minimum = float("inf")
        negative_blocks = 0
        for matrix in matrices:
            functional = np.asarray(matrix.T @ dual).ravel()
            eigenvalues, eigenvectors = np.linalg.eigh(slack_matrix(functional))
            block_candidates = []
            for index in range(min(args.rays_per_block, len(base.M2))):
                vector = eigenvectors[:, index]
                column = np.asarray(matrix @ ray_upper(vector)).ravel()
                reduced = float(dual @ column)
                if abs(reduced - eigenvalues[index]) > 5e-7 * (
                    1 + abs(eigenvalues[index])
                ):
                    raise AssertionError("reduced-cost mismatch")
                block_candidates.append((reduced, vector, column))
            if block_candidates[0][0] < -args.reduced_cost_tolerance:
                negative_blocks += 1
            reduced_minimum = min(reduced_minimum, block_candidates[0][0])
            candidates.append(block_candidates)
        degree4_candidates = []
        degree4_negative_blocks = 0
        degree4_reduced_minimum = float("inf")
        if args.price_degree4:
            if len(degree4_matrices) != 302:
                raise ValueError("--price-degree4 requires --degree4-seed")
            for block, matrix in enumerate(degree4_matrices):
                functional = np.asarray(matrix.T @ dual).ravel()
                eigenvalues, eigenvectors = np.linalg.eigh(
                    degree4.slack_matrix(functional)
                )
                block_candidates = []
                for index in range(
                    min(args.degree4_rays_per_block, base.NV)
                ):
                    vector = eigenvectors[:, index]
                    gram = np.multiply.outer(vector, vector)
                    column_value = np.asarray(
                        matrix @ degree4.pair_vector(gram)
                    ).ravel()
                    reduced = float(dual @ column_value)
                    block_candidates.append((reduced, block, gram, column_value))
                if block_candidates[0][0] < -args.reduced_cost_tolerance:
                    degree4_negative_blocks += 1
                degree4_reduced_minimum = min(
                    degree4_reduced_minimum, block_candidates[0][0]
                )
                degree4_candidates.extend(block_candidates)
        print(
            f"iteration={iteration} rays={number_of_rays} "
            f"active={int(np.count_nonzero(weights > 1e-11))} "
            f"margin={float(result.x[-1]):.12g} residual={float(residual.min()):.12g} "
            f"negative-blocks={negative_blocks} reduced-min={reduced_minimum:.12g}",
            flush=True,
        )
        if args.price_degree4:
            print(
                f"  degree4-negative-blocks={degree4_negative_blocks} "
                f"degree4-reduced-min={degree4_reduced_minimum:.12g}",
                flush=True,
            )
        if residual.min() > 1e-7:
            accepted_grams = []
            accepted_fixed_grams = [
                weight * gram
                for weight, gram in zip(weights[: len(fixed_columns)], fixed_grams)
            ]
            cursor = 0
            for block, size in enumerate(block_sizes):
                block_weights = dynamic_weights[cursor : cursor + size]
                accepted_grams.append(
                    sum(
                        (
                            weight * ray_gram(ray)
                            for weight, ray in zip(block_weights, ray_grams[block])
                        ),
                        np.zeros((len(base.M2), len(base.M2))),
                    )
                )
                cursor += size
            accepted_residual = residual
            final_margin = float(result.x[-1])
            break

        cursor = 0
        for block, size in enumerate(block_sizes):
            block_weights = dynamic_weights[cursor : cursor + size]
            aggregate = sum(
                (
                    weight * ray_gram(ray)
                    for weight, ray in zip(block_weights, ray_grams[block])
                ),
                np.zeros((len(base.M2), len(base.M2))),
            )
            trace = float(np.trace(aggregate))
            retained = []
            if trace > args.prune_tolerance:
                retained.append(aggregate / trace)
            if args.history_per_block:
                retained.extend(ray_grams[block][-args.history_per_block :])
            ray_grams[block] = retained
            ray_columns[block] = [
                np.asarray(matrices[block] @ ray_upper(ray)).ravel()
                for ray in retained
            ]
            cursor += size
        for block, block_candidates in enumerate(candidates):
            for reduced, ray, column in block_candidates:
                if reduced < -args.reduced_cost_tolerance:
                    ray_grams[block].append(ray)
                    ray_columns[block].append(column)
        for reduced, block, gram, column_value in degree4_candidates:
            if reduced < -args.reduced_cost_tolerance:
                fixed_grams.append(gram)
                fixed_gram_blocks.append(block)
                fixed_columns.append(column_value)
        if (
            reduced_minimum >= -args.reduced_cost_tolerance
            and degree4_reduced_minimum >= -args.reduced_cost_tolerance
        ):
            raise SystemExit("full SDP dual is feasible before a positive margin")

    if (
        accepted_grams is None
        or accepted_fixed_grams is None
        or accepted_residual is None
        or final_margin is None
    ):
        raise SystemExit("no positive certificate margin within the iteration limit")

    factor_blocks = []
    factors = []
    for block, gram in enumerate(accepted_grams):
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        for value, vector in zip(eigenvalues, eigenvectors.T):
            if value <= 1e-12:
                continue
            factor_blocks.append(block)
            factors.append(np.sqrt(value) * vector)
    degree4_factor_blocks = []
    degree4_factors = []
    for block, gram in zip(fixed_gram_blocks, accepted_fixed_grams):
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        for value, vector in zip(eigenvalues, eigenvectors.T):
            if value <= 1e-12:
                continue
            degree4_factor_blocks.append(block)
            degree4_factors.append(np.sqrt(value) * vector)
    print("ACCEPTED numerical certificate", flush=True)
    print("minimum residual", float(accepted_residual.min()), flush=True)
    print("nonzero factors", len(factors), flush=True)
    print("nonzero degree-four factors", len(degree4_factors), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        allowed=np.asarray(base.ALLOWED, dtype=np.int8),
        group=group,
        multipliers=multipliers,
        m2=base.M2,
        degree_six_representatives=degree_six_reps,
        degree_six_sizes=degree_six_sizes,
        target=target,
        residual=accepted_residual,
        margin=np.asarray(final_margin),
        factor_blocks=np.asarray(factor_blocks, dtype=np.int16),
        factors=np.asarray(factors, dtype=float),
        degree4_multipliers=degree4_multipliers,
        degree4_factor_blocks=np.asarray(degree4_factor_blocks, dtype=np.int16),
        degree4_factors=np.asarray(degree4_factors, dtype=float),
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
