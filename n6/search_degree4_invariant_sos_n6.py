#!/usr/bin/env python3
"""Stabilizer-reduced SDP for the n=6 degree-four-multiplier certificate."""
from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import sparse

import search_degree4_column_n6 as column
import search_pair_sos_n6 as pair_search
import search_sos_n6 as base


def upper_orbit_map(q_orbit: np.ndarray) -> sparse.csr_array:
    rows = np.arange(len(pair_search.LINEAR_UPPER), dtype=np.int32)
    columns = np.asarray(
        [q_orbit[int(i), int(j)] for i, j in pair_search.LINEAR_UPPER],
        dtype=np.int32,
    )
    return sparse.coo_array(
        (np.ones(len(rows), dtype=np.int64), (rows, columns)),
        shape=(len(rows), int(q_orbit.max()) + 1),
    ).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--rho-x", type=float, default=1e-6)
    parser.add_argument("--no-adaptive-scale", action="store_true")
    parser.add_argument("--minimize-trace", action="store_true")
    parser.add_argument("--isotypic", action="store_true")
    parser.add_argument("--initial-search-result", type=Path)
    parser.add_argument("--warmup-eps", type=float)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument(
        "--gram-scale",
        type=float,
        default=1.0,
        help="represent each Gram matrix multiplied by this positive scale",
    )
    parser.add_argument("--fixed-margin", type=float)
    parser.add_argument(
        "--output", type=Path, default=Path("n6_degree4_invariant_search.npz")
    )
    args = parser.parse_args()
    if args.fixed_margin is not None and args.fixed_margin <= 0:
        raise ValueError("--fixed-margin must be positive")
    if args.gram_scale <= 0:
        raise ValueError("--gram-scale must be positive")

    base.verify_ranking()
    group = base.variable_group()
    _, multipliers, _ = base.monomial_orbits(4, group)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(
        6, group
    )
    if len(multipliers) != 302 or len(degree_six_reps) != 5605:
        raise AssertionError("unexpected orbit count")
    target = base.target_coefficients(degree_six_reps)

    values_list = []
    q_orbits = []
    coefficient_matrices = []
    constraints = []
    traces = []
    stabilizer_sizes = []
    psd_block_sizes = []
    for multiplier in multipliers:
        images = np.sort(group[:, multiplier], axis=1)
        stabilizer = group[np.all(images == multiplier, axis=1)]
        q_orbit = base.invariant_gram_orbits(stabilizer)
        values = cp.Variable(int(q_orbit.max()) + 1)
        gram = cp.reshape(values[q_orbit.ravel()], q_orbit.shape, order="C")
        if args.isotypic:
            bases = base.isotypic_bases(
                q_orbit, int(q_orbit.max()) + 1, verbose=False
            )
            constraints.extend(basis.T @ gram @ basis >> 0 for basis in bases)
            psd_block_sizes.extend(basis.shape[1] for basis in bases)
        else:
            constraints.append(gram >> 0)
            psd_block_sizes.append(base.NV)
        traces.append(cp.trace(gram))
        original = column.multiplier_matrix(
            len(group), multiplier, degree_six_orbits, degree_six_sizes
        )
        coefficient_matrices.append(original @ upper_orbit_map(q_orbit))
        values_list.append(values)
        q_orbits.append(q_orbit)
        stabilizer_sizes.append(len(stabilizer))

    coefficient = sparse.hstack(coefficient_matrices, format="csr")
    scaled_coefficient = coefficient / args.gram_scale
    flat_values = cp.hstack(values_list)
    if args.fixed_margin is None:
        margin = cp.Variable()
        constraints.append(scaled_coefficient @ flat_values + margin <= target)
        problem = cp.Problem(cp.Maximize(margin), constraints)
    else:
        margin = cp.Constant(args.fixed_margin)
        constraints.append(scaled_coefficient @ flat_values + margin <= target)
        objective = cp.sum(cp.hstack(traces)) if args.minimize_trace else 0
        problem = cp.Problem(cp.Minimize(objective), constraints)
    print(
        f"blocks={len(multipliers)} invariant-coordinates={coefficient.shape[1]} "
        f"coefficient-shape={coefficient.shape} nnz={coefficient.nnz} "
        f"stabilizers={min(stabilizer_sizes)}..{max(stabilizer_sizes)} "
        f"psd-blocks={len(psd_block_sizes)} max-psd-block={max(psd_block_sizes)}",
        flush=True,
    )
    if args.initial_search_result:
        initial = np.load(args.initial_search_result, allow_pickle=False)
        initial_coordinates = np.asarray(initial["gram_coordinates"], dtype=float)
        if initial_coordinates.shape != (
            len(multipliers),
            len(pair_search.LINEAR_UPPER),
        ):
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
        for block, (values, q_orbit) in enumerate(zip(values_list, q_orbits)):
            gram = initial_coordinates[block, full_index] * args.gram_scale
            values.value = np.asarray(
                [np.mean(gram[q_orbit == orbit]) for orbit in range(values.size)]
            )
        if args.fixed_margin is None:
            margin.value = float(initial["margin"])
        print("loaded stabilizer-averaged warm start", flush=True)
    solver_options = dict(
        solver=cp.SCS,
        acceleration_lookback=20,
        use_indirect=not args.direct,
        scale=args.scale,
        rho_x=args.rho_x,
        adaptive_scale=not args.no_adaptive_scale,
    )
    if args.warmup_eps:
        problem.solve(
            **solver_options,
            verbose=False,
            eps=args.warmup_eps,
            max_iters=args.warmup_iters,
            warm_start=True,
        )
        print(
            f"warmup status={problem.status} margin={float(margin.value)}",
            flush=True,
        )
    result = problem.solve(
        **solver_options,
        verbose=True,
        eps=args.eps,
        max_iters=args.max_iters,
        warm_start=True,
    )
    print("status", problem.status, flush=True)
    print("objective", result, flush=True)
    if any(values.value is None for values in values_list):
        raise SystemExit("solver returned no Gram coordinates")

    coordinate_values = np.asarray(
        [
            [
                float(values.value[q_orbit[int(i), int(j)]]) / args.gram_scale
                for i, j in pair_search.LINEAR_UPPER
            ]
            for values, q_orbit in zip(values_list, q_orbits)
        ]
    )
    residual = target - scaled_coefficient @ np.concatenate(
        [np.asarray(values.value).ravel() for values in values_list]
    )
    upper_index = {
        (int(i), int(j)): index
        for index, (i, j) in enumerate(pair_search.LINEAR_UPPER)
    }
    full_index = np.empty((base.NV, base.NV), dtype=np.int32)
    for i in range(base.NV):
        for j in range(base.NV):
            full_index[i, j] = upper_index[tuple(sorted((i, j)))]
    grams = coordinate_values[:, full_index]
    minimum_eigenvalues = np.linalg.eigvalsh(grams)[:, 0]
    print("minimum residual", float(residual.min()), flush=True)
    print("minimum eigenvalue", float(minimum_eigenvalues.min()), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        allowed=np.asarray(base.ALLOWED, dtype=np.int8),
        group=group,
        multipliers=multipliers,
        degree_six_representatives=degree_six_reps,
        degree_six_sizes=degree_six_sizes,
        target=target,
        residual=residual,
        margin=np.asarray(float(margin.value)),
        minimum_eigenvalues=minimum_eigenvalues,
        gram_coordinates=coordinate_values,
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
