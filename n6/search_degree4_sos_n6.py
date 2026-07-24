#!/usr/bin/env python3
"""Full SDP for all 302 degree-four monomial/linear-SOS blocks."""
from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import sparse

import search_degree4_column_n6 as column
import search_pair_sos_n6 as pair_search
import search_sos_n6 as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--rho-x", type=float, default=1e-6)
    parser.add_argument("--no-adaptive-scale", action="store_true")
    parser.add_argument(
        "--fixed-margin",
        type=float,
        help="solve a feasibility problem at this prescribed positive margin",
    )
    parser.add_argument("--output", type=Path, default=Path("n6_degree4_sos_search.npz"))
    args = parser.parse_args()

    base.verify_ranking()
    group = base.variable_group()
    degree_four_orbits, multipliers, degree_four_sizes = base.monomial_orbits(4, group)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(6, group)
    if len(multipliers) != 302 or len(degree_six_reps) != 5605:
        raise AssertionError("unexpected orbit count")
    target = base.target_coefficients(degree_six_reps)
    coefficient_matrices = [
        column.multiplier_matrix(
            len(group), multiplier, degree_six_orbits, degree_six_sizes
        )
        for multiplier in multipliers
    ]
    coefficient = sparse.hstack(coefficient_matrices, format="csr")

    upper_index = {
        (int(i), int(j)): index
        for index, (i, j) in enumerate(pair_search.LINEAR_UPPER)
    }
    full_index = np.empty((base.NV, base.NV), dtype=np.int32)
    for i in range(base.NV):
        for j in range(base.NV):
            full_index[i, j] = upper_index[tuple(sorted((i, j)))]

    values = cp.Variable((len(multipliers), len(pair_search.LINEAR_UPPER)))
    constraints = []
    for block in range(len(multipliers)):
        gram = cp.reshape(
            values[block, full_index.ravel()],
            (base.NV, base.NV),
            order="C",
        )
        constraints.append(gram >> 0)
    flat_values = cp.reshape(values, (values.size,), order="C")
    if args.fixed_margin is None:
        margin = cp.Variable()
        constraints.append(coefficient @ flat_values + margin <= target)
        problem = cp.Problem(cp.Maximize(margin), constraints)
    else:
        if args.fixed_margin <= 0:
            raise ValueError("--fixed-margin must be positive")
        margin = cp.Constant(args.fixed_margin)
        constraints.append(coefficient @ flat_values + margin <= target)
        problem = cp.Problem(cp.Minimize(0), constraints)
    print(
        f"blocks={len(multipliers)} gram-coordinates={values.size} "
        f"coefficient-shape={coefficient.shape} nnz={coefficient.nnz}",
        flush=True,
    )
    result = problem.solve(
        solver=cp.SCS,
        verbose=True,
        eps=args.eps,
        max_iters=args.max_iters,
        acceleration_lookback=20,
        use_indirect=not args.direct,
        scale=args.scale,
        rho_x=args.rho_x,
        adaptive_scale=not args.no_adaptive_scale,
    )
    print("status", problem.status, flush=True)
    print("objective", result, flush=True)
    if values.value is None:
        raise SystemExit("solver returned no Gram coordinates")
    coordinate_values = np.asarray(values.value)
    residual = target - coefficient @ coordinate_values.ravel()
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
