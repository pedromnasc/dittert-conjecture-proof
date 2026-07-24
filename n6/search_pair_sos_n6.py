#!/usr/bin/env python3
"""Numerical search for an n=6 pair-multiplier SOS certificate.

The certificate cone is

    G_6 = sum_p sum_{gamma in Gamma}
              x_{gamma u_p} x_{gamma v_p} S^2
              (gamma x)^T Q_p (gamma x) + R,

where p runs over the 16 orbits of unordered variable pairs, Q_p is a
34 by 34 positive semidefinite matrix, and R has positive coefficients.
This is discovery code; a successful result must be rounded and checked by
an exact integer verifier.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import sparse

import search_sos_n6 as base


LINEAR_UPPER = np.asarray(
    tuple(itertools.combinations_with_replacement(range(base.NV), 2)), dtype=np.int16
)


def multiplication_by_s_squared(
    degree_four_orbits: np.ndarray,
    degree_six_representatives: np.ndarray,
    number_of_degree_four_orbits: int,
) -> sparse.csr_array:
    rows: list[int] = []
    columns: list[int] = []
    data: list[int] = []
    for row, monomial in enumerate(degree_six_representatives):
        values, counts = np.unique(monomial, return_counts=True)
        for left_index, left in enumerate(values):
            if counts[left_index] >= 2:
                predecessor = list(int(value) for value in monomial)
                predecessor.remove(int(left))
                predecessor.remove(int(left))
                index = int(base.rank_multisets(np.asarray([predecessor]), base.NV)[0])
                rows.append(row)
                columns.append(int(degree_four_orbits[index]))
                data.append(1)
            for right in values[left_index + 1 :]:
                predecessor = list(int(value) for value in monomial)
                predecessor.remove(int(left))
                predecessor.remove(int(right))
                index = int(base.rank_multisets(np.asarray([predecessor]), base.NV)[0])
                rows.append(row)
                columns.append(int(degree_four_orbits[index]))
                data.append(2)
    return sparse.coo_array(
        (data, (rows, columns)),
        shape=(len(degree_six_representatives), number_of_degree_four_orbits),
        dtype=np.int64,
    ).tocsr()


def pair_degree_four_matrix(
    group_size: int,
    pair: np.ndarray,
    degree_four_orbits: np.ndarray,
    degree_four_sizes: np.ndarray,
) -> sparse.csr_array:
    bases = np.concatenate(
        (
            np.repeat(pair.reshape(1, 2), len(LINEAR_UPPER), axis=0),
            LINEAR_UPPER,
        ),
        axis=1,
    )
    bases.sort(axis=1)
    monomial_indices = base.rank_multisets(bases, base.NV)
    rows = degree_four_orbits[monomial_indices]
    columns = np.arange(len(LINEAR_UPPER), dtype=np.int32)
    gram_factor = np.where(
        LINEAR_UPPER[:, 0] == LINEAR_UPPER[:, 1], 1, 2
    ).astype(np.int64)
    divisors = degree_four_sizes[rows]
    numerators = gram_factor * group_size
    if np.any(numerators % divisors):
        raise AssertionError("degree-four group averaging is not integral")
    data = numerators // divisors
    return sparse.coo_array(
        (data, (rows, columns)),
        shape=(len(degree_four_sizes), len(LINEAR_UPPER)),
        dtype=np.int64,
    ).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("CLARABEL", "SCS"), default="CLARABEL")
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--max-iters", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("n6_pair_sos_search.npz"))
    args = parser.parse_args()

    base.verify_ranking()
    group = base.variable_group()
    degree_two_orbits, pair_representatives, degree_two_sizes = base.monomial_orbits(2, group)
    degree_four_orbits, degree_four_reps, degree_four_sizes = base.monomial_orbits(4, group)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(6, group)
    if len(pair_representatives) != 16:
        raise AssertionError("unexpected pair-orbit count")
    if len(degree_four_reps) != 302 or len(degree_six_reps) != 5605:
        raise AssertionError("unexpected coefficient-orbit count")
    lift = multiplication_by_s_squared(
        degree_four_orbits, degree_six_reps, len(degree_four_reps)
    )
    target = base.target_coefficients(degree_six_reps)

    grams = []
    vectors = []
    coefficient_matrices = []
    constraints = []
    for number, pair in enumerate(pair_representatives):
        gram = cp.Variable((base.NV, base.NV), symmetric=True)
        vector = cp.hstack([gram[int(i), int(j)] for i, j in LINEAR_UPPER])
        degree_four = pair_degree_four_matrix(
            len(group), pair, degree_four_orbits, degree_four_sizes
        )
        coefficient = (lift @ degree_four).tocsr()
        grams.append(gram)
        vectors.append(vector)
        coefficient_matrices.append(coefficient)
        constraints.append(gram >> 0)
        print(
            f"pair {number + 1}/16={tuple(int(value) for value in pair)} "
            f"cells={tuple(base.ALLOWED[int(value)] for value in pair)}",
            flush=True,
        )

    margin = cp.Variable()
    produced = sum(
        (matrix @ vector for matrix, vector in zip(coefficient_matrices, vectors)),
        cp.Constant(np.zeros(len(target))),
    )
    constraints.append(produced + margin <= target)
    problem = cp.Problem(cp.Maximize(margin), constraints)
    if args.solver == "CLARABEL":
        result = problem.solve(
            solver=cp.CLARABEL,
            verbose=True,
            tol_gap_abs=args.eps,
            tol_feas=args.eps,
            tol_gap_rel=args.eps,
            max_iter=args.max_iters,
        )
    else:
        result = problem.solve(
            solver=cp.SCS,
            verbose=True,
            eps=args.eps,
            max_iters=args.max_iters,
        )
    print("status", problem.status, flush=True)
    print("objective", result, flush=True)
    if any(gram.value is None for gram in grams):
        raise SystemExit("solver returned no pair Gram matrices")
    gram_values = [np.asarray(gram.value) for gram in grams]
    vector_values = [
        np.asarray([gram[int(i), int(j)] for i, j in LINEAR_UPPER])
        for gram in gram_values
    ]
    residual = target - sum(
        (
            matrix @ vector
            for matrix, vector in zip(coefficient_matrices, vector_values)
        ),
        np.zeros(len(target)),
    )
    minimum_eigenvalues = [float(np.linalg.eigvalsh(gram).min()) for gram in gram_values]
    print("minimum residual", float(residual.min()), flush=True)
    print("minimum eigenvalue", min(minimum_eigenvalues), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        allowed=np.asarray(base.ALLOWED, dtype=np.int8),
        group=group,
        pair_representatives=pair_representatives,
        degree_six_representatives=degree_six_reps,
        degree_six_sizes=degree_six_sizes,
        target=target,
        residual=residual,
        margin=np.asarray(float(margin.value)),
        minimum_eigenvalues=np.asarray(minimum_eigenvalues),
        pair_gram_values=np.asarray(gram_values),
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
