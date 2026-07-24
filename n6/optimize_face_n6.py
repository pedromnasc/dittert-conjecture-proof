#!/usr/bin/env python3
"""Numerically optimize the two-zero n=6 face to expose hard configurations."""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy import optimize

import search_sos_n6 as base


ROWS = np.arange(base.N)[:, None]
PERMUTATIONS = np.asarray(tuple(itertools.permutations(range(base.N))), dtype=np.int8)
PERMUTATION_ROWS = np.arange(base.N)[None, :]
OBJECTIVE_SCALE = 1e8


def matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    matrix = np.zeros((base.N, base.N), dtype=float)
    matrix[tuple(np.asarray(base.ALLOWED).T)] = vector
    return matrix


def value_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
    matrix = matrix_from_vector(vector)
    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)
    row_product = float(np.prod(row_sums))
    column_product = float(np.prod(column_sums))

    selected = matrix[PERMUTATION_ROWS, PERMUTATIONS]
    prefix = np.ones_like(selected)
    suffix = np.ones_like(selected)
    prefix[:, 1:] = np.cumprod(selected[:, :-1], axis=1)
    suffix[:, :-1] = np.cumprod(selected[:, :0:-1], axis=1)[:, ::-1]
    cofactors = prefix * suffix
    permanent = float(np.prod(selected, axis=1).sum())
    permanent_gradient = np.zeros_like(matrix)
    for row in range(base.N):
        np.add.at(
            permanent_gradient[row], PERMUTATIONS[:, row], cofactors[:, row]
        )

    row_gradient = np.prod(
        np.where(np.eye(base.N, dtype=bool), 1.0, row_sums), axis=1
    )
    column_gradient = np.prod(
        np.where(np.eye(base.N, dtype=bool), 1.0, column_sums), axis=1
    )
    gradient_matrix = (
        row_gradient[:, None] + column_gradient[None, :] - permanent_gradient
    )
    gradient = gradient_matrix[tuple(np.asarray(base.ALLOWED).T)]
    return row_product + column_product - permanent, gradient


def symmetric_two_zero_start() -> np.ndarray:
    scalar = optimize.minimize_scalar(
        lambda x_value: (
            24
            * x_value**2
            * (
                x_value**2 * (-1 + 8 * x_value) ** 2
                + 8
                * x_value
                * (-1 + 8 * x_value)
                * ((1 - 4 * x_value) / 2) ** 2
                + 12 * ((1 - 4 * x_value) / 2) ** 4
            )
        ),
        bounds=(1 / 8, 1 / 4),
        method="bounded",
        options={"xatol": 1e-15},
    ).x
    top = -1 + 8 * scalar
    cross = (1 - 4 * scalar) / 2
    matrix = np.full((base.N, base.N), scalar)
    matrix[:2, 2:] = cross
    matrix[2:, :2] = cross
    matrix[0, 0] = 0
    matrix[1, 1] = 0
    matrix[0, 1] = top
    matrix[1, 0] = top
    return matrix[tuple(np.asarray(base.ALLOWED).T)] / base.N


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-starts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--maxiter", type=int, default=3000)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    starts = [
        symmetric_two_zero_start(),
        np.full(base.NV, 1 / base.NV),
    ]
    starts.extend(
        rng.dirichlet(np.full(base.NV, concentration))
        for concentration in (0.2, 0.5, 1.0, 2.0, 10.0)
        for _ in range((args.random_starts + 4) // 5)
    )

    results = []
    for index, start in enumerate(starts[: args.random_starts + 2]):
        result = optimize.minimize(
            lambda vector: -OBJECTIVE_SCALE * value_gradient(vector)[0],
            start,
            jac=lambda vector: -OBJECTIVE_SCALE * value_gradient(vector)[1],
            method="SLSQP",
            bounds=optimize.Bounds(np.zeros(base.NV), np.ones(base.NV)),
            constraints={
                "type": "eq",
                "fun": lambda vector: vector.sum() - 1,
                "jac": lambda vector: np.ones(base.NV),
            },
            options={"ftol": 1e-14, "maxiter": args.maxiter, "disp": False},
        )
        value, gradient = value_gradient(result.x)
        results.append((value, result.x, result.success, result.message))
        matrix = matrix_from_vector(result.x)
        print(
            f"start={index} success={result.success} iterations={result.nit} "
            f"F={value:.16g} gap={base.TARGET_CONSTANT - value:.16g} "
            f"positive={np.count_nonzero(result.x > 1e-9)} "
            f"gradient-spread={np.ptp(gradient[result.x > 1e-9]):.5g}",
            flush=True,
        )
        if index < 2 or value == max(item[0] for item in results):
            print("row-sums", np.array2string(matrix.sum(axis=1), precision=10))
            print("col-sums", np.array2string(matrix.sum(axis=0), precision=10))
            print(np.array2string(matrix, precision=8, suppress_small=True))

    value, vector, success, message = max(results, key=lambda item: item[0])
    print(
        f"best success={success} F={value:.17g} "
        f"target={base.TARGET_CONSTANT:.17g} "
        f"gap={base.TARGET_CONSTANT - value:.17g} message={message}"
    )
    print(np.array2string(matrix_from_vector(vector), precision=12, suppress_small=True))


if __name__ == "__main__":
    main()
