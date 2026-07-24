#!/usr/bin/env python3
"""Test degree-seven KKT identities against the projected degree-four seed.

At a simplex maximizer of F, complementarity gives

    x_v (S partial_v F - 6 F) = 0.

The three symmetry sums of these identities add to zero, leaving two free
degree-seven coefficient directions.  A positive residual here would be a
numerical KKT certificate, not yet an exact proof.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize

import search_degree4_column_n6 as degree4
import search_degree7_column_n6 as degree7
import search_pair_sos_n6 as linear
import search_sos_n6 as base


def variable_orbit_ids(group: np.ndarray) -> np.ndarray:
    result = np.full(base.NV, -1, dtype=np.int8)
    orbit = 0
    for variable in range(base.NV):
        if result[variable] >= 0:
            continue
        result[np.unique(group[:, variable])] = orbit
        orbit += 1
    if orbit != 3 or sorted(np.bincount(result)) != [2, 16, 16]:
        raise AssertionError("unexpected variable orbit structure")
    return result


def is_hyperedge(variables: list[int]) -> bool:
    if len(set(variables)) != base.N:
        return False
    cells = [base.ALLOWED[variable] for variable in variables]
    return len({row for row, _ in cells}) == base.N or len(
        {column for _, column in cells}
    ) == base.N


def kkt_columns(representatives: np.ndarray, orbit_ids: np.ndarray) -> np.ndarray:
    columns = np.zeros((len(representatives), 3), dtype=np.int64)
    weights = np.eye(3, dtype=np.int64)[orbit_ids]
    for row, monomial_array in enumerate(representatives):
        monomial = [int(variable) for variable in monomial_array]
        for extra in set(monomial):
            edge = monomial.copy()
            edge.remove(extra)
            if not is_hyperedge(edge):
                continue
            columns[row] += weights[edge].sum(axis=0) - 6 * weights[extra]
    if np.any(columns.sum(axis=1)):
        raise AssertionError("Euler sum of KKT columns did not vanish")
    return columns[:, :2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("degree4_seed", type=Path)
    parser.add_argument("--output", type=Path, default=Path("n6_kkt_seed_search.npz"))
    args = parser.parse_args()

    group = base.variable_group()
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(
        6, group
    )
    degree_seven_orbits, degree_seven_reps, _ = base.monomial_orbits(7, group)
    lift = degree7.multiplication_by_s(degree_six_orbits, degree_seven_reps)
    target = np.asarray(lift @ base.target_coefficients(degree_six_reps)).ravel()
    kkt = kkt_columns(degree_seven_reps, variable_orbit_ids(group))
    print(
        f"degree-seven-orbits={len(target)} kkt-ranks={np.linalg.matrix_rank(kkt)}",
        flush=True,
    )

    seed = np.load(args.degree4_seed, allow_pickle=False)
    multipliers = np.asarray(seed["multipliers"], dtype=np.int16)
    coordinates = np.asarray(seed["gram_coordinates"], dtype=float)
    lookup = {
        (int(i), int(j)): index for index, (i, j) in enumerate(linear.LINEAR_UPPER)
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
        normalized = projected / np.trace(projected)
        matrix = degree4.multiplier_matrix(
            len(group), multiplier, degree_six_orbits, degree_six_sizes
        )
        fixed_grams.append(normalized)
        fixed_columns.append(
            np.asarray(lift @ (matrix @ degree4.pair_vector(normalized))).ravel()
        )

    coefficient = np.column_stack((*fixed_columns, kkt))
    inequalities = np.column_stack((coefficient, np.ones(len(target))))
    objective = np.zeros(inequalities.shape[1])
    objective[-1] = -1
    result = optimize.linprog(
        objective,
        A_ub=inequalities,
        b_ub=target,
        bounds=[(0, None)] * len(fixed_columns)
        + [(None, None)] * (kkt.shape[1] + 1),
        method="highs-ipm",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
        },
    )
    if not result.success:
        raise SystemExit(result.message)
    weights = np.maximum(result.x[: len(fixed_columns)], 0)
    kkt_weights = result.x[len(fixed_columns) : -1]
    residual = target - coefficient @ result.x[:-1]
    dual = -np.asarray(result.ineqlin.marginals)
    print(f"margin={float(result.x[-1]):.15g}")
    print(f"residual-min={float(residual.min()):.15g}")
    print("kkt-weights", kkt_weights)
    print(
        f"dual-min={float(dual.min()):.6g} dual-sum={float(dual.sum()):.15g}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target=target,
        residual=residual,
        margin=np.asarray(float(result.x[-1])),
        multipliers=multipliers,
        fixed_grams=np.asarray(fixed_grams),
        fixed_weights=weights,
        kkt_columns=kkt,
        kkt_weights=kkt_weights,
    )
    print(args.output)


if __name__ == "__main__":
    main()
