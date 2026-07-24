#!/usr/bin/env python3
"""Test first-order KKT inequalities against the degree-four SOS seed.

At a maximizer of F on S=1, q_v = 6F-S*partial_v F is nonnegative for
every coordinate and vanishes on the positive support.  We add the three
symmetry sums of q_v as nonnegative certificate terms.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize

import search_degree4_column_n6 as degree4
import search_kkt_seed_n6 as kkt
import search_pair_sos_n6 as linear
import search_sos_n6 as base


def kkt_inequality_columns(
    representatives: np.ndarray, orbit_ids: np.ndarray
) -> np.ndarray:
    orbit_count = int(orbit_ids.max()) + 1
    orbit_sizes = np.bincount(orbit_ids, minlength=orbit_count)
    weights = np.eye(orbit_count, dtype=np.int64)[orbit_ids]
    columns = np.zeros((len(representatives), orbit_count), dtype=np.int64)
    for row, monomial_array in enumerate(representatives):
        monomial = [int(variable) for variable in monomial_array]
        edge = kkt.is_hyperedge(monomial)
        if edge:
            columns[row] += 6 * orbit_sizes
        for extra in set(monomial):
            derivative_base = monomial.copy()
            derivative_base.remove(extra)
            used = set(derivative_base)
            for differentiated in range(base.NV):
                if differentiated in used:
                    continue
                if kkt.is_hyperedge(derivative_base + [differentiated]):
                    columns[row] -= weights[differentiated]
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("degree4_seed", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("n6_kkt_inequality_search.npz")
    )
    args = parser.parse_args()

    group = base.variable_group()
    degree_six_orbits, representatives, sizes = base.monomial_orbits(6, group)
    target = base.target_coefficients(representatives)
    kkt_columns = kkt_inequality_columns(
        representatives, kkt.variable_orbit_ids(group)
    )
    print(
        f"degree-six-orbits={len(target)} kkt-columns={kkt_columns.shape[1]}",
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
            len(group), multiplier, degree_six_orbits, sizes
        )
        fixed_grams.append(normalized)
        fixed_columns.append(
            np.asarray(matrix @ degree4.pair_vector(normalized)).ravel()
        )

    coefficient = np.column_stack((*fixed_columns, kkt_columns))
    inequalities = np.column_stack((coefficient, np.ones(len(target))))
    objective = np.zeros(inequalities.shape[1])
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
    print(f"margin={float(result.x[-1]):.15g}")
    print(f"residual-min={float(residual.min()):.15g}")
    print("kkt-weights", weights[-kkt_columns.shape[1] :])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target=target,
        residual=residual,
        margin=np.asarray(float(result.x[-1])),
        multipliers=multipliers,
        fixed_grams=np.asarray(fixed_grams),
        fixed_weights=weights[: len(fixed_columns)],
        kkt_columns=kkt_columns,
        kkt_weights=weights[-kkt_columns.shape[1] :],
    )
    print(args.output)


if __name__ == "__main__":
    main()
