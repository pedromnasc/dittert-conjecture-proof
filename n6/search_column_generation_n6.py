#!/usr/bin/env python3
"""Dual-guided PSD column generation for the n=6 SOS search.

The master LP uses nonnegative combinations of invariant PSD projectors.  Its
dual coefficient weights define, for each vertex block, an invariant slack
matrix.  A projector onto the minimum eigenspace of that slack is the exact
most-negative reduced-cost PSD ray, so adding those three rays is genuine
column generation for the symmetry-reduced SDP.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_sos_n6 as base
from search_projector_n6 import spectral_projectors


def invariant_projector_values(q_orbit: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projector = basis @ basis.T
    number_of_q_orbits = int(q_orbit.max()) + 1
    counts = np.bincount(q_orbit.ravel(), minlength=number_of_q_orbits)
    values = np.bincount(
        q_orbit.ravel(), weights=projector.ravel(), minlength=number_of_q_orbits
    ) / counts
    error = np.max(np.abs(projector - values[q_orbit]))
    if error > 3e-7:
        raise AssertionError(f"projector invariance error {error}")
    values /= np.trace(projector)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--initial-seeds", type=int, default=1)
    parser.add_argument("--shift-padding", type=float, default=2e-9)
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    parser.add_argument("--rays-per-block", type=int, default=12)
    parser.add_argument("--method", choices=("highs", "highs-ipm", "highs-ds"), default="highs-ipm")
    parser.add_argument("--output", type=Path, default=Path("n6_column_generation_search.npz"))
    args = parser.parse_args()

    base.verify_ranking()
    group = base.variable_group()
    action2 = base.induced_m2_actions(group)
    vertices = base.vertex_representatives(group)
    degree_five_orbits, degree_five_reps, degree_five_sizes = base.monomial_orbits(5, group)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(6, group)
    if len(degree_five_reps) != 1292 or len(degree_six_reps) != 5605:
        raise AssertionError("unexpected coefficient-orbit counts")
    lift = base.multiplication_by_s(
        degree_five_orbits, degree_six_reps, len(degree_five_reps)
    )
    target = base.target_coefficients(degree_six_reps)

    q_orbits = []
    orbit_counts = []
    coefficient_matrices = []
    ray_values: list[list[np.ndarray]] = []
    ray_columns: list[list[np.ndarray]] = []
    for number, vertex in enumerate(vertices):
        stabilizer_actions = action2[group[:, vertex] == vertex]
        q_orbit = base.invariant_gram_orbits(stabilizer_actions)
        degree_five = base.vertex_degree_five_matrix(
            len(group), vertex, q_orbit, degree_five_orbits, degree_five_sizes
        )
        coefficient = (lift @ degree_five).tocsr()
        initial = spectral_projectors(q_orbit, args.initial_seeds, 10_000 * number)
        values_list = [initial[:, column].copy() for column in range(initial.shape[1])]
        columns_list = [np.asarray(coefficient @ values).ravel() for values in values_list]
        q_orbits.append(q_orbit)
        orbit_counts.append(
            np.bincount(q_orbit.ravel(), minlength=int(q_orbit.max()) + 1)
        )
        coefficient_matrices.append(coefficient)
        ray_values.append(values_list)
        ray_columns.append(columns_list)
        print(
            f"cell={base.ALLOWED[vertex]} initial-rays={len(values_list)}",
            flush=True,
        )

    accepted_values: list[np.ndarray] | None = None
    accepted_residual: np.ndarray | None = None
    accepted_eigenvalues: list[float] | None = None
    final_margin: float | None = None
    for iteration in range(args.iterations):
        block_sizes = [len(columns) for columns in ray_columns]
        produced_matrix = np.column_stack(
            [column for columns in ray_columns for column in columns]
        )
        number_of_rays = produced_matrix.shape[1]
        inequalities = sparse.csr_array(
            np.concatenate((produced_matrix, np.ones((len(target), 1))), axis=1)
        )
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
        weights = result.x[:-1]
        q_values = []
        cursor = 0
        for values_list, size in zip(ray_values, block_sizes):
            block_weights = weights[cursor : cursor + size]
            q_values.append(
                sum(
                    (weight * values for weight, values in zip(block_weights, values_list)),
                    np.zeros_like(values_list[0]),
                )
            )
            cursor += size
        if cursor != len(weights):
            raise AssertionError("ray weight partition failed")
        minimum_eigenvalues = [
            float(np.linalg.eigvalsh(values[q_orbit]).min())
            for values, q_orbit in zip(q_values, q_orbits)
        ]
        shifted_values = []
        shifts = []
        for values, q_orbit, minimum in zip(q_values, q_orbits, minimum_eigenvalues):
            shift = max(0.0, -minimum) + args.shift_padding
            shifted = values.copy()
            for qid in {int(q_orbit[i, i]) for i in range(len(base.M2))}:
                shifted[qid] += shift
            shifted_values.append(shifted)
            shifts.append(shift)
        shifted_residual = target - sum(
            (
                matrix @ shifted
                for matrix, shifted in zip(coefficient_matrices, shifted_values)
            ),
            np.zeros(len(target)),
        )
        dual = -np.asarray(result.ineqlin.marginals)
        if dual.min() < -2e-7 or abs(dual.sum() - 1) > 2e-6:
            raise AssertionError(
                f"unexpected master dual: min={dual.min()} sum={dual.sum()}"
            )
        reduced_costs: list[list[float]] = []
        new_rays: list[list[tuple[np.ndarray, np.ndarray]]] = []
        for block, (q_orbit, counts, coefficient) in enumerate(
            zip(q_orbits, orbit_counts, coefficient_matrices)
        ):
            functional = np.asarray(coefficient.T @ dual).ravel()
            slack = (functional / counts)[q_orbit]
            eigenvalues, eigenvectors = np.linalg.eigh(slack)
            block_costs: list[float] = []
            block_rays: list[tuple[np.ndarray, np.ndarray]] = []
            start = 0
            while start < len(eigenvalues) and len(block_rays) < args.rays_per_block:
                end = int(
                    np.searchsorted(
                        eigenvalues,
                        eigenvalues[start] + 1e-8 * (1 + abs(eigenvalues[start])),
                        side="right",
                    )
                )
                values = invariant_projector_values(q_orbit, eigenvectors[:, start:end])
                column = np.asarray(coefficient @ values).ravel()
                reduced = float(dual @ column)
                expected = float(np.mean(eigenvalues[start:end]))
                if abs(reduced - expected) > 5e-7 * (1 + abs(expected)):
                    raise AssertionError(
                        f"reduced-cost/eigenvalue mismatch {reduced} versus {expected}"
                    )
                block_costs.append(reduced)
                block_rays.append((values, column))
                start = end
            reduced_costs.append(block_costs)
            new_rays.append(block_rays)
        print(
            f"iteration={iteration} rays={number_of_rays} "
            f"margin={float(result.x[-1]):.12g} "
            f"shifted-residual={float(shifted_residual.min()):.12g} "
            f"gram-min={minimum_eigenvalues} "
            f"reduced-min={[costs[0] for costs in reduced_costs]}",
            flush=True,
        )
        if shifted_residual.min() > 0:
            accepted_values = shifted_values
            accepted_residual = shifted_residual
            accepted_eigenvalues = [
                float(np.linalg.eigvalsh(values[q_orbit]).min())
                for values, q_orbit in zip(accepted_values, q_orbits)
            ]
            final_margin = float(result.x[-1])
            break
        added = 0
        for block, (block_costs, block_rays) in enumerate(zip(reduced_costs, new_rays)):
            for reduced, (values, column) in zip(block_costs, block_rays):
                if reduced < -args.reduced_cost_tolerance:
                    ray_values[block].append(values)
                    ray_columns[block].append(column)
                    added += 1
        if not added:
            raise SystemExit(
                "column generation reached dual PSD feasibility without a shift-safe certificate"
            )

    if (
        accepted_values is None
        or accepted_residual is None
        or accepted_eigenvalues is None
        or final_margin is None
    ):
        raise SystemExit("no shift-safe certificate found within the iteration limit")
    print("ACCEPTED numerical certificate", flush=True)
    print("minimum residual", float(accepted_residual.min()), flush=True)
    print("minimum eigenvalues", accepted_eigenvalues, flush=True)

    payload: dict[str, np.ndarray] = {
        "allowed": np.asarray(base.ALLOWED, dtype=np.int8),
        "group": group,
        "vertices": np.asarray(vertices, dtype=np.int16),
        "degree_six_representatives": degree_six_reps,
        "degree_six_sizes": degree_six_sizes,
        "target": target,
        "residual": accepted_residual,
        "margin": np.asarray(final_margin),
        "minimum_eigenvalues": np.asarray(accepted_eigenvalues),
        "ray_counts": np.asarray([len(values) for values in ray_values], dtype=np.int32),
    }
    for number, (q_orbit, values) in enumerate(zip(q_orbits, accepted_values)):
        payload[f"q_orbit_{number}"] = q_orbit
        payload[f"q_values_{number}"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
