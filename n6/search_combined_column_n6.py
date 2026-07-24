#!/usr/bin/env python3
"""Dual-guided column generation for the combined n=6 SOS cone."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_pair_sos_n6 as pair_search
import search_sos_n6 as base
from search_column_generation_n6 import invariant_projector_values
from search_projector_n6 import spectral_projectors


def pair_vector(gram: np.ndarray) -> np.ndarray:
    return np.asarray(
        [gram[int(i), int(j)] for i, j in pair_search.LINEAR_UPPER], dtype=float
    )


def pair_slack(functional: np.ndarray) -> np.ndarray:
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
    parser.add_argument("--vertex-rays", type=int, default=8)
    parser.add_argument("--pair-rays", type=int, default=4)
    parser.add_argument("--shift-padding", type=float, default=2e-9)
    parser.add_argument("--reduced-cost-tolerance", type=float, default=1e-9)
    parser.add_argument("--method", choices=("highs", "highs-ipm", "highs-ds"), default="highs-ipm")
    parser.add_argument("--output", type=Path, default=Path("n6_combined_column_search.npz"))
    args = parser.parse_args()

    base.verify_ranking()
    group = base.variable_group()
    action2 = base.induced_m2_actions(group)
    vertices = base.vertex_representatives(group)
    _, pair_representatives, _ = base.monomial_orbits(2, group)
    degree_four_orbits, degree_four_reps, degree_four_sizes = base.monomial_orbits(4, group)
    degree_five_orbits, degree_five_reps, degree_five_sizes = base.monomial_orbits(5, group)
    degree_six_orbits, degree_six_reps, degree_six_sizes = base.monomial_orbits(6, group)
    if not (
        len(pair_representatives) == 16
        and len(degree_four_reps) == 302
        and len(degree_five_reps) == 1292
        and len(degree_six_reps) == 5605
    ):
        raise AssertionError("unexpected orbit count")
    lift_vertex = base.multiplication_by_s(
        degree_five_orbits, degree_six_reps, len(degree_five_reps)
    )
    lift_pair = pair_search.multiplication_by_s_squared(
        degree_four_orbits, degree_six_reps, len(degree_four_reps)
    )
    target = base.target_coefficients(degree_six_reps)

    vertex_q_orbits = []
    vertex_orbit_counts = []
    vertex_matrices = []
    vertex_rays: list[list[np.ndarray]] = []
    vertex_columns: list[list[np.ndarray]] = []
    for number, vertex in enumerate(vertices):
        stabilizer_actions = action2[group[:, vertex] == vertex]
        q_orbit = base.invariant_gram_orbits(stabilizer_actions)
        degree_five = base.vertex_degree_five_matrix(
            len(group), vertex, q_orbit, degree_five_orbits, degree_five_sizes
        )
        coefficient = (lift_vertex @ degree_five).tocsr()
        initial = spectral_projectors(q_orbit, 1, 10_000 * number)
        rays = [initial[:, column].copy() for column in range(initial.shape[1])]
        columns = [np.asarray(coefficient @ ray).ravel() for ray in rays]
        vertex_q_orbits.append(q_orbit)
        vertex_orbit_counts.append(
            np.bincount(q_orbit.ravel(), minlength=int(q_orbit.max()) + 1)
        )
        vertex_matrices.append(coefficient)
        vertex_rays.append(rays)
        vertex_columns.append(columns)
        print(f"vertex {number + 1}/3 initial-rays={len(rays)}", flush=True)

    pair_matrices = []
    pair_rays: list[list[np.ndarray]] = []
    pair_columns: list[list[np.ndarray]] = []
    for number, pair in enumerate(pair_representatives):
        degree_four = pair_search.pair_degree_four_matrix(
            len(group), pair, degree_four_orbits, degree_four_sizes
        )
        coefficient = (lift_pair @ degree_four).tocsr()
        rng = np.random.default_rng(20260723 + number)
        orthogonal, _ = np.linalg.qr(rng.normal(size=(base.NV, base.NV)))
        rays = [
            np.multiply.outer(orthogonal[:, column], orthogonal[:, column])
            for column in range(base.NV)
        ]
        columns = [np.asarray(coefficient @ pair_vector(ray)).ravel() for ray in rays]
        pair_matrices.append(coefficient)
        pair_rays.append(rays)
        pair_columns.append(columns)
    print(f"pair blocks=16 initial-rays={sum(map(len, pair_rays))}", flush=True)

    accepted_vertex: list[np.ndarray] | None = None
    accepted_pair: list[np.ndarray] | None = None
    accepted_residual: np.ndarray | None = None
    final_margin: float | None = None
    for iteration in range(args.iterations):
        vertex_sizes = [len(columns) for columns in vertex_columns]
        pair_sizes = [len(columns) for columns in pair_columns]
        all_columns = [
            column
            for block_columns in vertex_columns + pair_columns
            for column in block_columns
        ]
        produced_matrix = np.column_stack(all_columns)
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
        cursor = 0
        vertex_values = []
        for rays, size in zip(vertex_rays, vertex_sizes):
            block_weights = weights[cursor : cursor + size]
            vertex_values.append(
                sum(
                    (weight * ray for weight, ray in zip(block_weights, rays)),
                    np.zeros_like(rays[0]),
                )
            )
            cursor += size
        pair_values = []
        for rays, size in zip(pair_rays, pair_sizes):
            block_weights = weights[cursor : cursor + size]
            pair_values.append(
                sum(
                    (weight * ray for weight, ray in zip(block_weights, rays)),
                    np.zeros_like(rays[0]),
                )
            )
            cursor += size
        if cursor != len(weights):
            raise AssertionError("ray partition failed")

        shifted_vertex = []
        vertex_minimum = []
        for values, q_orbit in zip(vertex_values, vertex_q_orbits):
            minimum = float(np.linalg.eigvalsh(values[q_orbit]).min())
            shifted = values.copy()
            shift = max(0.0, -minimum) + args.shift_padding
            for qid in {int(q_orbit[i, i]) for i in range(len(base.M2))}:
                shifted[qid] += shift
            shifted_vertex.append(shifted)
            vertex_minimum.append(minimum)
        shifted_pair = []
        pair_minimum = []
        for gram in pair_values:
            minimum = float(np.linalg.eigvalsh(gram).min())
            shifted_pair.append(gram + (max(0.0, -minimum) + args.shift_padding) * np.eye(base.NV))
            pair_minimum.append(minimum)
        shifted_residual = target - sum(
            (
                matrix @ values
                for matrix, values in zip(vertex_matrices, shifted_vertex)
            ),
            np.zeros(len(target)),
        )
        shifted_residual -= sum(
            (
                matrix @ pair_vector(gram)
                for matrix, gram in zip(pair_matrices, shifted_pair)
            ),
            np.zeros(len(target)),
        )

        dual = -np.asarray(result.ineqlin.marginals)
        if dual.min() < -2e-7 or abs(dual.sum() - 1) > 2e-6:
            raise AssertionError(f"unexpected master dual min={dual.min()} sum={dual.sum()}")
        new_vertex: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
        vertex_reduced_min = []
        for q_orbit, counts, coefficient in zip(
            vertex_q_orbits, vertex_orbit_counts, vertex_matrices
        ):
            functional = np.asarray(coefficient.T @ dual).ravel()
            slack = (functional / counts)[q_orbit]
            eigenvalues, eigenvectors = np.linalg.eigh(slack)
            rays_to_add = []
            start = 0
            while start < len(eigenvalues) and len(rays_to_add) < args.vertex_rays:
                end = int(
                    np.searchsorted(
                        eigenvalues,
                        eigenvalues[start] + 1e-8 * (1 + abs(eigenvalues[start])),
                        side="right",
                    )
                )
                ray = invariant_projector_values(q_orbit, eigenvectors[:, start:end])
                column = np.asarray(coefficient @ ray).ravel()
                reduced = float(dual @ column)
                expected = float(np.mean(eigenvalues[start:end]))
                if abs(reduced - expected) > 5e-7 * (1 + abs(expected)):
                    raise AssertionError("vertex reduced-cost mismatch")
                rays_to_add.append((reduced, ray, column))
                start = end
            new_vertex.append(rays_to_add)
            vertex_reduced_min.append(rays_to_add[0][0])

        new_pair: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
        pair_reduced_min = []
        for coefficient in pair_matrices:
            functional = np.asarray(coefficient.T @ dual).ravel()
            eigenvalues, eigenvectors = np.linalg.eigh(pair_slack(functional))
            rays_to_add = []
            for index in range(min(args.pair_rays, base.NV)):
                vector = eigenvectors[:, index]
                ray = np.multiply.outer(vector, vector)
                column = np.asarray(coefficient @ pair_vector(ray)).ravel()
                reduced = float(dual @ column)
                if abs(reduced - eigenvalues[index]) > 5e-7 * (1 + abs(eigenvalues[index])):
                    raise AssertionError("pair reduced-cost mismatch")
                rays_to_add.append((reduced, ray, column))
            new_pair.append(rays_to_add)
            pair_reduced_min.append(rays_to_add[0][0])

        print(
            f"iteration={iteration} rays={number_of_rays} "
            f"margin={float(result.x[-1]):.12g} "
            f"shifted-residual={float(shifted_residual.min()):.12g} "
            f"vertex-reduced={vertex_reduced_min} "
            f"pair-reduced-min={min(pair_reduced_min):.12g}",
            flush=True,
        )
        if shifted_residual.min() > 0:
            accepted_vertex = shifted_vertex
            accepted_pair = shifted_pair
            accepted_residual = shifted_residual
            final_margin = float(result.x[-1])
            break
        added = 0
        for block, rays_to_add in enumerate(new_vertex):
            for reduced, ray, column in rays_to_add:
                if reduced < -args.reduced_cost_tolerance:
                    vertex_rays[block].append(ray)
                    vertex_columns[block].append(column)
                    added += 1
        for block, rays_to_add in enumerate(new_pair):
            for reduced, ray, column in rays_to_add:
                if reduced < -args.reduced_cost_tolerance:
                    pair_rays[block].append(ray)
                    pair_columns[block].append(column)
                    added += 1
        if not added:
            raise SystemExit("dual PSD feasible without a shift-safe certificate")

    if (
        accepted_vertex is None
        or accepted_pair is None
        or accepted_residual is None
        or final_margin is None
    ):
        raise SystemExit("no shift-safe certificate found within the iteration limit")
    print("ACCEPTED numerical certificate", flush=True)
    print("minimum residual", float(accepted_residual.min()), flush=True)

    payload: dict[str, np.ndarray] = {
        "allowed": np.asarray(base.ALLOWED, dtype=np.int8),
        "group": group,
        "vertices": np.asarray(vertices, dtype=np.int16),
        "pair_representatives": pair_representatives,
        "degree_six_representatives": degree_six_reps,
        "degree_six_sizes": degree_six_sizes,
        "target": target,
        "residual": accepted_residual,
        "margin": np.asarray(final_margin),
        "pair_gram_values": np.asarray(accepted_pair),
        "vertex_ray_counts": np.asarray([len(rays) for rays in vertex_rays]),
        "pair_ray_counts": np.asarray([len(rays) for rays in pair_rays]),
    }
    for number, (q_orbit, values) in enumerate(zip(vertex_q_orbits, accepted_vertex)):
        payload[f"q_orbit_{number}"] = q_orbit
        payload[f"q_values_{number}"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
