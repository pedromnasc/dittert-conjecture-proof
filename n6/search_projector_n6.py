#!/usr/bin/env python3
"""Inner PSD-cone search using invariant spectral projectors.

A generic symmetric matrix in a finite permutation commutant has invariant
eigenspaces.  Their orthogonal projectors are positive semidefinite and lie
in the same commutant.  Nonnegative combinations of projectors from several
generic commutant matrices therefore give a rigorous inner approximation to
the three Gram cones, while numerical discovery reduces to a nonnegative LP.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_sos_n6 as base


def spectral_projectors(
    q_orbit: np.ndarray, number_of_seeds: int, seed_offset: int
) -> np.ndarray:
    number_of_q_orbits = int(q_orbit.max()) + 1
    orbit_counts = np.bincount(q_orbit.ravel(), minlength=number_of_q_orbits)
    columns: list[np.ndarray] = []
    for seed in range(number_of_seeds):
        rng = np.random.default_rng(20260723 + seed_offset + seed)
        gram = rng.normal(size=number_of_q_orbits)[q_orbit]
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        start = 0
        seed_columns = 0
        for end in range(1, len(eigenvalues) + 1):
            separated = end == len(eigenvalues) or abs(eigenvalues[end] - eigenvalues[start]) > (
                1e-7 * (1 + abs(eigenvalues[start]))
            )
            if not separated:
                continue
            basis = eigenvectors[:, start:end]
            projector = basis @ basis.T
            q_values = np.bincount(
                q_orbit.ravel(),
                weights=projector.ravel(),
                minlength=number_of_q_orbits,
            ) / orbit_counts
            invariant_error = np.max(np.abs(projector - q_values[q_orbit]))
            if invariant_error > 2e-7:
                raise AssertionError(f"spectral projector invariance error {invariant_error}")
            q_values /= np.trace(projector)
            columns.append(q_values)
            seed_columns += 1
            start = end
        print(f"seed {seed + 1}/{number_of_seeds}: projectors={seed_columns}", flush=True)
    return np.asarray(columns, dtype=float).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--shift-padding", type=float, default=2e-9)
    parser.add_argument("--method", choices=("highs", "highs-ipm", "highs-ds"), default="highs-ipm")
    parser.add_argument("--output", type=Path, default=Path("n6_projector_search.npz"))
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
    coefficient_matrices = []
    projector_matrices = []
    produced_blocks = []
    for number, vertex in enumerate(vertices):
        stabilizer_actions = action2[group[:, vertex] == vertex]
        q_orbit = base.invariant_gram_orbits(stabilizer_actions)
        degree_five = base.vertex_degree_five_matrix(
            len(group), vertex, q_orbit, degree_five_orbits, degree_five_sizes
        )
        coefficient = (lift @ degree_five).tocsr()
        projectors = spectral_projectors(q_orbit, args.seeds, 10_000 * number)
        produced = coefficient @ projectors
        q_orbits.append(q_orbit)
        coefficient_matrices.append(coefficient)
        projector_matrices.append(projectors)
        produced_blocks.append(produced)
        print(
            f"cell={base.ALLOWED[vertex]} qvars={int(q_orbit.max()) + 1} "
            f"projector-rays={projectors.shape[1]}",
            flush=True,
        )

    produced_matrix = np.concatenate(produced_blocks, axis=1)
    number_of_rays = produced_matrix.shape[1]
    inequalities = sparse.csr_array(
        np.concatenate((produced_matrix, np.ones((len(target), 1))), axis=1)
    )
    objective = np.zeros(number_of_rays + 1)
    objective[-1] = -1.0
    bounds = [(0.0, None)] * number_of_rays + [(None, None)]
    print(
        f"LP variables={number_of_rays + 1} inequalities={len(target)} "
        f"dense-coefficients={produced_matrix.size}",
        flush=True,
    )
    result = optimize.linprog(
        objective,
        A_ub=inequalities,
        b_ub=target,
        bounds=bounds,
        method=args.method,
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    print(result.message, flush=True)
    if not result.success:
        raise SystemExit(1)
    weights = result.x[:-1]
    q_values = []
    cursor = 0
    for projectors in projector_matrices:
        block_weights = weights[cursor : cursor + projectors.shape[1]]
        q_values.append(projectors @ block_weights)
        cursor += projectors.shape[1]
    if cursor != len(weights):
        raise AssertionError("projector weight partition failed")
    residual = target - sum(
        (matrix @ values for matrix, values in zip(coefficient_matrices, q_values)),
        np.zeros(len(target)),
    )
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
    shifted_eigenvalues = [
        float(np.linalg.eigvalsh(values[q_orbit]).min())
        for values, q_orbit in zip(shifted_values, q_orbits)
    ]
    print("LP margin", float(result.x[-1]), flush=True)
    print("minimum residual", float(residual.min()), flush=True)
    print("minimum eigenvalues", minimum_eigenvalues, flush=True)
    print("diagonal shifts", shifts, flush=True)
    print("shifted minimum residual", float(shifted_residual.min()), flush=True)
    print("shifted minimum eigenvalues", shifted_eigenvalues, flush=True)
    if shifted_residual.min() <= 0:
        raise SystemExit("projector cone did not produce a shift-safe certificate")

    payload: dict[str, np.ndarray] = {
        "allowed": np.asarray(base.ALLOWED, dtype=np.int8),
        "group": group,
        "vertices": np.asarray(vertices, dtype=np.int16),
        "degree_six_representatives": degree_six_reps,
        "degree_six_sizes": degree_six_sizes,
        "target": target,
        "residual": shifted_residual,
        "margin": np.asarray(float(result.x[-1])),
        "minimum_eigenvalues": np.asarray(shifted_eigenvalues),
        "projector_ray_count": np.asarray(number_of_rays),
    }
    for number, (q_orbit, values) in enumerate(zip(q_orbits, shifted_values)):
        payload[f"q_orbit_{number}"] = q_orbit
        payload[f"q_values_{number}"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print("ACCEPTED numerical certificate", flush=True)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
