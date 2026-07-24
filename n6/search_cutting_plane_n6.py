#!/usr/bin/env python3
"""PSD cutting-plane search for the n=6 vertex SOS certificate.

The full SCS formulation requires repeated projections onto three 595 by 595
PSD cones.  This discovery program instead solves sparse LP outer
approximations.  After every LP solve, negative Gram eigenvectors supply new
linear inequalities v^T Q v >= 0.  A candidate is accepted only after a
literal diagonal shift makes all three full Gram matrices positive definite
and leaves every coefficient residual positive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_sos_n6 as base


def build_problem_data():
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
    for vertex in vertices:
        stabilizer_actions = action2[group[:, vertex] == vertex]
        q_orbit = base.invariant_gram_orbits(stabilizer_actions)
        degree_five = base.vertex_degree_five_matrix(
            len(group), vertex, q_orbit, degree_five_orbits, degree_five_sizes
        )
        q_orbits.append(q_orbit)
        coefficient_matrices.append((lift @ degree_five).tocsr())
        print(
            f"cell={base.ALLOWED[vertex]} qvars={int(q_orbit.max()) + 1}",
            flush=True,
        )
    return (
        group,
        vertices,
        degree_six_reps,
        degree_six_sizes,
        target,
        q_orbits,
        coefficient_matrices,
    )


def quadratic_form_weights(q_orbit: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.bincount(
        q_orbit.ravel(),
        weights=np.multiply.outer(vector, vector).ravel(),
        minlength=int(q_orbit.max()) + 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--cuts-per-block", type=int, default=12)
    parser.add_argument("--trace-cap", type=float, default=10.0)
    parser.add_argument("--shift-padding", type=float, default=2e-9)
    parser.add_argument("--lp-method", choices=("highs", "highs-ipm", "highs-ds"), default="highs-ipm")
    parser.add_argument("--output", type=Path, default=Path("n6_cutting_plane_search.npz"))
    args = parser.parse_args()

    (
        group,
        vertices,
        degree_six_reps,
        degree_six_sizes,
        target,
        q_orbits,
        coefficient_matrices,
    ) = build_problem_data()
    q_offsets = []
    cursor = 0
    for q_orbit in q_orbits:
        q_offsets.append(cursor)
        cursor += int(q_orbit.max()) + 1
    margin_index = cursor
    number_of_variables = margin_index + 1
    coefficient_block = sparse.hstack(coefficient_matrices, format="csr")
    coefficient_constraints = sparse.hstack(
        (coefficient_block, np.ones((len(target), 1))), format="csr"
    )

    static_rows: list[int] = []
    static_columns: list[int] = []
    static_data: list[float] = []
    static_rhs: list[float] = []
    static_row = 0
    for q_orbit, offset in zip(q_orbits, q_offsets):
        diagonal_qids = sorted({int(q_orbit[i, i]) for i in range(len(base.M2))})
        for qid in diagonal_qids:
            static_rows.append(static_row)
            static_columns.append(offset + qid)
            static_data.append(-1.0)
            static_rhs.append(0.0)
            static_row += 1
        trace_counts = np.bincount(
            np.diag(q_orbit), minlength=int(q_orbit.max()) + 1
        )
        for qid, count in enumerate(trace_counts):
            if count:
                static_rows.append(static_row)
                static_columns.append(offset + qid)
                static_data.append(float(count))
        static_rhs.append(args.trace_cap)
        static_row += 1

        seen_pairs: set[int] = set()
        for left in range(len(base.M2)):
            for right in range(left + 1, len(base.M2)):
                offdiag_qid = int(q_orbit[left, right])
                if offdiag_qid in seen_pairs:
                    continue
                seen_pairs.add(offdiag_qid)
                left_qid = int(q_orbit[left, left])
                right_qid = int(q_orbit[right, right])
                for sign in (-2.0, 2.0):
                    static_rows.extend((static_row, static_row, static_row))
                    static_columns.extend(
                        (offset + left_qid, offset + right_qid, offset + offdiag_qid)
                    )
                    static_data.extend((-1.0, -1.0, sign))
                    static_rhs.append(0.0)
                    static_row += 1
        print(
            f"initial pair tests={2 * len(seen_pairs)} diagonal tests={len(diagonal_qids)}",
            flush=True,
        )

    static_constraints = sparse.coo_array(
        (static_data, (static_rows, static_columns)),
        shape=(static_row, number_of_variables),
        dtype=float,
    ).tocsr()
    spectral_cuts: list[tuple[int, np.ndarray]] = []
    objective = np.zeros(number_of_variables)
    objective[margin_index] = -1.0
    bounds = [(None, None)] * number_of_variables
    accepted_values: list[np.ndarray] | None = None
    accepted_residual: np.ndarray | None = None
    accepted_eigenvalues: list[float] | None = None

    for iteration in range(args.iterations):
        if spectral_cuts:
            cut_rows = []
            for block, weights in spectral_cuts:
                columns = q_offsets[block] + np.arange(len(weights), dtype=np.int64)
                row = sparse.coo_array(
                    (-weights, (np.zeros(len(weights), dtype=np.int64), columns)),
                    shape=(1, number_of_variables),
                    dtype=float,
                ).tocsr()
                cut_rows.append(row)
            cut_matrix = sparse.vstack(cut_rows, format="csr")
            inequalities = sparse.vstack(
                (coefficient_constraints, static_constraints, cut_matrix), format="csr"
            )
        else:
            inequalities = sparse.vstack(
                (coefficient_constraints, static_constraints), format="csr"
            )
        right_hand_side = np.concatenate(
            (target, np.asarray(static_rhs), np.zeros(len(spectral_cuts)))
        )
        result = optimize.linprog(
            objective,
            A_ub=inequalities,
            b_ub=right_hand_side,
            bounds=bounds,
            method=args.lp_method,
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            print(result.message, flush=True)
            raise SystemExit(1)
        values = [
            result.x[offset : offset + int(q_orbit.max()) + 1]
            for offset, q_orbit in zip(q_offsets, q_orbits)
        ]
        eigenpairs = [np.linalg.eigh(value[q_orbit]) for value, q_orbit in zip(values, q_orbits)]
        minimum_eigenvalues = [float(eigenvalues[0]) for eigenvalues, _ in eigenpairs]
        shifted_values = []
        shifts = []
        for value, q_orbit, minimum in zip(values, q_orbits, minimum_eigenvalues):
            shift = max(0.0, -minimum) + args.shift_padding
            shifted = value.copy()
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
        print(
            f"iteration={iteration} LP-margin={result.x[margin_index]:.12g} "
            f"eigen-min={minimum_eigenvalues} shifts={shifts} "
            f"shifted-residual={float(shifted_residual.min()):.12g} "
            f"cuts={len(spectral_cuts)}",
            flush=True,
        )
        if shifted_residual.min() > 0:
            accepted_values = shifted_values
            accepted_residual = shifted_residual
            accepted_eigenvalues = [
                float(np.linalg.eigvalsh(value[q_orbit]).min())
                for value, q_orbit in zip(accepted_values, q_orbits)
            ]
            break
        for block, (eigenvalues, eigenvectors) in enumerate(eigenpairs):
            number = min(args.cuts_per_block, len(eigenvalues))
            for index in range(number):
                if eigenvalues[index] >= -1e-10:
                    break
                spectral_cuts.append(
                    (block, quadratic_form_weights(q_orbits[block], eigenvectors[:, index]))
                )

    if accepted_values is None or accepted_residual is None or accepted_eigenvalues is None:
        raise SystemExit("no shifted positive certificate found")
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
        "minimum_eigenvalues": np.asarray(accepted_eigenvalues),
        "spectral_cut_count": np.asarray(len(spectral_cuts)),
    }
    for number, (q_orbit, value) in enumerate(zip(q_orbits, accepted_values)):
        payload[f"q_orbit_{number}"] = q_orbit
        payload[f"q_values_{number}"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
