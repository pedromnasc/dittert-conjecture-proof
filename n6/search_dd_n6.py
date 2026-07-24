#!/usr/bin/env python3
"""Cheap diagonally-dominant feasibility probe for the n=6 SOS search.

Every symmetric diagonally dominant Gram matrix with nonnegative diagonal is
positive semidefinite.  This script replaces the three semidefinite cones in
``search_sos_n6.py`` by linear diagonal-dominance constraints and solves the
resulting LP with HiGHS.  A positive answer can be rounded into an exact SOS
certificate; a negative answer only says that the DD subcone is too small.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

import search_sos_n6 as base


def point_orbits(actions: np.ndarray) -> tuple[np.ndarray, list[int]]:
    ids = np.full(actions.shape[1], -1, dtype=np.int32)
    representatives: list[int] = []
    for seed in range(actions.shape[1]):
        if ids[seed] >= 0:
            continue
        orbit = np.unique(actions[:, seed])
        orbit_id = len(representatives)
        ids[orbit] = orbit_id
        representatives.append(seed)
    if np.any(ids < 0):
        raise AssertionError("point orbit assignment incomplete")
    return ids, representatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("n6_dd_search.npz"))
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

    q_orbits: list[np.ndarray] = []
    coefficient_matrices: list[sparse.csr_array] = []
    row_representatives: list[list[int]] = []
    diagonal_ids: list[set[int]] = []
    offdiagonal_ids: list[list[int]] = []
    for vertex in vertices:
        stabilizer_actions = action2[group[:, vertex] == vertex]
        q_orbit = base.invariant_gram_orbits(stabilizer_actions)
        _, row_reps = point_orbits(stabilizer_actions)
        degree_five = base.vertex_degree_five_matrix(
            len(group), vertex, q_orbit, degree_five_orbits, degree_five_sizes
        )
        coefficient_matrices.append((lift @ degree_five).tocsr())
        q_orbits.append(q_orbit)
        row_representatives.append(row_reps)
        diagonal = {int(q_orbit[i, i]) for i in range(len(base.M2))}
        offdiagonal = sorted(set(range(int(q_orbit.max()) + 1)) - diagonal)
        diagonal_ids.append(diagonal)
        offdiagonal_ids.append(offdiagonal)
        print(
            f"cell={base.ALLOWED[vertex]} qvars={int(q_orbit.max()) + 1} "
            f"row-orbits={len(row_reps)} offdiagonal-orbits={len(offdiagonal)}",
            flush=True,
        )

    q_offsets: list[int] = []
    cursor = 0
    for q_orbit in q_orbits:
        q_offsets.append(cursor)
        cursor += int(q_orbit.max()) + 1
    number_of_q_variables = cursor
    t_offsets: list[int] = []
    for ids in offdiagonal_ids:
        t_offsets.append(cursor)
        cursor += len(ids)
    margin_index = cursor
    number_of_variables = margin_index + 1
    t_indices = [
        {qid: offset + number for number, qid in enumerate(ids)}
        for offset, ids in zip(t_offsets, offdiagonal_ids)
    ]

    coefficient_block = sparse.hstack(coefficient_matrices, format="csr")
    coefficient_rows = sparse.hstack(
        (
            coefficient_block,
            sparse.csr_array((len(target), margin_index - number_of_q_variables)),
            np.ones((len(target), 1)),
        ),
        format="csr",
    )
    row_indices: list[int] = list(coefficient_rows.tocoo().row)
    column_indices: list[int] = list(coefficient_rows.tocoo().col)
    data: list[float] = list(coefficient_rows.tocoo().data)
    right_hand_side: list[float] = list(target)
    next_row = len(target)

    for block, (q_orbit, q_offset, offdiagonal, t_index, reps) in enumerate(
        zip(q_orbits, q_offsets, offdiagonal_ids, t_indices, row_representatives)
    ):
        for qid in offdiagonal:
            for sign in (-1.0, 1.0):
                row_indices.extend((next_row, next_row))
                column_indices.extend((q_offset + qid, t_index[qid]))
                data.extend((sign, -1.0))
                right_hand_side.append(0.0)
                next_row += 1
        for representative in reps:
            diagonal_qid = int(q_orbit[representative, representative])
            row_indices.append(next_row)
            column_indices.append(q_offset + diagonal_qid)
            data.append(-1.0)
            counts: dict[int, int] = {}
            for column in range(len(base.M2)):
                if column == representative:
                    continue
                qid = int(q_orbit[representative, column])
                counts[qid] = counts.get(qid, 0) + 1
            for qid, count in counts.items():
                row_indices.append(next_row)
                column_indices.append(t_index[qid])
                data.append(float(count))
            right_hand_side.append(0.0)
            next_row += 1

    inequalities = sparse.coo_array(
        (data, (row_indices, column_indices)),
        shape=(next_row, number_of_variables),
        dtype=float,
    ).tocsr()
    objective = np.zeros(number_of_variables)
    objective[margin_index] = -1.0
    bounds = [(None, None)] * number_of_q_variables
    bounds.extend((0.0, None) for _ in range(margin_index - number_of_q_variables))
    bounds.append((None, None))
    print(
        f"LP variables={number_of_variables} inequalities={next_row} nnz={inequalities.nnz}",
        flush=True,
    )
    result = optimize.linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.asarray(right_hand_side),
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    print(result.message, flush=True)
    if not result.success:
        raise SystemExit(1)
    margin = float(result.x[margin_index])
    values = [
        result.x[offset : offset + int(q_orbit.max()) + 1]
        for offset, q_orbit in zip(q_offsets, q_orbits)
    ]
    residual = target - sum(
        (matrix @ value for matrix, value in zip(coefficient_matrices, values)),
        np.zeros(len(target)),
    )
    minimum_eigenvalues = [
        float(np.linalg.eigvalsh(value[q_orbit]).min())
        for value, q_orbit in zip(values, q_orbits)
    ]
    print("margin", margin, flush=True)
    print("minimum residual", float(residual.min()), flush=True)
    print("minimum eigenvalues", minimum_eigenvalues, flush=True)

    payload: dict[str, np.ndarray] = {
        "allowed": np.asarray(base.ALLOWED, dtype=np.int8),
        "group": group,
        "vertices": np.asarray(vertices, dtype=np.int16),
        "degree_six_representatives": degree_six_reps,
        "degree_six_sizes": degree_six_sizes,
        "target": target,
        "residual": residual,
        "margin": np.asarray(margin),
        "minimum_eigenvalues": np.asarray(minimum_eigenvalues),
    }
    for number, (q_orbit, value) in enumerate(zip(q_orbits, values)):
        payload[f"q_orbit_{number}"] = q_orbit
        payload[f"q_values_{number}"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
