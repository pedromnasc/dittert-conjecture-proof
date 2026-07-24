#!/usr/bin/env python3
"""Combined vertex- and pair-multiplier SDP search for Dittert n=6."""
from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import numpy as np

import search_pair_sos_n6 as pair_search
import search_sos_n6 as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--warm-vertex", type=Path)
    parser.add_argument("--output", type=Path, default=Path("n6_combined_sos_search.npz"))
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

    warm = np.load(args.warm_vertex, allow_pickle=False) if args.warm_vertex else None
    vertex_variables = []
    vertex_q_orbits = []
    vertex_matrices = []
    constraints = []
    produced_terms = []
    for number, vertex in enumerate(vertices):
        stabilizer_actions = action2[group[:, vertex] == vertex]
        q_orbit = base.invariant_gram_orbits(stabilizer_actions)
        values = cp.Variable(int(q_orbit.max()) + 1)
        if warm is not None:
            saved_q_orbit = warm[f"q_orbit_{number}"]
            if not np.array_equal(saved_q_orbit, q_orbit):
                raise AssertionError("warm-start Gram orbit mismatch")
            values.value = warm[f"q_values_{number}"]
        gram = cp.reshape(values[q_orbit.ravel()], q_orbit.shape, order="C")
        degree_five = base.vertex_degree_five_matrix(
            len(group), vertex, q_orbit, degree_five_orbits, degree_five_sizes
        )
        coefficient = (lift_vertex @ degree_five).tocsr()
        constraints.append(gram >> 0)
        produced_terms.append(coefficient @ values)
        vertex_variables.append(values)
        vertex_q_orbits.append(q_orbit)
        vertex_matrices.append(coefficient)
        print(
            f"vertex {number + 1}/3 cell={base.ALLOWED[vertex]} "
            f"qvars={int(q_orbit.max()) + 1}",
            flush=True,
        )

    pair_grams = []
    pair_matrices = []
    for number, pair in enumerate(pair_representatives):
        gram = cp.Variable((base.NV, base.NV), symmetric=True)
        vector = cp.hstack(
            [gram[int(i), int(j)] for i, j in pair_search.LINEAR_UPPER]
        )
        degree_four = pair_search.pair_degree_four_matrix(
            len(group), pair, degree_four_orbits, degree_four_sizes
        )
        coefficient = (lift_pair @ degree_four).tocsr()
        constraints.append(gram >> 0)
        produced_terms.append(coefficient @ vector)
        pair_grams.append(gram)
        pair_matrices.append(coefficient)
        print(f"pair {number + 1}/16={tuple(int(v) for v in pair)}", flush=True)

    margin = cp.Variable()
    produced = sum(produced_terms, cp.Constant(np.zeros(len(target))))
    constraints.append(produced + margin <= target)
    problem = cp.Problem(cp.Maximize(margin), constraints)
    result = problem.solve(
        solver=cp.SCS,
        verbose=True,
        eps=args.eps,
        max_iters=args.max_iters,
        acceleration_lookback=20,
        use_indirect=True,
        warm_start=warm is not None,
    )
    print("status", problem.status, flush=True)
    print("objective", result, flush=True)
    if any(values.value is None for values in vertex_variables) or any(
        gram.value is None for gram in pair_grams
    ):
        raise SystemExit("solver returned an incomplete solution")
    vertex_values = [np.asarray(values.value) for values in vertex_variables]
    pair_values = [np.asarray(gram.value) for gram in pair_grams]
    pair_vectors = [
        np.asarray(
            [gram[int(i), int(j)] for i, j in pair_search.LINEAR_UPPER]
        )
        for gram in pair_values
    ]
    residual = target - sum(
        (matrix @ values for matrix, values in zip(vertex_matrices, vertex_values)),
        np.zeros(len(target)),
    )
    residual -= sum(
        (matrix @ values for matrix, values in zip(pair_matrices, pair_vectors)),
        np.zeros(len(target)),
    )
    vertex_eigenvalues = [
        float(np.linalg.eigvalsh(values[q_orbit]).min())
        for values, q_orbit in zip(vertex_values, vertex_q_orbits)
    ]
    pair_eigenvalues = [float(np.linalg.eigvalsh(gram).min()) for gram in pair_values]
    print("minimum residual", float(residual.min()), flush=True)
    print("vertex minimum eigenvalues", vertex_eigenvalues, flush=True)
    print("pair minimum eigenvalue", min(pair_eigenvalues), flush=True)

    payload: dict[str, np.ndarray] = {
        "allowed": np.asarray(base.ALLOWED, dtype=np.int8),
        "group": group,
        "vertices": np.asarray(vertices, dtype=np.int16),
        "pair_representatives": pair_representatives,
        "degree_six_representatives": degree_six_reps,
        "degree_six_sizes": degree_six_sizes,
        "target": target,
        "residual": residual,
        "margin": np.asarray(float(margin.value)),
        "vertex_minimum_eigenvalues": np.asarray(vertex_eigenvalues),
        "pair_minimum_eigenvalues": np.asarray(pair_eigenvalues),
        "pair_gram_values": np.asarray(pair_values),
    }
    for number, (q_orbit, values) in enumerate(zip(vertex_q_orbits, vertex_values)):
        payload[f"q_orbit_{number}"] = q_orbit
        payload[f"q_values_{number}"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
