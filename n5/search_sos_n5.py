#!/usr/bin/env python3
"""Numerical search for a symmetry-invariant n=5 SOS certificate.

This is a discovery program, not a verifier.  It searches for

    G = S * m_2(x)^T Q m_2(x) + R,

where Q is positive semidefinite and R has positive coefficients.  The
two-zero face is x[0,0] = x[1,1] = 0.  All coefficients and Gram entries are
identified under the full automorphism group of that face.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import optimize, sparse


N = 5
ZEROS = {(0, 0), (1, 1)}
ALLOWED = tuple((i, j) for i in range(N) for j in range(N) if (i, j) not in ZEROS)
NV = len(ALLOWED)
M2 = tuple(itertools.combinations_with_replacement(range(NV), 2))
M3 = tuple(itertools.combinations_with_replacement(range(NV), 3))
M5 = tuple(itertools.combinations_with_replacement(range(NV), 5))
C = 1226 / 1_953_125


def variable_group() -> list[tuple[int, ...]]:
    index = {cell: i for i, cell in enumerate(ALLOWED)}
    group: list[tuple[int, ...]] = []
    for swap in range(2):
        for lower_rows in itertools.permutations(range(2, N)):
            for lower_cols in itertools.permutations(range(2, N)):
                row = ([1, 0] if swap else [0, 1]) + list(lower_rows)
                col = ([1, 0] if swap else [0, 1]) + list(lower_cols)
                for transpose in range(2):
                    action = []
                    for i, j in ALLOWED:
                        image = (row[i], col[j])
                        if transpose:
                            image = image[::-1]
                        action.append(index[image])
                    group.append(tuple(action))
    return list(dict.fromkeys(group))


def induced_action(
    monomials: tuple[tuple[int, ...], ...], group: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    index = {monomial: i for i, monomial in enumerate(monomials)}
    return [
        tuple(index[tuple(sorted(action[v] for v in monomial))] for monomial in monomials)
        for action in group
    ]


def orbit_ids(size: int, actions: list[tuple[int, ...]]) -> tuple[np.ndarray, list[int]]:
    ids = np.full(size, -1, dtype=np.int32)
    representatives: list[int] = []
    for seed in range(size):
        if ids[seed] >= 0:
            continue
        orbit = {action[seed] for action in actions}
        orbit_id = len(representatives)
        representatives.append(min(orbit))
        ids[list(orbit)] = orbit_id
    assert np.all(ids >= 0)
    return ids, representatives


def target_coefficient(monomial: tuple[int, ...]) -> float:
    multiplicities = {v: monomial.count(v) for v in set(monomial)}
    multinomial = math.factorial(5)
    for count in multiplicities.values():
        multinomial //= math.factorial(count)
    cells = [ALLOWED[v] for v in monomial]
    edge = len(multiplicities) == 5 and (
        len({i for i, _ in cells}) == 5 or len({j for _, j in cells}) == 5
    )
    return C * multinomial - int(edge)


def coefficient_matrix(
    q_orbit: np.ndarray,
    m5_orbit: np.ndarray,
    m5_representatives: list[int],
) -> sparse.csr_array:
    """Return coefficients of S*m2^T Q*m2, one row per degree-5 orbit."""
    m5_index = {monomial: i for i, monomial in enumerate(M5)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for a in range(len(M2)):
        for b in range(a, len(M2)):
            qid = int(q_orbit[a, b])
            gram_factor = 1 if a == b else 2
            base = M2[a] + M2[b]
            for v in range(NV):
                output = m5_index[tuple(sorted(base + (v,)))]
                rows.append(int(m5_orbit[output]))
                cols.append(qid)
                data.append(gram_factor)
    totals = sparse.coo_array(
        (data, (rows, cols)),
        shape=(len(m5_representatives), int(q_orbit.max()) + 1),
        dtype=np.int64,
    ).tocsr()
    orbit_sizes = np.bincount(m5_orbit)
    for row, divisor in enumerate(orbit_sizes):
        start, end = totals.indptr[row : row + 2]
        assert np.all(totals.data[start:end] % divisor == 0)
        totals.data[start:end] //= divisor
    return totals


def pair_coefficient_matrices(
    group: list[tuple[int, ...]],
    pair_representatives: list[int],
    m5_orbit: np.ndarray,
    m5_representatives: list[int],
) -> list[sparse.csr_array]:
    """Coefficients of S times the group-averaged pair-multiplier forms."""
    m5_index = {monomial: i for i, monomial in enumerate(M5)}
    orbit_sizes = np.bincount(m5_orbit)
    upper = tuple(itertools.combinations_with_replacement(range(NV), 2))
    matrices = []
    for number, representative in enumerate(pair_representatives, 1):
        u, v = M2[representative]
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        for column, (a, b) in enumerate(upper):
            counts: dict[int, int] = {}
            gram_factor = 1 if a == b else 2
            for action in group:
                base = (action[u], action[v], action[a], action[b])
                for extra in range(NV):
                    output = m5_index[tuple(sorted(base + (extra,)))]
                    row = int(m5_orbit[output])
                    counts[row] = counts.get(row, 0) + gram_factor
            for row, total in counts.items():
                assert total % orbit_sizes[row] == 0
                rows.append(row)
                cols.append(column)
                data.append(total // orbit_sizes[row])
        matrices.append(
            sparse.coo_array(
                (data, (rows, cols)),
                shape=(len(m5_representatives), len(upper)),
                dtype=np.int64,
            ).tocsr()
        )
        print(f"built pair block {number}/{len(pair_representatives)}", flush=True)
    return matrices


def triple_coefficient_matrices(
    group: list[tuple[int, ...]],
    triple_representatives: list[int],
    m5_orbit: np.ndarray,
    m5_representatives: list[int],
) -> list[sparse.csr_array]:
    """Coefficients of group-averaged triple-multiplier linear SOS forms."""
    m5_index = {monomial: i for i, monomial in enumerate(M5)}
    orbit_sizes = np.bincount(m5_orbit)
    upper = tuple(itertools.combinations_with_replacement(range(NV), 2))
    matrices = []
    for number, representative in enumerate(triple_representatives, 1):
        multiplier = M3[representative]
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        for column, (a, b) in enumerate(upper):
            counts: dict[int, int] = {}
            gram_factor = 1 if a == b else 2
            for action in group:
                output = m5_index[
                    tuple(sorted(tuple(action[v] for v in multiplier) + (action[a], action[b])))
                ]
                row = int(m5_orbit[output])
                counts[row] = counts.get(row, 0) + gram_factor
            for row, total in counts.items():
                assert total % orbit_sizes[row] == 0
                rows.append(row)
                cols.append(column)
                data.append(total // orbit_sizes[row])
        matrices.append(
            sparse.coo_array(
                (data, (rows, cols)),
                shape=(len(m5_representatives), len(upper)),
                dtype=np.int64,
            ).tocsr()
        )
        print(f"built triple block {number}/{len(triple_representatives)}", flush=True)
    return matrices


def isotypic_bases(q_orbit: np.ndarray, number_of_orbits: int) -> list[np.ndarray]:
    """Numerically find the isotypic blocks of the invariant Gram algebra."""
    rng = np.random.default_rng(20260723)
    random_gram = rng.normal(size=number_of_orbits)[q_orbit]
    eigenvalues, eigenvectors = np.linalg.eigh(random_gram)
    eigenspaces: list[np.ndarray] = []
    start = 0
    for end in range(1, len(eigenvalues) + 1):
        separated = end == len(eigenvalues) or abs(eigenvalues[end] - eigenvalues[start]) > (
            1e-7 * (1 + abs(eigenvalues[start]))
        )
        if separated:
            eigenspaces.append(np.arange(start, end))
            start = end
    adjacency = [{i} for i in range(len(eigenspaces))]
    for _ in range(3):
        test = eigenvectors.T @ rng.normal(size=number_of_orbits)[q_orbit] @ eigenvectors
        for i, left in enumerate(eigenspaces):
            for j in range(i):
                right = eigenspaces[j]
                if np.linalg.norm(test[np.ix_(left, right)]) > 1e-7:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
    components: list[list[int]] = []
    seen: set[int] = set()
    for seed in range(len(eigenspaces)):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        component = []
        while stack:
            item = stack.pop()
            component.append(item)
            for neighbor in adjacency[item]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return [
        eigenvectors[:, np.concatenate([eigenspaces[i] for i in component])]
        for component in components
    ]


def invariant_gram_orbits(actions: list[tuple[int, ...]]) -> np.ndarray:
    """Orbit identifiers for symmetric Gram entries under a subgroup."""
    result = np.full((len(M2), len(M2)), -1, dtype=np.int32)
    orbit_id = 0
    for a in range(len(M2)):
        for b in range(a, len(M2)):
            if result[a, b] >= 0:
                continue
            orbit = {tuple(sorted((action[a], action[b]))) for action in actions}
            for i, j in orbit:
                result[i, j] = result[j, i] = orbit_id
            orbit_id += 1
    assert np.all(result >= 0)
    return result


def vertex_coefficient_matrix(
    group: list[tuple[int, ...]],
    vertex: int,
    q_orbit: np.ndarray,
    m5_orbit: np.ndarray,
    m5_representatives: list[int],
) -> sparse.csr_array:
    """Coefficients of the group average of x_vertex times a quadratic SOS."""
    m5_index = {monomial: i for i, monomial in enumerate(M5)}
    orbit_sizes = np.bincount(m5_orbit)
    totals: dict[tuple[int, int], int] = {}
    for a in range(len(M2)):
        for b in range(a, len(M2)):
            qid = int(q_orbit[a, b])
            gram_factor = 1 if a == b else 2
            for action in group:
                output = m5_index[
                    tuple(
                        sorted(
                            (action[vertex],)
                            + tuple(action[v] for v in M2[a])
                            + tuple(action[v] for v in M2[b])
                        )
                    )
                ]
                row = int(m5_orbit[output])
                key = (row, qid)
                totals[key] = totals.get(key, 0) + gram_factor
    rows = []
    cols = []
    data = []
    for (row, column), total in totals.items():
        assert total % orbit_sizes[row] == 0
        rows.append(row)
        cols.append(column)
        data.append(total // orbit_sizes[row])
    return sparse.coo_array(
        (data, (rows, cols)),
        shape=(len(m5_representatives), int(q_orbit.max()) + 1),
        dtype=np.int64,
    ).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=(
            "CLARABEL",
            "BLOCK",
            "BLOCK_SCS",
            "PAIR",
            "PAIR_SCS",
            "COMBO",
            "COMBO_SCS",
            "TRIPLE",
            "TRIPLE_SCS",
            "MIX",
            "MIX_SCS",
            "VERTEX_SCS",
            "VERTEX_FULL_SCS",
            "SCS",
            "DD",
        ),
        default="VERTEX_FULL_SCS",
    )
    parser.add_argument("--output", type=Path, default=Path("n5_sos_search.npz"))
    args = parser.parse_args()

    group = variable_group()
    assert len(group) == 144
    action2 = induced_action(M2, group)
    action3 = induced_action(M3, group)
    action5 = induced_action(M5, group)
    m2_orbit, reps2 = orbit_ids(len(M2), action2)
    _, vertex_reps = orbit_ids(NV, group)
    _, reps3 = orbit_ids(len(M3), action3)
    m5_orbit, reps5 = orbit_ids(len(M5), action5)

    q_orbit = np.full((len(M2), len(M2)), -1, dtype=np.int32)
    q_representatives: list[tuple[int, int]] = []
    for a in range(len(M2)):
        for b in range(a, len(M2)):
            if q_orbit[a, b] >= 0:
                continue
            orbit = {
                tuple(sorted((action[a], action[b])))
                for action in action2
            }
            qid = len(q_representatives)
            q_representatives.append(min(orbit))
            for i, j in orbit:
                q_orbit[i, j] = q_orbit[j, i] = qid
    assert np.all(q_orbit >= 0)

    coefficients = coefficient_matrix(q_orbit, m5_orbit, reps5)
    target = np.array([target_coefficient(M5[i]) for i in reps5])
    print(
        f"group={len(group)} m2={len(M2)} q_orbits={len(q_representatives)} "
        f"m5={len(M5)} coefficient_orbits={len(reps5)}",
        flush=True,
    )

    if args.solver in ("PAIR", "PAIR_SCS", "COMBO", "COMBO_SCS"):
        pair_coefficients = pair_coefficient_matrices(group, reps2, m5_orbit, reps5)
        upper = tuple(itertools.combinations_with_replacement(range(NV), 2))
        pair_grams = [cp.Variable((NV, NV), symmetric=True) for _ in reps2]
        pair_vectors = [cp.hstack([gram[i, j] for i, j in upper]) for gram in pair_grams]
        margin = cp.Variable()
        produced = sum(
            (matrix @ vector for matrix, vector in zip(pair_coefficients, pair_vectors)),
            cp.Constant(np.zeros(len(reps5))),
        )
        constraints = [gram >> 0 for gram in pair_grams]
        global_values = None
        if args.solver in ("COMBO", "COMBO_SCS"):
            global_values = cp.Variable(len(q_representatives))
            global_gram = cp.reshape(
                global_values[q_orbit.ravel()], q_orbit.shape, order="C"
            )
            bases = isotypic_bases(q_orbit, len(q_representatives))
            print("PSD block sizes", sorted(base.shape[1] for base in bases), flush=True)
            constraints.extend(base.T @ global_gram @ base >> 0 for base in bases)
            produced = produced + coefficients @ global_values
        constraints.append(produced + margin <= target)
        problem = cp.Problem(cp.Maximize(margin), constraints)
        if args.solver in ("PAIR", "COMBO"):
            result = problem.solve(
                solver=cp.CLARABEL,
                verbose=True,
                tol_gap_abs=1e-9,
                tol_feas=1e-9,
                tol_gap_rel=1e-9,
                max_iter=500,
            )
        else:
            result = problem.solve(solver=cp.SCS, verbose=True, eps=1e-7, max_iters=200_000)
        print("status", problem.status)
        print("objective", result)
        if any(gram.value is None for gram in pair_grams):
            raise SystemExit("solver returned no pair Gram matrices")
        gram_values = np.asarray([gram.value for gram in pair_grams])
        vectors = [np.asarray([gram[i, j] for i, j in upper]) for gram in gram_values]
        residual = target - sum(
            (matrix @ vector for matrix, vector in zip(pair_coefficients, vectors)),
            np.zeros(len(reps5)),
        )
        if global_values is not None:
            residual -= coefficients @ np.asarray(global_values.value)
        eig = np.linalg.eigvalsh(gram_values)
        print("minimum eigenvalue", eig.min())
        print("minimum residual", residual.min())
        np.savez_compressed(
            args.output,
            allowed=np.asarray(ALLOWED, dtype=np.int8),
            group=np.asarray(group, dtype=np.int16),
            m2=np.asarray(M2, dtype=np.int16),
            pair_representatives=np.asarray(reps2, dtype=np.int16),
            pair_gram_values=gram_values,
            m5_representatives=np.asarray(reps5, dtype=np.int32),
            target=target,
            residual=residual,
            margin=float(margin.value),
            global_q_values=(
                np.asarray(global_values.value) if global_values is not None else np.asarray([])
            ),
        )
        return

    if args.solver in ("TRIPLE", "TRIPLE_SCS", "MIX", "MIX_SCS"):
        triple_coefficients = triple_coefficient_matrices(group, reps3, m5_orbit, reps5)
        upper = tuple(itertools.combinations_with_replacement(range(NV), 2))
        triple_grams = [cp.Variable((NV, NV), symmetric=True) for _ in reps3]
        triple_vectors = [cp.hstack([gram[i, j] for i, j in upper]) for gram in triple_grams]
        margin = cp.Variable()
        produced = sum(
            (matrix @ vector for matrix, vector in zip(triple_coefficients, triple_vectors)),
            cp.Constant(np.zeros(len(reps5))),
        )
        constraints = [gram >> 0 for gram in triple_grams]
        pair_grams = None
        pair_coefficients = None
        if args.solver in ("MIX", "MIX_SCS"):
            pair_coefficients = pair_coefficient_matrices(group, reps2, m5_orbit, reps5)
            pair_grams = [cp.Variable((NV, NV), symmetric=True) for _ in reps2]
            pair_vectors = [cp.hstack([gram[i, j] for i, j in upper]) for gram in pair_grams]
            constraints.extend(gram >> 0 for gram in pair_grams)
            produced = produced + sum(
                (matrix @ vector for matrix, vector in zip(pair_coefficients, pair_vectors)),
                cp.Constant(np.zeros(len(reps5))),
            )
        constraints.append(produced + margin <= target)
        problem = cp.Problem(cp.Maximize(margin), constraints)
        if args.solver in ("TRIPLE", "MIX"):
            result = problem.solve(
                solver=cp.CLARABEL,
                verbose=True,
                tol_gap_abs=1e-9,
                tol_feas=1e-9,
                tol_gap_rel=1e-9,
                max_iter=500,
            )
        else:
            result = problem.solve(solver=cp.SCS, verbose=True, eps=1e-7, max_iters=200_000)
        print("status", problem.status)
        print("objective", result)
        if any(gram.value is None for gram in triple_grams):
            raise SystemExit("solver returned no triple Gram matrices")
        gram_values = np.asarray([gram.value for gram in triple_grams])
        vectors = [np.asarray([gram[i, j] for i, j in upper]) for gram in gram_values]
        residual = target - sum(
            (matrix @ vector for matrix, vector in zip(triple_coefficients, vectors)),
            np.zeros(len(reps5)),
        )
        pair_gram_values = np.asarray([])
        if pair_grams is not None and pair_coefficients is not None:
            pair_gram_values = np.asarray([gram.value for gram in pair_grams])
            pair_vectors_value = [
                np.asarray([gram[i, j] for i, j in upper]) for gram in pair_gram_values
            ]
            residual -= sum(
                (
                    matrix @ vector
                    for matrix, vector in zip(pair_coefficients, pair_vectors_value)
                ),
                np.zeros(len(reps5)),
            )
        eig = np.linalg.eigvalsh(gram_values)
        print("minimum eigenvalue", eig.min())
        print("minimum residual", residual.min())
        np.savez_compressed(
            args.output,
            allowed=np.asarray(ALLOWED, dtype=np.int8),
            group=np.asarray(group, dtype=np.int16),
            m3=np.asarray(M3, dtype=np.int16),
            triple_representatives=np.asarray(reps3, dtype=np.int16),
            triple_gram_values=gram_values,
            m5_representatives=np.asarray(reps5, dtype=np.int32),
            target=target,
            residual=residual,
            margin=float(margin.value),
            pair_gram_values=pair_gram_values,
        )
        return

    if args.solver in ("VERTEX_SCS", "VERTEX_FULL_SCS"):
        vertex_values = []
        vertex_grams = []
        vertex_matrices = []
        constraints = []
        for number, vertex in enumerate(vertex_reps, 1):
            stabilizer_actions = [
                action2 for action, action2 in zip(group, action2) if action[vertex] == vertex
            ]
            q_vertex = invariant_gram_orbits(stabilizer_actions)
            values_vertex = cp.Variable(int(q_vertex.max()) + 1)
            gram_vertex = cp.reshape(
                values_vertex[q_vertex.ravel()], q_vertex.shape, order="C"
            )
            if args.solver == "VERTEX_SCS":
                bases_vertex = isotypic_bases(q_vertex, int(q_vertex.max()) + 1)
                constraints.extend(base.T @ gram_vertex @ base >> 0 for base in bases_vertex)
            else:
                bases_vertex = []
                constraints.append(gram_vertex >> 0)
            matrix_vertex = vertex_coefficient_matrix(
                group, vertex, q_vertex, m5_orbit, reps5
            )
            vertex_values.append(values_vertex)
            vertex_grams.append((q_vertex, bases_vertex))
            vertex_matrices.append(matrix_vertex)
            print(
                f"built vertex block {number}/{len(vertex_reps)}: vertex={vertex} "
                f"variables={int(q_vertex.max()) + 1} "
                f"PSD-blocks={sorted(base.shape[1] for base in bases_vertex) or [len(M2)]}",
                flush=True,
            )
        margin = cp.Variable()
        produced = sum(
            (matrix @ values for matrix, values in zip(vertex_matrices, vertex_values)),
            cp.Constant(np.zeros(len(reps5))),
        )
        constraints.append(produced + margin <= target)
        problem = cp.Problem(cp.Maximize(margin), constraints)
        result = problem.solve(solver=cp.SCS, verbose=True, eps=1e-7, max_iters=200_000)
        print("status", problem.status)
        print("objective", result)
        if any(values.value is None for values in vertex_values):
            raise SystemExit("solver returned no vertex Gram matrices")
        residual = target - sum(
            (
                matrix @ np.asarray(values.value)
                for matrix, values in zip(vertex_matrices, vertex_values)
            ),
            np.zeros(len(reps5)),
        )
        print("minimum residual", residual.min())
        payload = {
            "allowed": np.asarray(ALLOWED, dtype=np.int8),
            "group": np.asarray(group, dtype=np.int16),
            "vertex_representatives": np.asarray(vertex_reps, dtype=np.int16),
            "m5_representatives": np.asarray(reps5, dtype=np.int32),
            "target": target,
            "residual": residual,
            "margin": float(margin.value),
        }
        for number, (values, (q_vertex, _)) in enumerate(zip(vertex_values, vertex_grams)):
            payload[f"vertex_q_orbit_{number}"] = q_vertex
            payload[f"vertex_q_values_{number}"] = np.asarray(values.value)
        np.savez_compressed(args.output, **payload)
        return

    if args.solver == "DD":
        # A symmetric diagonally dominant Gram matrix is PSD.  This LP is a
        # cheap feasibility probe before attempting the full semidefinite cone.
        nq = len(q_representatives)
        diagonal_qids = {int(q_orbit[i, i]) for i in range(len(M2))}
        offdiag_qids = sorted(set(range(nq)) - diagonal_qids)
        t_index = {qid: nq + k for k, qid in enumerate(offdiag_qids)}
        margin_index = nq + len(offdiag_qids)
        row_blocks = [
            sparse.hstack(
                (
                    coefficients,
                    sparse.csr_array((coefficients.shape[0], len(offdiag_qids))),
                    np.ones((coefficients.shape[0], 1)),
                ),
                format="csr",
            )
        ]
        rhs = [target]
        # |Q_ab| <= t_orbit(a,b).
        abs_rows = []
        abs_rhs = []
        for qid in offdiag_qids:
            for sign in (-1, 1):
                row = sparse.dok_array((1, margin_index + 1), dtype=float)
                row[0, qid] = sign
                row[0, t_index[qid]] = -1
                abs_rows.append(row.tocsr())
                abs_rhs.append(0.0)
        row_blocks.append(sparse.vstack(abs_rows, format="csr"))
        rhs.append(np.asarray(abs_rhs))
        # One diagonal-dominance inequality per orbit of quadratic monomials.
        dd_rows = []
        dd_rhs = []
        for i in sorted(set(int(v) for v in m2_orbit)):
            representative = int(np.flatnonzero(m2_orbit == i)[0])
            row = sparse.dok_array((1, margin_index + 1), dtype=float)
            row[0, int(q_orbit[representative, representative])] = -1
            counts: dict[int, int] = {}
            for j in range(len(M2)):
                if j == representative:
                    continue
                qid = int(q_orbit[representative, j])
                counts[qid] = counts.get(qid, 0) + 1
            for qid, count in counts.items():
                row[0, t_index[qid]] = count
            dd_rows.append(row.tocsr())
            dd_rhs.append(0.0)
        row_blocks.append(sparse.vstack(dd_rows, format="csr"))
        rhs.append(np.asarray(dd_rhs))
        objective = np.zeros(margin_index + 1)
        objective[margin_index] = -1
        bounds = [(None, None)] * nq + [(0, None)] * len(offdiag_qids) + [(None, None)]
        lp = optimize.linprog(
            objective,
            A_ub=sparse.vstack(row_blocks, format="csr"),
            b_ub=np.concatenate(rhs),
            bounds=bounds,
            method="highs",
            options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
        )
        print(lp.message)
        print("objective", None if lp.fun is None else -lp.fun)
        if not lp.success:
            raise SystemExit(1)
        q_value = lp.x[:nq]
        gram_value = q_value[q_orbit]
        residual = target - coefficients @ q_value
        eig = np.linalg.eigvalsh(gram_value)
        print("minimum eigenvalue", eig[0])
        print("minimum residual", residual.min())
        np.savez_compressed(
            args.output,
            allowed=np.asarray(ALLOWED, dtype=np.int8),
            group=np.asarray(group, dtype=np.int16),
            m2=np.asarray(M2, dtype=np.int16),
            q_orbit=q_orbit,
            q_representatives=np.asarray(q_representatives, dtype=np.int16),
            q_values=q_value,
            m5_representatives=np.asarray(reps5, dtype=np.int32),
            target=target,
            residual=residual,
            margin=float(-lp.fun),
        )
        return

    values = cp.Variable(len(q_representatives))
    gram = cp.reshape(values[q_orbit.ravel()], q_orbit.shape, order="C")
    margin = cp.Variable()
    if args.solver in ("BLOCK", "BLOCK_SCS"):
        # Numerically split the permutation representation into isotypic
        # components.  Every invariant Gram matrix is block diagonal in this
        # basis.  Tiny numerical off-block errors are handled during the later
        # exact positive-definite reconstruction; this remains a discovery run.
        bases = isotypic_bases(q_orbit, len(q_representatives))
        print("PSD block sizes", sorted(base.shape[1] for base in bases), flush=True)
        constraints = [base.T @ gram @ base >> 0 for base in bases]
        constraints.append(coefficients @ values + margin <= target)
    else:
        constraints = [gram >> 0, coefficients @ values + margin <= target]
    problem = cp.Problem(cp.Maximize(margin), constraints)
    if args.solver in ("CLARABEL", "BLOCK"):
        result = problem.solve(
            solver=cp.CLARABEL,
            verbose=True,
            tol_gap_abs=1e-10,
            tol_feas=1e-10,
            tol_gap_rel=1e-10,
            max_iter=500,
        )
    else:
        result = problem.solve(solver=cp.SCS, verbose=True, eps=1e-7, max_iters=200_000)
    print("status", problem.status)
    print("objective", result)
    if values.value is None:
        raise SystemExit("solver returned no Gram matrix")
    q_value = np.asarray(values.value)
    gram_value = q_value[q_orbit]
    residual = target - coefficients @ q_value
    eig = np.linalg.eigvalsh(gram_value)
    print("minimum eigenvalue", eig[0])
    print("minimum residual", residual.min())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        allowed=np.asarray(ALLOWED, dtype=np.int8),
        group=np.asarray(group, dtype=np.int16),
        m2=np.asarray(M2, dtype=np.int16),
        q_orbit=q_orbit,
        q_representatives=np.asarray(q_representatives, dtype=np.int16),
        q_values=q_value,
        m5_representatives=np.asarray(reps5, dtype=np.int32),
        target=target,
        residual=residual,
        margin=float(margin.value),
    )


if __name__ == "__main__":
    main()
