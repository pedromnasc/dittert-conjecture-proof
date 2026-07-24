#!/usr/bin/env python3
"""Numerical search for a symmetry-invariant n=6 SOS certificate.

This is discovery code, not a proof verifier.  It searches for

    G_6 = sum_p sum_{gamma in Gamma}
              x_{gamma u_p} S m_2(gamma x)^T Q_p m_2(gamma x) + R,

where the three Q_p are positive semidefinite, invariant under the
stabilizer of u_p, and R has coefficientwise positive residual.  Multiplying
the degree-five vertex forms by S is the parity lift of the successful n=5
certificate architecture.

All coefficient constraints are reduced under the full automorphism group
of the face x[0,0] = x[1,1] = 0.  There are 5,605 degree-six coefficient
orbits instead of 3,262,623 individual coefficients.
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy import sparse


N = 6
ZEROS = {(0, 0), (1, 1)}
ALLOWED = tuple((i, j) for i in range(N) for j in range(N) if (i, j) not in ZEROS)
NV = len(ALLOWED)
M2 = np.asarray(
    tuple(itertools.combinations_with_replacement(range(NV), 2)), dtype=np.int16
)
TARGET_CONSTANT = 643 / 15_116_544


def variable_group() -> np.ndarray:
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
    result = np.asarray(list(dict.fromkeys(group)), dtype=np.int16)
    if result.shape != (2304, NV):
        raise AssertionError(f"unexpected group shape {result.shape}")
    return result


def binomial_table(limit: int) -> np.ndarray:
    table = np.zeros((limit + 1, limit + 1), dtype=np.int64)
    for n in range(limit + 1):
        for k in range(n + 1):
            table[n, k] = math.comb(n, k)
    return table


BINOMIAL = binomial_table(NV + 6)


def multiset_count(degree: int) -> int:
    return math.comb(NV + degree - 1, degree)


def multiset_array(degree: int) -> np.ndarray:
    count = multiset_count(degree)
    flat = np.fromiter(
        (
            value
            for monomial in itertools.combinations_with_replacement(range(NV), degree)
            for value in monomial
        ),
        dtype=np.int16,
        count=count * degree,
    )
    if flat.size != count * degree:
        raise AssertionError("multiset generation ended early")
    return flat.reshape(count, degree)


def rank_multisets(monomials: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Lexicographic ranks of sorted rows among combinations with replacement."""
    monomials = np.asarray(monomials, dtype=np.int64)
    if monomials.ndim != 2:
        raise ValueError("monomials must be a matrix")
    count, degree = monomials.shape
    strict = monomials + np.arange(degree, dtype=np.int64)
    universe = alphabet_size + degree - 1
    ranks = np.zeros(count, dtype=np.int64)
    previous = np.full(count, -1, dtype=np.int64)
    for position in range(degree):
        remaining = degree - position
        current = strict[:, position]
        ranks += BINOMIAL[universe - previous - 1, remaining]
        ranks -= BINOMIAL[universe - current, remaining]
        previous = current
    return ranks


def verify_ranking() -> None:
    for degree in (2, 5, 6):
        sample = multiset_array(degree)
        if degree == 6:
            sample = np.concatenate((sample[:1000], sample[-1000:]), axis=0)
            expected = np.concatenate(
                (
                    np.arange(1000, dtype=np.int64),
                    np.arange(multiset_count(6) - 1000, multiset_count(6)),
                )
            )
        else:
            expected = np.arange(len(sample), dtype=np.int64)
        actual = rank_multisets(sample, NV)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"degree-{degree} multiset rank failure")


def monomial_orbits(
    degree: int, group: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orbit ids, representatives, and sizes for degree-d multisets."""
    monomials = multiset_array(degree)
    orbit_ids = np.full(len(monomials), -1, dtype=np.int32)
    representatives: list[np.ndarray] = []
    for seed in range(len(monomials)):
        if orbit_ids[seed] >= 0:
            continue
        images = np.sort(group[:, monomials[seed]], axis=1)
        ranks = rank_multisets(images, NV)
        orbit_id = len(representatives)
        orbit_ids[ranks] = orbit_id
        representatives.append(monomials[seed].copy())
    if np.any(orbit_ids < 0):
        raise AssertionError(f"degree-{degree} orbit assignment incomplete")
    sizes = np.bincount(orbit_ids, minlength=len(representatives)).astype(np.int64)
    return orbit_ids, np.asarray(representatives, dtype=np.int16), sizes


def induced_m2_actions(group: np.ndarray) -> np.ndarray:
    transformed = np.sort(group[:, M2], axis=2)
    flat = transformed.reshape(-1, 2)
    return rank_multisets(flat, NV).reshape(len(group), len(M2)).astype(np.int16)


def vertex_representatives(group: np.ndarray) -> list[int]:
    unseen = set(range(NV))
    representatives: list[int] = []
    while unseen:
        seed = min(unseen)
        orbit = set(int(value) for value in group[:, seed])
        representatives.append(seed)
        unseen.difference_update(orbit)
    if [len(set(int(value) for value in group[:, v])) for v in representatives] != [2, 16, 16]:
        raise AssertionError("unexpected vertex orbits")
    return representatives


def invariant_gram_orbits(actions: np.ndarray) -> np.ndarray:
    size = actions.shape[1]
    result = np.full((size, size), -1, dtype=np.int32)
    orbit_id = 0
    for left in range(size):
        for right in range(left, size):
            if result[left, right] >= 0:
                continue
            images = np.sort(actions[:, (left, right)], axis=1)
            result[images[:, 0], images[:, 1]] = orbit_id
            result[images[:, 1], images[:, 0]] = orbit_id
            orbit_id += 1
    if np.any(result < 0):
        raise AssertionError("Gram orbit assignment incomplete")
    return result


def isotypic_bases(
    q_orbit: np.ndarray, number_of_orbits: int, *, verbose: bool = True
) -> list[np.ndarray]:
    """Numerically split the invariant Gram algebra into central blocks."""
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
    for _ in range(4):
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
        component: list[int] = []
        while stack:
            item = stack.pop()
            component.append(item)
            for neighbor in adjacency[item]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    if verbose:
        print(
            "isotypic irreducible dimensions",
            [[len(eigenspaces[i]) for i in component] for component in components],
            flush=True,
        )
    return [
        eigenvectors[:, np.concatenate([eigenspaces[i] for i in component])]
        for component in components
    ]


def vertex_degree_five_matrix(
    group_size: int,
    vertex: int,
    q_orbit: np.ndarray,
    degree_five_orbits: np.ndarray,
    degree_five_sizes: np.ndarray,
) -> sparse.csr_array:
    """Coefficients of the group-summed degree-five vertex form."""
    upper_left, upper_right = np.triu_indices(len(M2))
    gram_factor = np.where(upper_left == upper_right, 1, 2).astype(np.int64)
    bases = np.concatenate(
        (
            np.full((len(upper_left), 1), vertex, dtype=np.int16),
            M2[upper_left],
            M2[upper_right],
        ),
        axis=1,
    )
    bases.sort(axis=1)
    monomial_indices = rank_multisets(bases, NV)
    rows = degree_five_orbits[monomial_indices]
    columns = q_orbit[upper_left, upper_right]
    data = gram_factor * group_size
    matrix = sparse.coo_array(
        (data, (rows, columns)),
        shape=(len(degree_five_sizes), int(q_orbit.max()) + 1),
        dtype=np.int64,
    ).tocsr()
    for row, divisor in enumerate(degree_five_sizes):
        start, end = matrix.indptr[row : row + 2]
        if np.any(matrix.data[start:end] % divisor):
            raise AssertionError("degree-five orbit averaging is not integral")
        matrix.data[start:end] //= divisor
    return matrix


def multiplication_by_s(
    degree_five_orbits: np.ndarray,
    degree_six_representatives: np.ndarray,
    number_of_degree_five_orbits: int,
) -> sparse.csr_array:
    rows: list[int] = []
    columns: list[int] = []
    data: list[int] = []
    for row, monomial in enumerate(degree_six_representatives):
        for variable in np.unique(monomial):
            predecessor = list(int(value) for value in monomial)
            predecessor.remove(int(variable))
            index = int(rank_multisets(np.asarray([predecessor]), NV)[0])
            rows.append(row)
            columns.append(int(degree_five_orbits[index]))
            data.append(1)
    return sparse.coo_array(
        (data, (rows, columns)),
        shape=(len(degree_six_representatives), number_of_degree_five_orbits),
        dtype=np.int64,
    ).tocsr()


def target_coefficients(representatives: np.ndarray) -> np.ndarray:
    result = []
    for monomial in representatives:
        values, counts = np.unique(monomial, return_counts=True)
        multinomial = math.factorial(6)
        for count in counts:
            multinomial //= math.factorial(int(count))
        cells = [ALLOWED[int(value)] for value in values]
        edge = len(values) == 6 and (
            len({i for i, _ in cells}) == 6 or len({j for _, j in cells}) == 6
        )
        result.append(TARGET_CONSTANT * multinomial - int(edge))
    return np.asarray(result, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("BLOCK_SCS", "FULL_SCS"), default="BLOCK_SCS")
    parser.add_argument("--eps", type=float, default=2e-6)
    parser.add_argument("--max-iters", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("n6_sos_search.npz"))
    args = parser.parse_args()

    verify_ranking()
    group = variable_group()
    action2 = induced_m2_actions(group)
    vertices = vertex_representatives(group)
    print(f"group={len(group)} variables={NV} m2={len(M2)}", flush=True)

    degree_five_orbits, degree_five_reps, degree_five_sizes = monomial_orbits(5, group)
    print(
        f"degree-5 monomials={len(degree_five_orbits)} orbits={len(degree_five_reps)}",
        flush=True,
    )
    degree_six_orbits, degree_six_reps, degree_six_sizes = monomial_orbits(6, group)
    print(
        f"degree-6 monomials={len(degree_six_orbits)} orbits={len(degree_six_reps)}",
        flush=True,
    )
    if len(degree_five_reps) != 1292 or len(degree_six_reps) != 5605:
        raise AssertionError("unexpected coefficient-orbit counts")
    lift = multiplication_by_s(degree_five_orbits, degree_six_reps, len(degree_five_reps))
    target = target_coefficients(degree_six_reps)

    values_list = []
    q_orbits = []
    coefficient_matrices = []
    constraints = []
    block_sizes: list[list[int]] = []
    for number, vertex in enumerate(vertices):
        stabilizer_mask = group[:, vertex] == vertex
        stabilizer_actions = action2[stabilizer_mask]
        q_orbit = invariant_gram_orbits(stabilizer_actions)
        number_of_q_orbits = int(q_orbit.max()) + 1
        values = cp.Variable(number_of_q_orbits)
        gram = cp.reshape(values[q_orbit.ravel()], q_orbit.shape, order="C")
        if args.solver == "BLOCK_SCS":
            bases = isotypic_bases(q_orbit, number_of_q_orbits)
            constraints.extend(base.T @ gram @ base >> 0 for base in bases)
            sizes = sorted(base.shape[1] for base in bases)
        else:
            constraints.append(gram >> 0)
            sizes = [len(M2)]
        degree_five = vertex_degree_five_matrix(
            len(group), vertex, q_orbit, degree_five_orbits, degree_five_sizes
        )
        coefficient = (lift @ degree_five).tocsr()
        values_list.append(values)
        q_orbits.append(q_orbit)
        coefficient_matrices.append(coefficient)
        block_sizes.append(sizes)
        print(
            f"vertex {number + 1}/3 cell={ALLOWED[vertex]} orbit="
            f"{len(set(int(value) for value in group[:, vertex]))} "
            f"qvars={number_of_q_orbits} PSD-blocks={sizes}",
            flush=True,
        )

    margin = cp.Variable()
    produced = sum(
        (matrix @ values for matrix, values in zip(coefficient_matrices, values_list)),
        cp.Constant(np.zeros(len(degree_six_reps))),
    )
    constraints.append(produced + margin <= target)
    problem = cp.Problem(cp.Maximize(margin), constraints)
    result = problem.solve(
        solver=cp.SCS,
        verbose=True,
        eps=args.eps,
        max_iters=args.max_iters,
        acceleration_lookback=20,
        use_indirect=True,
    )
    print("status", problem.status, flush=True)
    print("objective", result, flush=True)
    if any(values.value is None for values in values_list):
        raise SystemExit("solver returned no Gram matrices")
    numerical_values = [np.asarray(values.value) for values in values_list]
    residual = target - sum(
        (matrix @ values for matrix, values in zip(coefficient_matrices, numerical_values)),
        np.zeros(len(target)),
    )
    minimum_eigenvalues = [
        float(np.linalg.eigvalsh(values[q_orbit]).min())
        for values, q_orbit in zip(numerical_values, q_orbits)
    ]
    print("minimum residual", float(residual.min()), flush=True)
    print("minimum eigenvalues", minimum_eigenvalues, flush=True)

    payload: dict[str, np.ndarray] = {
        "allowed": np.asarray(ALLOWED, dtype=np.int8),
        "group": group,
        "vertices": np.asarray(vertices, dtype=np.int16),
        "degree_six_representatives": degree_six_reps,
        "degree_six_sizes": degree_six_sizes,
        "target": target,
        "residual": residual,
        "margin": np.asarray(float(margin.value)),
        "minimum_eigenvalues": np.asarray(minimum_eigenvalues),
    }
    for number, (q_orbit, values) in enumerate(zip(q_orbits, numerical_values)):
        payload[f"q_orbit_{number}"] = q_orbit
        payload[f"q_values_{number}"] = values
        payload[f"block_sizes_{number}"] = np.asarray(block_sizes[number], dtype=np.int16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
