#!/usr/bin/env python3
"""Exact verifier for the n=6 Dittert certificate.

NumPy reads the archive and constructs finite orbit tables.  All certificate
coefficients, Gram products, residuals, and proof decisions use exact integer
arithmetic (the small NumPy indices cannot overflow their declared types).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


N = 6
ZEROS = {(0, 0), (1, 1)}
ALLOWED = tuple(
    (i, j) for i in range(N) for j in range(N) if (i, j) not in ZEROS
)
VARIABLES = len(ALLOWED)
GROUP_ORDER = 2304
TARGET_DENOMINATOR = 15_116_544
TARGET_NUMERATOR = 643


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def variable_group() -> np.ndarray:
    """Generate the automorphism group of the two-zero face."""
    index = {cell: i for i, cell in enumerate(ALLOWED)}
    actions: list[tuple[int, ...]] = []
    for swap in range(2):
        for lower_rows in itertools.permutations(range(2, N)):
            for lower_columns in itertools.permutations(range(2, N)):
                row = ([1, 0] if swap else [0, 1]) + list(lower_rows)
                column = ([1, 0] if swap else [0, 1]) + list(lower_columns)
                for transpose in range(2):
                    action = []
                    for i, j in ALLOWED:
                        image = (row[i], column[j])
                        if transpose:
                            image = image[::-1]
                        action.append(index[image])
                    actions.append(tuple(action))
    require(len(actions) == GROUP_ORDER, "wrong number of generated actions")
    require(len(set(actions)) == GROUP_ORDER, "generated actions are not unique")
    identity = tuple(range(VARIABLES))
    require(identity in actions, "identity is missing from the generated group")
    require(
        all(tuple(sorted(action)) == identity for action in actions),
        "a generated action is not a permutation",
    )
    return np.asarray(actions, dtype=np.int16)


def binomial_table(limit: int) -> np.ndarray:
    table = np.zeros((limit + 1, limit + 1), dtype=np.int64)
    for n in range(limit + 1):
        for k in range(n + 1):
            table[n, k] = math.comb(n, k)
    return table


BINOMIAL = binomial_table(VARIABLES + 6)


def multiset_count(degree: int) -> int:
    return math.comb(VARIABLES + degree - 1, degree)


def multiset_array(degree: int) -> np.ndarray:
    count = multiset_count(degree)
    flat = np.fromiter(
        (
            value
            for monomial in itertools.combinations_with_replacement(
                range(VARIABLES), degree
            )
            for value in monomial
        ),
        dtype=np.int16,
        count=count * degree,
    )
    require(flat.size == count * degree, "multiset generation ended early")
    return flat.reshape(count, degree)


def rank_multisets(monomials: np.ndarray) -> np.ndarray:
    monomials = np.asarray(monomials, dtype=np.int64)
    require(monomials.ndim == 2, "monomials must be a matrix")
    count, degree = monomials.shape
    strict = monomials + np.arange(degree, dtype=np.int64)
    universe = VARIABLES + degree - 1
    ranks = np.zeros(count, dtype=np.int64)
    previous = np.full(count, -1, dtype=np.int64)
    for position in range(degree):
        remaining = degree - position
        current = strict[:, position]
        ranks += BINOMIAL[universe - previous - 1, remaining]
        ranks -= BINOMIAL[universe - current, remaining]
        previous = current
    return ranks


def rank_multiset(monomial: tuple[int, ...]) -> int:
    degree = len(monomial)
    universe = VARIABLES + degree - 1
    result = 0
    previous = -1
    for position, value in enumerate(monomial):
        current = value + position
        remaining = degree - position
        result += math.comb(universe - previous - 1, remaining)
        result -= math.comb(universe - current, remaining)
        previous = current
    return result


def monomial_orbits(
    degree: int, group: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exhaustively partition every degree-d monomial into group orbits."""
    monomials = multiset_array(degree)
    orbit_ids = np.full(len(monomials), -1, dtype=np.int32)
    representatives: list[np.ndarray] = []
    for seed in range(len(monomials)):
        if orbit_ids[seed] >= 0:
            continue
        images = np.sort(group[:, monomials[seed]], axis=1)
        ranks = np.unique(rank_multisets(images))
        orbit_id = len(representatives)
        old = orbit_ids[ranks]
        require(
            bool(np.all((old == -1) | (old == orbit_id))),
            "monomial orbits overlap inconsistently",
        )
        orbit_ids[ranks] = orbit_id
        representatives.append(monomials[seed].copy())
    require(bool(np.all(orbit_ids >= 0)), "monomial orbit partition is incomplete")
    sizes = np.bincount(orbit_ids, minlength=len(representatives)).astype(np.int64)
    require(int(sizes.sum()) == len(monomials), "monomial orbit sizes do not add up")
    require(
        all(GROUP_ORDER % int(size) == 0 for size in sizes),
        "a monomial orbit size does not divide the group order",
    )
    return orbit_ids, np.asarray(representatives, dtype=np.int16), sizes


def multinomial_six(monomial: tuple[int, ...]) -> int:
    result = math.factorial(6)
    for variable in set(monomial):
        result //= math.factorial(monomial.count(variable))
    return result


def is_hyperedge(monomial: tuple[int, ...]) -> bool:
    if len(set(monomial)) != N:
        return False
    cells = [ALLOWED[variable] for variable in monomial]
    return len({row for row, _ in cells}) == N or len(
        {column for _, column in cells}
    ) == N


def hyperedges() -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
    index = {cell: variable for variable, cell in enumerate(ALLOWED)}
    row_edges = {
        tuple(sorted(index[(row, columns[row])] for row in range(N)))
        for columns in itertools.product(range(N), repeat=N)
        if all((row, columns[row]) in index for row in range(N))
    }
    column_edges = {
        tuple(sorted(index[(rows[column], column)] for column in range(N)))
        for rows in itertools.product(range(N), repeat=N)
        if all((rows[column], column) in index for column in range(N))
    }
    return row_edges, column_edges


def exact_gram(factor: np.ndarray) -> list[list[int]]:
    rows = [[int(value) for value in row] for row in factor]
    result = [[0] * VARIABLES for _ in range(VARIABLES)]
    for i in range(VARIABLES):
        for j in range(i, VARIABLES):
            value = sum(rows[i][k] * rows[j][k] for k in range(VARIABLES))
            result[i][j] = result[j][i] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "certificate",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("dittert_n6_exact_certificate.npz"),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    path = args.certificate.resolve()
    certificate_digest = sha256(path)
    archive = np.load(path, allow_pickle=False)
    require(
        set(archive.files)
        == {
            "denominator_exponent",
            "pattern",
            "allowed",
            "group",
            "multipliers",
            "factor_numerators",
            "search_result_sha256",
        },
        "unexpected certificate archive fields",
    )

    exponent = int(archive["denominator_exponent"])
    require(1 <= exponent < 62, "invalid denominator exponent")
    require(
        str(archive["pattern"])
        == "100000/010000/000000/000000/000000/000000",
        "wrong zero pattern",
    )
    require(
        np.array_equal(archive["allowed"], np.asarray(ALLOWED, dtype=np.int8)),
        "wrong variable list",
    )
    provenance = str(archive["search_result_sha256"])
    require(
        len(provenance) == 64 and all(c in "0123456789abcdef" for c in provenance),
        "invalid search-result provenance digest",
    )

    group = variable_group()
    require(np.array_equal(archive["group"], group), "wrong symmetry group")
    degree_four_ids, degree_four_representatives, degree_four_sizes = (
        monomial_orbits(4, group)
    )
    require(len(degree_four_ids) == math.comb(37, 4), "wrong degree-four count")
    require(
        len(degree_four_representatives) == 302,
        "wrong number of degree-four orbits",
    )
    require(
        int(degree_four_sizes.sum()) == math.comb(37, 4),
        "wrong degree-four orbit coverage",
    )
    multipliers = np.asarray(archive["multipliers"])
    require(
        np.array_equal(multipliers, degree_four_representatives),
        "multipliers are not the canonical degree-four orbit representatives",
    )

    degree_six_ids, degree_six_representatives, degree_six_sizes = monomial_orbits(
        6, group
    )
    require(len(degree_six_ids) == math.comb(39, 6), "wrong degree-six count")
    require(
        len(degree_six_representatives) == 5605,
        "wrong number of degree-six orbits",
    )

    factors = np.asarray(archive["factor_numerators"])
    require(
        factors.shape == (302, VARIABLES, VARIABLES),
        "wrong SOS-factor array shape",
    )
    require(np.issubdtype(factors.dtype, np.integer), "SOS factors are not integers")

    row_edges, column_edges = hyperedges()
    edges = row_edges | column_edges
    require(len(row_edges) == 32_400, "wrong row-transversal count")
    require(len(column_edges) == 32_400, "wrong column-transversal count")
    require(len(row_edges & column_edges) == 504, "wrong permanent count")
    require(len(edges) == 64_296, "wrong hyperedge count")

    produced = [0] * len(degree_six_representatives)
    for block, multiplier_array in enumerate(multipliers):
        multiplier = tuple(int(value) for value in multiplier_array)
        q_matrix = exact_gram(factors[block])
        for i in range(VARIABLES):
            for j in range(i, VARIABLES):
                q_value = q_matrix[i][j]
                if q_value == 0:
                    continue
                monomial = tuple(sorted(multiplier + (i, j)))
                rank = rank_multiset(monomial)
                orbit = int(degree_six_ids[rank])
                numerator = GROUP_ORDER * (1 if i == j else 2)
                divisor = int(degree_six_sizes[orbit])
                require(
                    numerator % divisor == 0,
                    "nonintegral orbit coefficient encountered",
                )
                produced[orbit] += (numerator // divisor) * q_value

    two_scale = 1 << (2 * exponent)
    common_denominator = TARGET_DENOMINATOR * two_scale
    minimum: int | None = None
    minimum_monomial: tuple[int, ...] | None = None
    maximum: int | None = None
    negative_target_orbits = 0
    for orbit, representative_array in enumerate(degree_six_representatives):
        monomial = tuple(int(value) for value in representative_array)
        edge = is_hyperedge(monomial)
        negative_target_orbits += int(edge)
        target = (
            TARGET_NUMERATOR * multinomial_six(monomial)
            - int(edge) * TARGET_DENOMINATOR
        ) * two_scale
        residual = target - TARGET_DENOMINATOR * produced[orbit]
        if minimum is None or residual < minimum:
            minimum = residual
            minimum_monomial = monomial
        if maximum is None or residual > maximum:
            maximum = residual
    require(negative_target_orbits == 148, "wrong number of hyperedge orbits")
    require(
        minimum is not None and minimum > 0,
        f"a residual coefficient is not positive: minimum={minimum}, "
        f"monomial={minimum_monomial}",
    )
    require(minimum_monomial is not None and maximum is not None, "missing residual extrema")

    evaluation_points = [tuple([1] * VARIABLES)]
    evaluation_points += [
        tuple((3 * i + 1) % 5 for i in range(VARIABLES)),
        tuple((i * i + 2 * i + 3) % 7 for i in range(VARIABLES)),
    ]
    for point in evaluation_points:
        matrix = [[0] * N for _ in range(N)]
        for value, (row, column) in zip(point, ALLOWED):
            matrix[row][column] = value
        row_product = math.prod(sum(row) for row in matrix)
        column_product = math.prod(
            sum(matrix[row][column] for row in range(N)) for column in range(N)
        )
        permanent = sum(
            math.prod(matrix[row][permutation[row]] for row in range(N))
            for permutation in itertools.permutations(range(N))
        )
        hypergraph = sum(math.prod(point[v] for v in edge) for edge in edges)
        require(
            hypergraph == row_product + column_product - permanent,
            "hypergraph identity failed an exact evaluation",
        )

    report = {
        "status": "CERTIFIED",
        "certificate_sha256": certificate_digest,
        "variables": VARIABLES,
        "symmetry_group_order": GROUP_ORDER,
        "degree_four_multiplier_orbits": len(degree_four_representatives),
        "degree_six_monomials": len(degree_six_ids),
        "degree_six_coefficient_orbits": len(degree_six_representatives),
        "hyperedges": len(edges),
        "minimum_residual_numerator": minimum,
        "common_denominator": common_denominator,
        "minimum_residual_decimal": minimum / common_denominator,
        "minimum_residual_monomial": list(minimum_monomial),
        "maximum_residual_numerator": maximum,
        "maximum_residual_decimal": maximum / common_denominator,
        "exact_evaluation_tests": len(evaluation_points),
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("Dittert n=6 exact certificate verifier")
    print("certificate SHA-256:", certificate_digest)
    print("symmetry group order:", GROUP_ORDER)
    print("sextic hyperedges in F:", len(edges))
    print("degree-four multiplier orbits:", len(degree_four_representatives))
    print("sextic monomials covered:", len(degree_six_ids))
    print("sextic coefficient orbits checked:", len(degree_six_representatives))
    print(
        f"minimum residual: {minimum}/{common_denominator} "
        f"= {minimum / common_denominator:.17g}"
    )
    print(
        f"maximum residual: {maximum}/{common_denominator} "
        f"= {maximum / common_denominator:.17g}"
    )
    print("minimum monomial:", minimum_monomial)
    print("CERTIFIED")


if __name__ == "__main__":
    main()
