#!/usr/bin/env python3
"""Exact verifier for the n=5 Dittert certificate.

NumPy is used only to read the certificate archive.  Every proof decision and
every polynomial coefficient calculation uses Python arbitrary-precision
integers.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


if not __debug__:
    raise RuntimeError("Run without Python -O; the verifier uses assertions.")


N = 5
POWER_FIVE = 5**9
EXPECTED_MINIMUM = 7_628_882_599_067_611_080
EXPECTED_MINIMUM_MONOMIAL = (0, 0, 4, 10, 10)
EXPECTED_MAXIMUM = 7_743_319_887_559_506_820


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gram(factor: np.ndarray) -> list[list[int]]:
    size = factor.shape[0]
    rows = [[int(factor[i, j]) for j in range(i + 1)] for i in range(size)]
    result = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = sum(rows[i][k] * rows[j][k] for k in range(j + 1))
            result[i][j] = result[j][i] = value
    return result


def multinomial_five(monomial: tuple[int, ...]) -> int:
    result = math.factorial(5)
    for variable in set(monomial):
        result //= math.factorial(monomial.count(variable))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "certificate",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("dittert_n5_exact_certificate.npz"),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    path = args.certificate.resolve()
    archive = np.load(path, allow_pickle=False)
    assert set(archive.files) == {
        "denominator_exponent",
        "pattern",
        "allowed",
        "group",
        "vertex_representatives",
        "m2",
        "Lvertex_num",
    }

    exponent = int(archive["denominator_exponent"])
    assert exponent == 28
    assert str(archive["pattern"]) == "10000/01000/00000/00000/00000"
    allowed = [tuple(map(int, cell)) for cell in archive["allowed"]]
    expected_allowed = [
        (i, j) for i in range(N) for j in range(N) if (i, j) not in {(0, 0), (1, 1)}
    ]
    assert allowed == expected_allowed
    variables = len(allowed)
    assert variables == 23

    group = [tuple(map(int, action)) for action in archive["group"]]
    identity = tuple(range(variables))
    group_set = set(group)
    assert len(group) == len(group_set) == 144 and identity in group_set
    for action in group:
        assert tuple(sorted(action)) == identity
        transformed = {allowed[action[i]] for i in range(variables)}
        assert transformed == set(allowed)
        for other in group:
            assert tuple(other[action[i]] for i in range(variables)) in group_set

    m2 = [tuple(map(int, pair)) for pair in archive["m2"]]
    assert m2 == list(itertools.combinations_with_replacement(range(variables), 2))
    assert len(m2) == 276
    vertex_representatives = tuple(map(int, archive["vertex_representatives"]))
    assert vertex_representatives == (0, 1, 10)
    unseen = set(range(variables))
    vertex_orbits = []
    while unseen:
        seed = next(iter(unseen))
        orbit = {action[seed] for action in group}
        vertex_orbits.append(orbit)
        unseen -= orbit
    assert sorted(map(len, vertex_orbits)) == [2, 9, 12]
    assert len({next(i for i, orbit in enumerate(vertex_orbits) if v in orbit) for v in vertex_representatives}) == 3

    factors = np.asarray(archive["Lvertex_num"])
    assert factors.shape == (3, len(m2), len(m2))
    assert np.issubdtype(factors.dtype, np.integer)
    assert np.all(np.triu(factors, 1) == 0)
    assert np.all(np.diagonal(factors, axis1=1, axis2=2) > 0)

    edges = set()
    for support in itertools.combinations(range(variables), 5):
        cells = [allowed[v] for v in support]
        if len({i for i, _ in cells}) == 5 or len({j for _, j in cells}) == 5:
            edges.add(support)
    assert len(edges) == 3922
    for action in group:
        assert {tuple(sorted(action[v] for v in edge)) for edge in edges} == edges

    grams = [gram(factor) for factor in factors]
    sos: defaultdict[tuple[int, ...], int] = defaultdict(int)
    for number, vertex in enumerate(vertex_representatives):
        q_matrix = grams[number]
        for action in group:
            transformed_m2 = [
                (action[first], action[second]) for first, second in m2
            ]
            transformed_vertex = action[vertex]
            for i in range(len(m2)):
                left = transformed_m2[i]
                row = q_matrix[i]
                for j in range(i, len(m2)):
                    coefficient = row[j] * (1 if i == j else 2)
                    if coefficient:
                        right = transformed_m2[j]
                        monomial = tuple(
                            sorted((transformed_vertex, left[0], left[1], right[0], right[1]))
                        )
                        sos[monomial] += coefficient

    two_scale = 1 << (2 * exponent)
    common_denominator = POWER_FIVE * two_scale
    minimum = None
    minimum_monomial = None
    maximum = None
    residual: dict[tuple[int, ...], int] = {}
    for monomial in itertools.combinations_with_replacement(range(variables), 5):
        edge = len(set(monomial)) == 5 and monomial in edges
        target = (1226 * multinomial_five(monomial) - int(edge) * POWER_FIVE) * two_scale
        value = target - POWER_FIVE * sos.get(monomial, 0)
        residual[monomial] = value
        if minimum is None or value < minimum:
            minimum = value
            minimum_monomial = monomial
        if maximum is None or value > maximum:
            maximum = value
    assert len(residual) == math.comb(variables + 4, 5) == 80_730
    assert minimum is not None and minimum > 0
    assert minimum_monomial is not None and maximum is not None
    assert minimum == EXPECTED_MINIMUM
    assert minimum_monomial == EXPECTED_MINIMUM_MONOMIAL
    assert maximum == EXPECTED_MAXIMUM

    # Direct checks that the hypergraph polynomial is row product + column
    # product - permanent, independent of the coefficient construction above.
    evaluation_points = [tuple([1] * variables)]
    evaluation_points += [
        tuple((3 * i + 1) % 5 for i in range(variables)),
        tuple((i * i + 2 * i + 3) % 7 for i in range(variables)),
    ]
    for point in evaluation_points:
        matrix = [[0] * N for _ in range(N)]
        for value, (row, column) in zip(point, allowed):
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
        assert hypergraph == row_product + column_product - permanent

    report = {
        "status": "CERTIFIED",
        "certificate_sha256": sha256(path),
        "variables": variables,
        "symmetry_group_order": len(group),
        "quintic_hyperedges": len(edges),
        "quintic_monomials": len(residual),
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
    print("Dittert n=5 exact certificate verifier")
    print("certificate SHA-256:", report["certificate_sha256"])
    print("quintic hyperedges in F:", len(edges))
    print("quintic monomials checked:", len(residual))
    print(f"minimum residual: {minimum}/{common_denominator} = {minimum / common_denominator:.17g}")
    print(f"maximum residual: {maximum}/{common_denominator} = {maximum / common_denominator:.17g}")
    print("minimum monomial:", minimum_monomial)
    print("CERTIFIED")


if __name__ == "__main__":
    main()
