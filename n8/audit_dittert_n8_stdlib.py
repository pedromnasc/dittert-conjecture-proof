#!/usr/bin/env python3
"""Independent standard-library audit of the n=8 Dittert proof.

This program imports no primary-verifier code.  It reconstructs the two-zero
polynomial from literal Ryser sums and reconstructs every marginal, support,
and exceptional proper-cut polynomial from exact scalar evaluations before
checking independent Bernstein certificates.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

Polynomial = list[Fraction]  # increasing power order

N = 8
M = N - 2
LOWER_BOUND = Fraction(12_249, 5_000_000)
RISK_LEFT = Fraction(1, 400)
RISK_RIGHT = Fraction(1, 75)


def gamma(order: int) -> Fraction:
    return Fraction(1) if order == 0 else Fraction(factorial(order), order**order)


def normalize(poly: Polynomial) -> Polynomial:
    result = poly[:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = [Fraction(0)] * max(len(left), len(right))
    for index, value in enumerate(left):
        result[index] += value
    for index, value in enumerate(right):
        result[index] += value
    return normalize(result)


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            result[i + j] += left_value * right_value
    return normalize(result)


def evaluate(poly: Polynomial, point: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(poly):
        result = result * point + coefficient
    return result


def interpolate(points: list[tuple[Fraction, Fraction]]) -> Polynomial:
    result: Polynomial = [Fraction(0)]
    for index, (x_value, y_value) in enumerate(points):
        basis: Polynomial = [Fraction(1)]
        denominator = Fraction(1)
        for other_index, (other_x, _) in enumerate(points):
            if other_index == index:
                continue
            if x_value == other_x:
                raise ValueError("interpolation points must be distinct")
            basis = multiply(basis, [-other_x, Fraction(1)])
            denominator *= x_value - other_x
        result = add(result, [value * y_value / denominator for value in basis])
    return normalize(result)


def interpolate_consecutive(values: list[Fraction]) -> Polynomial:
    rows = [values[:]]
    while len(rows[-1]) > 1:
        previous = rows[-1]
        rows.append(
            [previous[index + 1] - previous[index] for index in range(len(previous) - 1)]
        )
    forward = [row[0] for row in rows]
    result: Polynomial = [Fraction(0)]
    binomial_poly: Polynomial = [Fraction(1)]
    for order, coefficient in enumerate(forward):
        result = add(result, [coefficient * value for value in binomial_poly])
        binomial_poly = [
            value / (order + 1)
            for value in multiply(binomial_poly, [Fraction(-order), Fraction(1)])
        ]
    return normalize(result)


def affine_to_unit_interval(
    poly: Polynomial, left: Fraction, width: Fraction
) -> Polynomial:
    result = [Fraction(0)] * len(poly)
    for power, coefficient in enumerate(poly):
        for unit_power in range(power + 1):
            result[unit_power] += (
                coefficient
                * comb(power, unit_power)
                * left ** (power - unit_power)
                * width**unit_power
            )
    return normalize(result)


def power_to_bernstein(poly: Polynomial, degree: int) -> list[Fraction]:
    if len(poly) - 1 > degree:
        raise ValueError("Bernstein degree is too small")
    return [
        sum(
            Fraction(comb(index, power), comb(degree, power)) * poly[power]
            for power in range(min(index, len(poly) - 1) + 1)
        )
        for index in range(degree + 1)
    ]


def exact_permanent(matrix: list[list[Fraction]]) -> Fraction:
    order = len(matrix)
    if any(len(row) != order for row in matrix):
        raise ValueError("permanent requires a square matrix")
    total = Fraction(0)
    for mask in range(1, 1 << order):
        product = Fraction(1)
        for row in matrix:
            subset_sum = sum(
                (row[column] for column in range(order) if mask >> column & 1),
                Fraction(0),
            )
            product *= subset_sum
            if not product:
                break
        total += product if (order - mask.bit_count()) % 2 == 0 else -product
    return total


def two_zero_matrix(x_value: Fraction) -> list[list[Fraction]]:
    a_value = (Fraction(2 - M) + M * M * x_value) / 2
    b_value = (Fraction(1) - M * x_value) / 2
    matrix = [[Fraction(0) for _ in range(N)] for _ in range(N)]
    matrix[0][0] = a_value
    matrix[1][1] = a_value
    for index in range(2, N):
        matrix[0][index] = b_value
        matrix[1][index] = b_value
        matrix[index][0] = b_value
        matrix[index][1] = b_value
        for column in range(2, N):
            matrix[index][column] = x_value
    return matrix


def averaged_face_bracket(s_value: Fraction, p_value: Fraction) -> Fraction:
    x_value = (1 - s_value) / M
    return (
        x_value * x_value * (1 - M * s_value + M * M * p_value)
        + M * x_value * (s_value * s_value - (2 + M * s_value) * p_value)
        + M * (M - 1) * p_value * p_value
    )


def audit_equalization() -> None:
    step = Fraction(1, 100 * M * M)
    for s_value in (Fraction(0), Fraction(1, 2 * M), Fraction(1, M)):
        for p_value in (Fraction(0), Fraction(1, 4 * M * M)):
            derivative = (
                averaged_face_bracket(s_value, p_value + step)
                - averaged_face_bracket(s_value, p_value - step)
            ) / (2 * step)
            expected = (
                2 * M * (M - 1) * p_value
                + (M + 1) * s_value * s_value
                - M * s_value
                - 1
            )
            if derivative != expected:
                raise AssertionError("independent equalization identity failed")
    if Fraction(-1) + Fraction(2, M) + Fraction(4, M * M) >= 0:
        raise AssertionError("independent equalization sign failed")


def audit_two_zero_bound() -> tuple[int, Fraction]:
    face_left = Fraction(M - 2, M * M)
    face_right = Fraction(1, M)
    samples = []
    for index in range(N + 1):
        point = face_left + (face_right - face_left) * Fraction(index, N)
        samples.append((point, exact_permanent(two_zero_matrix(point))))
    permanent_poly = interpolate(samples)
    if len(permanent_poly) - 1 != N:
        raise AssertionError("Ryser interpolation has unexpected degree")
    difference = permanent_poly[:]
    difference[0] -= LOWER_BOUND
    subdivisions = 256
    width = (face_right - face_left) / subdivisions
    minimum: Fraction | None = None
    for subdivision in range(subdivisions):
        local = affine_to_unit_interval(
            difference, face_left + subdivision * width, width
        )
        coefficients = power_to_bernstein(local, N)
        if min(coefficients) <= 0:
            raise AssertionError(f"two-zero Bernstein failure at {subdivision}")
        minimum = min(coefficients) if minimum is None else min(minimum, *coefficients)
    if minimum is None:
        raise AssertionError("two-zero audit checked no coefficients")
    return subdivisions, minimum


def marginal_quantities(a_value: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    gamma_n = gamma(N)
    e_value = 1 - (2 - gamma_n) * a_value
    if e_value == 0:
        raise ValueError("marginal denominator vanishes")
    h_value = (1 - gamma_n) * a_value / e_value
    row_bound = (1 - a_value) * (1 + a_value / (N - 1)) ** (N - 1)
    column_bound = (1 + h_value) * (1 - h_value / (N - 1)) ** (N - 1)
    deficit = 2 - row_bound - column_bound
    contradiction = LOWER_BOUND * (1 - a_value) ** N - gamma_n + deficit
    common = Fraction((N - 1) ** (N - 1)) * e_value**N
    return deficit, contradiction, common


def audit_safe_marginal() -> tuple[int, Fraction, Fraction]:
    feasible_right = Fraction(1, 20)
    deficit_right, _, _ = marginal_quantities(feasible_right)
    if deficit_right <= gamma(N):
        raise AssertionError("D(1/20) does not exceed gamma_8")
    values = []
    for point in range(17):
        _, contradiction, common = marginal_quantities(Fraction(point))
        values.append(contradiction * common)
    numerator = interpolate_consecutive(values)
    if len(numerator) - 1 != 16:
        raise AssertionError("marginal interpolant has unexpected degree")
    for point in (Fraction(1, 3), Fraction(7, 5), Fraction(23, 7)):
        _, contradiction, common = marginal_quantities(point)
        if evaluate(numerator, point) != contradiction * common:
            raise AssertionError("marginal interpolation failed off-grid")
    intervals = ((Fraction(0), RISK_LEFT), (RISK_RIGHT, feasible_right))
    minimum: Fraction | None = None
    checked = 0
    for left, right in intervals:
        local = affine_to_unit_interval(numerator, left, right - left)
        coefficients = power_to_bernstein(local, 16)
        if min(coefficients) <= 0:
            raise AssertionError("safe marginal Bernstein failure")
        minimum = min(coefficients) if minimum is None else min(minimum, *coefficients)
        checked += len(coefficients)
    if minimum is None:
        raise AssertionError("safe marginal audit checked no coefficients")
    return checked, minimum, deficit_right - gamma(N)


def support_bound(zero_count: int) -> Fraction:
    staircase = (
        gamma(N - zero_count)
        * gamma(N - 1)
        / gamma(N - zero_count - 1)
    )
    return max(LOWER_BOUND, staircase)


def support_gap_value(a_value: Fraction, zero_count: int) -> tuple[Fraction, Fraction]:
    gamma_n = gamma(N)
    support_size = N - zero_count
    e_value = 1 - (2 - gamma_n) * a_value
    h_numerator = (1 - gamma_n) * a_value
    t_value = e_value - h_numerator / (N - 1)
    q_numerator = (2 - gamma_n) * e_value ** (N - 1) - t_value ** (N - 1)
    case_bound = support_bound(zero_count)
    r_numerator = (
        (2 - gamma_n + case_bound * (1 - a_value) ** N) * e_value**N
        - (e_value + h_numerator) * t_value ** (N - 1)
    )
    weighted_q = (
        (N - 1 + a_value) * q_numerator
        - (support_size - 1) * e_value ** (N - 1)
    )
    gap = (
        zero_count**zero_count * q_numerator ** (N - 1) * r_numerator
        - (1 - a_value)
        * e_value ** (N + (N - 1) * (support_size - 1))
        * weighted_q**zero_count
    )
    return q_numerator / e_value ** (N - 1), gap


def audit_support_gaps() -> tuple[int, int, Fraction]:
    h_right = (1 - gamma(N)) * RISK_RIGHT / (1 - (2 - gamma(N)) * RISK_RIGHT)
    if not 0 < h_right < N - 1:
        raise AssertionError("h leaves its monotonicity range")
    q_left, _ = support_gap_value(RISK_LEFT, 1)
    if q_left <= 1:
        raise AssertionError("q is not greater than one")
    cases = 0
    minimum: Fraction | None = None
    for zero_count in range(1, N - 1):
        values = [support_gap_value(Fraction(point), zero_count)[1] for point in range(66)]
        gap_poly = interpolate_consecutive(values)
        if len(gap_poly) - 1 != 65:
            raise AssertionError(f"support interpolant degree failure for z={zero_count}")
        for point in (Fraction(1, 3), Fraction(7, 5)):
            if evaluate(gap_poly, point) != support_gap_value(point, zero_count)[1]:
                raise AssertionError(f"support interpolation failed for z={zero_count}")
        local = affine_to_unit_interval(gap_poly, RISK_LEFT, RISK_RIGHT - RISK_LEFT)
        coefficients = power_to_bernstein(local, 65)
        if min(coefficients) <= 0:
            raise AssertionError(f"support Bernstein failure for z={zero_count}")
        minimum = min(coefficients) if minimum is None else min(minimum, *coefficients)
        cases += 1
    if minimum is None:
        raise AssertionError("support audit checked no coefficients")
    return cases, 66, minimum


SUBSET_CONSTANTS = {
    1: Fraction(1, 2),
    2: Fraction(3, 10),
    3: Fraction(1, 4),
    4: Fraction(6, 25),
    5: Fraction(1, 4),
    6: Fraction(3, 10),
    7: Fraction(1, 2),
}


def grouped_product(deviation: Fraction, subset_size: int) -> Fraction:
    return (1 + deviation / subset_size) ** subset_size * (
        1 - deviation / (N - subset_size)
    ) ** (N - subset_size)


def subset_quotient_value(
    deviation: Fraction, subset_size: int, constant: Fraction
) -> Fraction:
    if deviation == 0:
        raise ValueError("use a nonzero interpolation point")
    return (
        1
        - grouped_product(deviation, subset_size)
        - constant * deviation * deviation
    ) / (deviation * deviation)


def audit_subset_deficits() -> tuple[int, Fraction, Fraction]:
    minimum: Fraction | None = None
    endpoint_margin: Fraction | None = None
    for subset_size, constant in SUBSET_CONSTANTS.items():
        samples = [
            (
                Fraction(point),
                subset_quotient_value(Fraction(point), subset_size, constant),
            )
            for point in range(1, 8)
        ]
        quotient = interpolate(samples)
        if len(quotient) - 1 != 6:
            raise AssertionError("subset quotient has unexpected degree")
        for point in (Fraction(1, 3), Fraction(-2, 5)):
            if evaluate(quotient, point) != subset_quotient_value(
                point, subset_size, constant
            ):
                raise AssertionError("subset quotient interpolation failed")
        local = affine_to_unit_interval(
            quotient, Fraction(-1, 10), Fraction(1, 5)
        )
        coefficients = power_to_bernstein(local, 6)
        if min(coefficients) <= 0:
            raise AssertionError(f"subset Bernstein failure for k={subset_size}")
        minimum = min(coefficients) if minimum is None else min(minimum, *coefficients)
        for endpoint in (Fraction(-1, 10), Fraction(1, 10)):
            margin = 1 - grouped_product(endpoint, subset_size) - gamma(N)
            if margin <= 0:
                raise AssertionError("subset localization endpoint failed")
            endpoint_margin = (
                margin if endpoint_margin is None else min(endpoint_margin, margin)
            )
    if minimum is None or endpoint_margin is None:
        raise AssertionError("subset audit checked no coefficients")
    return len(SUBSET_CONSTANTS), minimum, endpoint_margin


def special_value(s_value: Fraction, u: int) -> Fraction:
    v_value = N - 1 - u
    effective = (
        SUBSET_CONSTANTS[u]
        * SUBSET_CONSTANTS[v_value]
        / (SUBSET_CONSTANTS[u] + SUBSET_CONSTANTS[v_value])
    )
    staircase = gamma(N - u) * gamma(N - v_value)
    return staircase * (1 - s_value) ** N - gamma(N) + effective * s_value**2


def audit_proper_cuts() -> tuple[int, int, Fraction, Fraction]:
    gamma_n = gamma(N)
    if N * gamma_n >= 1 - gamma_n or 2401 * gamma_n >= 6:
        raise AssertionError("a proper-cut interval bound failed")
    generic = 0
    special = 0
    generic_minimum: Fraction | None = None
    special_minimum: Fraction | None = None
    for u in range(1, N):
        for v_value in range(1, N - u):
            k = N - u - v_value
            staircase = gamma(N - u) * gamma(N - v_value) / gamma(k)
            count = Fraction(factorial(N - v_value) * factorial(N - u), factorial(k))
            weight = (
                Fraction(1, (N - v_value) ** u)
                * Fraction(1, (N - u) ** v_value)
                * Fraction(k**k, ((N - u) * (N - v_value)) ** k)
            )
            if count * weight != staircase:
                raise AssertionError("staircase count failed")
            eta = (
                staircase
                - gamma_n
                - Fraction(N**3, 4 * k * k) * staircase * staircase / (1 - gamma_n)
            )
            if eta > 0:
                generic += 1
                generic_minimum = eta if generic_minimum is None else min(generic_minimum, eta)
                continue
            if k != 1:
                raise AssertionError("unexpected proper-cut obstruction")
            special += 1

    for u in range(1, 4):
        values = [special_value(Fraction(point), u) for point in range(9)]
        polynomial = interpolate_consecutive(values)
        if len(polynomial) - 1 != 8:
            raise AssertionError("special-cut interpolant degree failure")
        if evaluate(polynomial, Fraction(1, 3)) != special_value(Fraction(1, 3), u):
            raise AssertionError("special-cut interpolation failed")
        coefficients: list[Fraction] = []
        for half in range(2):
            local = affine_to_unit_interval(
                polynomial, Fraction(half, 14), Fraction(1, 14)
            )
            coefficients.extend(power_to_bernstein(local, 8))
        if min(coefficients) <= 0:
            raise AssertionError(f"special-cut Bernstein failure for u={u}")
        special_minimum = (
            min(coefficients)
            if special_minimum is None
            else min(special_minimum, *coefficients)
        )
    if special != 6:
        raise AssertionError("unexpected number of special cuts")
    if generic_minimum is None or special_minimum is None:
        raise AssertionError("proper-cut audit checked no coefficients")
    return generic, special, generic_minimum, special_minimum


def main() -> None:
    if LOWER_BOUND <= gamma(N):
        raise AssertionError("the two-zero bound does not exceed gamma_8")
    audit_equalization()
    subdivisions, face_minimum = audit_two_zero_bound()
    safe_count, safe_minimum, deficit_margin = audit_safe_marginal()
    support_cases, support_samples, support_minimum = audit_support_gaps()
    subset_cases, subset_minimum, endpoint_margin = audit_subset_deficits()
    generic, special, generic_minimum, special_minimum = audit_proper_cuts()
    print("Dittert n=8 independent standard-library audit")
    print("exact Ryser interpolation samples =", N + 1)
    print("two-zero Bernstein subintervals =", subdivisions)
    print("smallest two-zero Bernstein coefficient =", face_minimum)
    print("safe marginal Bernstein coefficients =", safe_count)
    print("smallest safe marginal Bernstein coefficient =", safe_minimum)
    print("D(1/20)-gamma_8 =", deficit_margin)
    print("support cases checked =", support_cases)
    print("exact samples per support polynomial =", support_samples)
    print("smallest support Bernstein coefficient =", support_minimum)
    print("subset-deficit cases checked =", subset_cases)
    print("smallest subset Bernstein coefficient =", subset_minimum)
    print("smallest subset endpoint margin =", endpoint_margin)
    print("generic proper cuts checked =", generic)
    print("special proper cuts checked =", special)
    print("smallest generic eta =", generic_minimum)
    print("smallest special Bernstein coefficient =", special_minimum)
    print("INDEPENDENT AUDIT CERTIFIED")


if __name__ == "__main__":
    main()
