#!/usr/bin/env python3
"""Independent standard-library audit of the n=9 Dittert proof.

This program does not import the primary verifier.  It reconstructs the
two-zero polynomial from literal Ryser evaluations, the marginal numerator
from exact scalar evaluations, and each degree-82 support polynomial from 83
exact values.  Positivity is then checked in exact Bernstein bases, rather
than with the primary verifier's Sturm/support expansions.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

Polynomial = list[Fraction]  # coefficients in increasing powers

N = 9
M = N - 2
LOWER_BOUND = Fraction(47_533, 50_000_000)


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
    """Interpolate f(0),...,f(d) in the Newton forward-difference basis."""
    difference_rows = [values[:]]
    while len(difference_rows[-1]) > 1:
        previous = difference_rows[-1]
        difference_rows.append(
            [previous[index + 1] - previous[index] for index in range(len(previous) - 1)]
        )
    forward = [row[0] for row in difference_rows]
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
    if not (
        Fraction(-1) < 0
        and Fraction(-1) + Fraction(2, M) + Fraction(4, M * M) < 0
    ):
        raise AssertionError("independent equalization sign failed")


def audit_two_zero_bound() -> tuple[int, Fraction]:
    face_left = Fraction(M - 2, M * M)
    face_right = Fraction(1, M)
    samples = [
        (
            face_left + (face_right - face_left) * Fraction(index, N),
            Fraction(0),
        )
        for index in range(N + 1)
    ]
    samples = [(point, exact_permanent(two_zero_matrix(point))) for point, _ in samples]
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
        local_minimum = min(coefficients)
        minimum = local_minimum if minimum is None else min(minimum, local_minimum)
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
    feasible_right = Fraction(3, 100)
    deficit_right, _, _ = marginal_quantities(feasible_right)
    if deficit_right <= gamma(N):
        raise AssertionError("D(3/100) does not exceed gamma_9")
    values = []
    for point in range(19):
        _, contradiction, common = marginal_quantities(Fraction(point))
        values.append(contradiction * common)
    numerator = interpolate_consecutive(values)
    if len(numerator) - 1 != 18:
        raise AssertionError("marginal interpolant has unexpected degree")
    for point in (Fraction(1, 3), Fraction(7, 5), Fraction(23, 7)):
        _, contradiction, common = marginal_quantities(point)
        if evaluate(numerator, point) != contradiction * common:
            raise AssertionError("marginal interpolation failed off-grid")
    intervals = ((Fraction(0), Fraction(1, 500)), (Fraction(1, 200), feasible_right))
    minimum: Fraction | None = None
    checked = 0
    for left, right in intervals:
        local = affine_to_unit_interval(numerator, left, right - left)
        coefficients = power_to_bernstein(local, 18)
        if min(coefficients) <= 0:
            raise AssertionError("safe marginal Bernstein failure")
        minimum = min(coefficients) if minimum is None else min(minimum, *coefficients)
        checked += len(coefficients)
    if minimum is None:
        raise AssertionError("safe marginal audit checked no coefficients")
    return checked, minimum, deficit_right - gamma(N)


def q_and_support_gap(a_value: Fraction, zero_count: int) -> tuple[Fraction, Fraction]:
    """Return q and the scaled support-gap polynomial evaluated at a."""
    gamma_n = gamma(N)
    support_size = N - zero_count
    e_value = 1 - (2 - gamma_n) * a_value
    h_numerator = (1 - gamma_n) * a_value
    t_value = e_value - h_numerator / (N - 1)
    q_numerator = (2 - gamma_n) * e_value**8 - t_value**8
    r_numerator = (
        (2 - gamma_n + LOWER_BOUND * (1 - a_value) ** N) * e_value**9
        - (e_value + h_numerator) * t_value**8
    )
    weighted_q = (8 + a_value) * q_numerator - (support_size - 1) * e_value**8
    gap = (
        zero_count**zero_count * q_numerator**8 * r_numerator
        - (1 - a_value)
        * e_value ** (9 + 8 * (support_size - 1))
        * weighted_q**zero_count
    )
    return q_numerator / e_value**8, gap


def audit_support_gaps() -> tuple[int, int, Fraction]:
    risk_left = Fraction(1, 500)
    risk_right = Fraction(1, 200)
    h_right = (1 - gamma(N)) * risk_right / (1 - (2 - gamma(N)) * risk_right)
    if not 0 < h_right < N - 1:
        raise AssertionError("h leaves the monotonicity range")
    q_left, _ = q_and_support_gap(risk_left, 1)
    if q_left <= 1:
        raise AssertionError("q is not greater than one")
    cases = 0
    minimum: Fraction | None = None
    for zero_count in range(1, N - 1):
        values = [q_and_support_gap(Fraction(point), zero_count)[1] for point in range(83)]
        gap_poly = interpolate_consecutive(values)
        if len(gap_poly) - 1 != 82:
            raise AssertionError(f"support interpolant degree failure for z={zero_count}")
        for point in (Fraction(1, 3), Fraction(7, 5)):
            if evaluate(gap_poly, point) != q_and_support_gap(point, zero_count)[1]:
                raise AssertionError(f"support interpolation failed for z={zero_count}")
        local = affine_to_unit_interval(gap_poly, risk_left, risk_right - risk_left)
        coefficients = power_to_bernstein(local, 82)
        if min(coefficients) <= 0:
            raise AssertionError(f"support Bernstein failure for z={zero_count}")
        minimum = min(coefficients) if minimum is None else min(minimum, *coefficients)
        cases += 1
    if minimum is None:
        raise AssertionError("support audit checked no coefficients")
    return cases, 83, minimum


def audit_proper_cuts() -> tuple[int, tuple[int, int, int, Fraction], Fraction]:
    gamma_n = gamma(N)
    if N * gamma_n >= 1 - gamma_n:
        raise AssertionError("proper-cut square-root bound is unavailable")
    checked = 0
    minimum_eta: Fraction | None = None
    minimum_case: tuple[int, int, int, Fraction] | None = None
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
                raise AssertionError(f"staircase count failed at u={u}, v={v_value}")
            eta = (
                staircase
                - gamma_n
                - Fraction(N**3, 4 * k * k) * staircase * staircase / (1 - gamma_n)
            )
            if eta <= 0:
                raise AssertionError(f"proper-cut failure at u={u}, v={v_value}")
            checked += 1
            if minimum_eta is None or eta < minimum_eta:
                minimum_eta = eta
                minimum_case = (u, v_value, k, staircase)
    if minimum_eta is None or minimum_case is None:
        raise AssertionError("proper-cut audit checked no cases")
    return checked, minimum_case, minimum_eta


def main() -> None:
    if LOWER_BOUND <= gamma(N):
        raise AssertionError("the two-zero bound does not exceed gamma_9")
    audit_equalization()
    subdivisions, face_minimum = audit_two_zero_bound()
    safe_count, safe_minimum, deficit_margin = audit_safe_marginal()
    support_cases, support_samples, support_minimum = audit_support_gaps()
    proper_count, proper_case, proper_eta = audit_proper_cuts()
    print("Dittert n=9 independent standard-library audit")
    print("exact Ryser interpolation samples =", N + 1)
    print("two-zero Bernstein subintervals =", subdivisions)
    print("smallest two-zero Bernstein coefficient =", face_minimum)
    print("safe marginal Bernstein coefficients =", safe_count)
    print("smallest safe marginal Bernstein coefficient =", safe_minimum)
    print("D(3/100)-gamma_9 =", deficit_margin)
    print("support cases checked =", support_cases)
    print("exact samples per support polynomial =", support_samples)
    print("smallest support Bernstein coefficient =", support_minimum)
    print("proper cuts checked =", proper_count)
    print("smallest proper eta case =", proper_case)
    print("smallest proper eta =", proper_eta)
    print("INDEPENDENT AUDIT CERTIFIED")


if __name__ == "__main__":
    main()
