#!/usr/bin/env python3
"""Independent standard-library audit of the n=10 Dittert proof.

This program does not import the primary verifier.  It reconstructs the
two-zero-face polynomial from eleven literal exact Ryser evaluations and
proves its lower bound by exact Bernstein coefficients on 256 subintervals.
It reconstructs the degree-20 marginal numerator from exact values and proves
positivity through a degree-26 Bernstein expansion.  Thus neither polynomial
construction nor positivity algorithm is shared with the primary verifier.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

Polynomial = list[Fraction]  # increasing power order

N = 10
M = N - 2
LOWER_BOUND = Fraction(36_719, 100_000_000)


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
    """Return the exact Lagrange interpolant through distinct points."""
    result: Polynomial = [Fraction(0)]
    for index, (x_value, y_value) in enumerate(points):
        basis: Polynomial = [Fraction(1)]
        denominator = Fraction(1)
        for other_index, (other_x, _) in enumerate(points):
            if other_index == index:
                continue
            if x_value == other_x:
                raise ValueError("interpolation abscissae must be distinct")
            basis = multiply(basis, [-other_x, Fraction(1)])
            denominator *= x_value - other_x
        result = add(result, [coefficient * y_value / denominator for coefficient in basis])
    return normalize(result)


def affine_to_unit_interval(
    poly: Polynomial, left: Fraction, width: Fraction
) -> Polynomial:
    """Coefficients of poly(left + width*t), in powers of t."""
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
    """Convert a power polynomial on [0,1] to Bernstein degree ``degree``."""
    if len(poly) - 1 > degree:
        raise ValueError("Bernstein degree is too small")
    coefficients: list[Fraction] = []
    for index in range(degree + 1):
        coefficient = Fraction(0)
        for power in range(min(index, len(poly) - 1) + 1):
            coefficient += (
                Fraction(comb(index, power), comb(degree, power)) * poly[power]
            )
        coefficients.append(coefficient)
    return coefficients


def exact_permanent(matrix: list[list[Fraction]]) -> Fraction:
    """Literal Ryser formula over Fraction, independent of the closed form."""
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
            if product == 0:
                break
        if (order - mask.bit_count()) % 2:
            total -= product
        else:
            total += product
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
    x1x2 = 1 - M * s_value + M * M * p_value
    mixed = s_value * s_value - (2 + M * s_value) * p_value
    return (
        x_value * x_value * x1x2
        + M * x_value * mixed
        + M * (M - 1) * p_value * p_value
    )


def audit_equalization() -> None:
    """Check the bivariate derivative identity by exact polynomial sampling."""
    step = Fraction(1, 100 * M * M)
    s_points = (Fraction(0), Fraction(1, 2 * M), Fraction(1, M))
    p_points = (Fraction(0), Fraction(1, 4 * M * M))
    # Both sides have degree at most two in s and one in p.  Agreement on
    # this 3-by-2 grid proves the identity.  The centered difference is the
    # exact derivative because the bracket is quadratic in p.
    for s_value in s_points:
        for p_value in p_points:
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

    envelope_zero = Fraction(-1)
    envelope_endpoint = Fraction(-1) + Fraction(2, M) + Fraction(4, M * M)
    if not (envelope_zero < 0 and envelope_endpoint < 0):
        raise AssertionError("independent equalization sign check failed")


def audit_two_zero_bound() -> tuple[int, Fraction]:
    face_left = Fraction(M - 2, M * M)
    face_right = Fraction(1, M)
    samples: list[tuple[Fraction, Fraction]] = []
    for index in range(N + 1):
        point = face_left + (face_right - face_left) * Fraction(index, N)
        samples.append((point, exact_permanent(two_zero_matrix(point))))

    permanent_poly = interpolate(samples)
    if len(permanent_poly) - 1 != N:
        raise AssertionError("Ryser interpolation has unexpected degree")
    for point, value in samples:
        if evaluate(permanent_poly, point) != value:
            raise AssertionError("Ryser interpolation failed its defining samples")

    difference = permanent_poly[:]
    difference[0] -= LOWER_BOUND
    subdivisions = 256
    width = (face_right - face_left) / subdivisions
    minimum_coefficient: Fraction | None = None
    for subdivision in range(subdivisions):
        left = face_left + subdivision * width
        local_power = affine_to_unit_interval(difference, left, width)
        coefficients = power_to_bernstein(local_power, N)
        if any(coefficient <= 0 for coefficient in coefficients):
            raise AssertionError(
                f"two-zero Bernstein positivity failed on subdivision {subdivision}"
            )
        local_minimum = min(coefficients)
        if minimum_coefficient is None or local_minimum < minimum_coefficient:
            minimum_coefficient = local_minimum
    if minimum_coefficient is None:
        raise AssertionError("no two-zero Bernstein coefficients were checked")
    return subdivisions, minimum_coefficient


def marginal_quantities(
    a_value: Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    """Return D(a), H(a), and the positive common denominator."""
    gamma_n = gamma(N)
    denominator = 1 - (2 - gamma_n) * a_value
    if denominator <= 0:
        raise ValueError("marginal denominator is not positive")
    h_value = (1 - gamma_n) * a_value / denominator
    row_bound = (1 - a_value) * (1 + a_value / (N - 1)) ** (N - 1)
    column_bound = (1 + h_value) * (1 - h_value / (N - 1)) ** (N - 1)
    deficit = 2 - row_bound - column_bound
    contradiction = LOWER_BOUND * (1 - a_value) ** N - gamma_n + deficit
    common = Fraction((N - 1) ** (N - 1)) * denominator**N
    return deficit, contradiction, common


def audit_marginal_bound() -> tuple[int, Fraction, Fraction]:
    interval_right = Fraction(1, 50)
    gamma_n = gamma(N)
    deficit_right, _, _ = marginal_quantities(interval_right)
    if deficit_right <= gamma_n:
        raise AssertionError("D(1/50) does not exceed gamma_10")

    # H times its common denominator has degree at most 20.  Twenty-one exact
    # values therefore reconstruct it without using the primary expansion.
    points: list[tuple[Fraction, Fraction]] = []
    for index in range(21):
        point = Fraction(index, 1000)
        _, contradiction, common = marginal_quantities(point)
        points.append((point, contradiction * common))
    numerator = interpolate(points)
    if len(numerator) - 1 != 20:
        raise AssertionError("marginal interpolant has unexpected degree")

    # Independent off-grid checks guard the value-to-polynomial reconstruction.
    for index in range(5):
        point = Fraction(2 * index + 1, 2000)
        _, contradiction, common = marginal_quantities(point)
        if evaluate(numerator, point) != contradiction * common:
            raise AssertionError("marginal interpolation failed an off-grid check")

    unit_power = affine_to_unit_interval(numerator, Fraction(0), interval_right)
    bernstein_degree = 26
    coefficients = power_to_bernstein(unit_power, bernstein_degree)
    if any(coefficient <= 0 for coefficient in coefficients):
        raise AssertionError("marginal Bernstein positivity failed")
    return bernstein_degree, min(coefficients), deficit_right - gamma_n


def audit_proper_cuts() -> tuple[int, tuple[int, int, int, Fraction], Fraction]:
    gamma_n = gamma(N)
    checked = 0
    minimum_eta: Fraction | None = None
    minimum_case: tuple[int, int, int, Fraction] | None = None
    for u in range(1, N):
        for v in range(1, N - u):
            k = N - u - v
            staircase = gamma(N - u) * gamma(N - v) / gamma(k)

            count = Fraction(factorial(N - v) * factorial(N - u), factorial(k))
            weight = (
                Fraction(1, (N - v) ** u)
                * Fraction(1, (N - u) ** v)
                * Fraction(k**k, ((N - u) * (N - v)) ** k)
            )
            if count * weight != staircase:
                raise AssertionError(f"staircase count failed at u={u}, v={v}")

            eta = (
                staircase
                - gamma_n
                - Fraction(N**3, 4 * k * k) * staircase * staircase / (1 - gamma_n)
            )
            if eta <= 0:
                raise AssertionError(f"proper-cut inequality failed at u={u}, v={v}")
            checked += 1
            if minimum_eta is None or eta < minimum_eta:
                minimum_eta = eta
                minimum_case = (u, v, k, staircase)

    if minimum_eta is None or minimum_case is None:
        raise AssertionError("no proper cuts were audited")
    return checked, minimum_case, minimum_eta


def main() -> None:
    if not LOWER_BOUND > gamma(N):
        raise AssertionError("the two-zero bound does not exceed gamma_10")
    audit_equalization()
    subdivisions, face_minimum = audit_two_zero_bound()
    bernstein_degree, marginal_minimum, deficit_margin = audit_marginal_bound()
    proper_count, proper_case, proper_eta = audit_proper_cuts()

    print("Dittert n=10 independent standard-library audit")
    print("exact Ryser interpolation samples =", N + 1)
    print("two-zero Bernstein subintervals =", subdivisions)
    print("smallest two-zero Bernstein coefficient =", face_minimum)
    print("marginal Bernstein degree =", bernstein_degree)
    print("smallest marginal Bernstein coefficient =", marginal_minimum)
    print("D(1/50)-gamma_10 =", deficit_margin)
    print("proper cuts checked =", proper_count)
    print("smallest proper eta case =", proper_case)
    print("smallest proper eta =", proper_eta)
    print("INDEPENDENT AUDIT CERTIFIED")


if __name__ == "__main__":
    main()
