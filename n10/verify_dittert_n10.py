#!/usr/bin/env python3
"""Primary exact verifier for the n=10 Dittert working proof.

External mathematical inputs (not reproved here) are listed precisely in the
proof note.  Everything dimension-specific is checked with Python's exact
integer and Fraction arithmetic.  Floating-point values are used only for
human-readable output.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Iterable, Sequence

Poly = tuple[Fraction, ...]  # coefficients in increasing powers
BiPoly = dict[tuple[int, int], Fraction]  # coefficients of s^i p^j


def trim(poly: Sequence[Fraction]) -> Poly:
    result = list(poly)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result) if result else (Fraction(0),)


def poly_add(left: Poly, right: Poly) -> Poly:
    result = [Fraction(0) for _ in range(max(len(left), len(right)))]
    for index, coefficient in enumerate(left):
        result[index] += coefficient
    for index, coefficient in enumerate(right):
        result[index] += coefficient
    return trim(result)


def poly_sub(left: Poly, right: Poly) -> Poly:
    return poly_add(left, tuple(-coefficient for coefficient in right))


def poly_mul(left: Poly, right: Poly) -> Poly:
    result = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
    for i, x_value in enumerate(left):
        for j, y_value in enumerate(right):
            result[i + j] += x_value * y_value
    return trim(result)


def poly_scale(poly: Poly, scalar: Fraction) -> Poly:
    return trim(tuple(scalar * coefficient for coefficient in poly))


def poly_pow(poly: Poly, exponent: int) -> Poly:
    if exponent < 0:
        raise ValueError("negative exponent")
    result: Poly = (Fraction(1),)
    base = poly
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = poly_mul(result, base)
        base = poly_mul(base, base)
        remaining >>= 1
    return result


def poly_derivative(poly: Poly) -> Poly:
    if len(poly) <= 1:
        return (Fraction(0),)
    return trim(tuple(Fraction(index) * poly[index] for index in range(1, len(poly))))


def poly_eval(poly: Poly, value: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(poly):
        result = result * value + coefficient
    return result


def poly_divmod(dividend: Poly, divisor: Poly) -> tuple[Poly, Poly]:
    dividend = trim(dividend)
    divisor = trim(divisor)
    if divisor == (0,):
        raise ZeroDivisionError("polynomial division by zero")
    if len(dividend) < len(divisor):
        return (Fraction(0),), dividend

    remainder = list(dividend)
    quotient = [Fraction(0) for _ in range(len(dividend) - len(divisor) + 1)]
    divisor_degree = len(divisor) - 1
    divisor_lead = divisor[-1]
    while True:
        remainder = list(trim(remainder))
        if remainder == [Fraction(0)] or len(remainder) - 1 < divisor_degree:
            break
        shift = len(remainder) - 1 - divisor_degree
        coefficient = remainder[-1] / divisor_lead
        quotient[shift] += coefficient
        for index, divisor_coefficient in enumerate(divisor):
            remainder[shift + index] -= coefficient * divisor_coefficient
    return trim(quotient), trim(remainder)


def sturm_sequence(poly: Poly) -> list[Poly]:
    poly = trim(poly)
    if poly == (0,):
        raise ValueError("zero polynomial has no Sturm sequence")
    sequence = [poly, poly_derivative(poly)]
    if sequence[-1] == (0,):
        return sequence[:-1]
    while True:
        _, remainder = poly_divmod(sequence[-2], sequence[-1])
        if remainder == (0,):
            break
        sequence.append(poly_scale(remainder, Fraction(-1)))
    return sequence


def sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def sign_variations(values: Iterable[Fraction]) -> int:
    signs = [sign(value) for value in values if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def sturm_root_count(poly: Poly, left: Fraction, right: Fraction) -> int:
    if not left < right:
        raise ValueError("bad Sturm interval")
    if poly_eval(poly, left) == 0 or poly_eval(poly, right) == 0:
        raise ValueError("a Sturm endpoint is a root")
    sequence = sturm_sequence(poly)
    left_variations = sign_variations(poly_eval(item, left) for item in sequence)
    right_variations = sign_variations(poly_eval(item, right) for item in sequence)
    return left_variations - right_variations


def monomial(power: int) -> Poly:
    return tuple([Fraction(0)] * power + [Fraction(1)])


def linear(constant: int | Fraction, slope: int | Fraction) -> Poly:
    return (Fraction(constant), Fraction(slope))


def gamma(n: int) -> Fraction:
    return Fraction(1) if n == 0 else Fraction(factorial(n), n**n)


def decimal_string(value: Fraction, digits: int = 24) -> str:
    prefix = "-" if value < 0 else ""
    value = abs(value)
    whole, remainder = divmod(value.numerator, value.denominator)
    decimal_digits: list[str] = []
    for _ in range(digits):
        remainder *= 10
        digit, remainder = divmod(remainder, value.denominator)
        decimal_digits.append(str(digit))
    return f"{prefix}{whole}." + "".join(decimal_digits)


def bipoly_add(left: BiPoly, right: BiPoly) -> BiPoly:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, Fraction(0)) + value
        if result[key] == 0:
            del result[key]
    return result


def bipoly_scale(poly: BiPoly, scalar: Fraction) -> BiPoly:
    return {key: scalar * value for key, value in poly.items() if scalar * value}


def bipoly_mul(left: BiPoly, right: BiPoly) -> BiPoly:
    result: BiPoly = {}
    for (i, j), x_value in left.items():
        for (k, ell), y_value in right.items():
            key = (i + k, j + ell)
            result[key] = result.get(key, Fraction(0)) + x_value * y_value
    return {key: value for key, value in result.items() if value}


def bipoly_diff_p(poly: BiPoly) -> BiPoly:
    return {(i, j - 1): j * coefficient for (i, j), coefficient in poly.items() if j}


def verify_two_zero_equalization(n: int) -> None:
    """Check the exact two-parameter derivative used to equalize the face."""
    m = n - 2
    one: BiPoly = {(0, 0): Fraction(1)}
    s_value: BiPoly = {(1, 0): Fraction(1)}
    p_value: BiPoly = {(0, 1): Fraction(1)}
    x_value = bipoly_scale(
        bipoly_add(one, bipoly_scale(s_value, Fraction(-1))), Fraction(1, m)
    )
    x1x2: BiPoly = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(-m),
        (0, 1): Fraction(m * m),
    }
    mixed: BiPoly = {
        (2, 0): Fraction(1),
        (0, 1): Fraction(-2),
        (1, 1): Fraction(-m),
    }
    expression = bipoly_mul(bipoly_mul(x_value, x_value), x1x2)
    expression = bipoly_add(
        expression, bipoly_scale(bipoly_mul(x_value, mixed), Fraction(m))
    )
    expression = bipoly_add(
        expression,
        bipoly_scale(bipoly_mul(p_value, p_value), Fraction(m * (m - 1))),
    )
    expected: BiPoly = {
        (0, 1): Fraction(2 * m * (m - 1)),
        (2, 0): Fraction(m + 1),
        (1, 0): Fraction(-m),
        (0, 0): Fraction(-1),
    }
    if bipoly_diff_p(expression) != expected:
        raise AssertionError("two-zero equalization identity failed")
    endpoint_zero = Fraction(-1)
    endpoint_two_over_m = Fraction(-1) + Fraction(2, m) + Fraction(4, m * m)
    if not (endpoint_zero < 0 and endpoint_two_over_m < 0):
        raise AssertionError("two-zero equalization derivative sign failed")


def two_zero_permanent_polynomial(n: int) -> Poly:
    """Return the permanent polynomial on the equalized two-zero face."""
    m = n - 2
    x_value = monomial(1)
    a_value = poly_scale(linear(2 - m, m * m), Fraction(1, 2))
    b_value = poly_scale(linear(1, -m), Fraction(1, 2))
    bracket = poly_add(
        poly_mul(poly_mul(x_value, x_value), poly_mul(a_value, a_value)),
        poly_add(
            poly_scale(
                poly_mul(poly_mul(x_value, a_value), poly_mul(b_value, b_value)),
                Fraction(2 * m),
            ),
            poly_scale(poly_pow(b_value, 4), Fraction(m * (m - 1))),
        ),
    )
    return poly_scale(poly_mul(monomial(m - 2), bracket), Fraction(factorial(m)))


def marginal_numerator(n: int, lower_bound: Fraction) -> tuple[Poly, Poly, Poly]:
    """Construct numerator polynomials for H, D-gamma, and the denominator."""
    gamma_n = gamma(n)
    one_minus_a = linear(1, -1)
    row_product_bound = poly_mul(
        one_minus_a, poly_pow(linear(1, Fraction(1, n - 1)), n - 1)
    )

    denominator = linear(1, -(2 - gamma_n))
    h_numerator = poly_scale(monomial(1), 1 - gamma_n)
    column_numerator = poly_mul(
        poly_add(denominator, h_numerator),
        poly_pow(
            poly_sub(poly_scale(denominator, Fraction(n - 1)), h_numerator),
            n - 1,
        ),
    )
    common_denominator = poly_scale(
        poly_pow(denominator, n), Fraction((n - 1) ** (n - 1))
    )

    deficit_numerator = poly_sub(
        poly_scale(common_denominator, Fraction(2)),
        poly_add(poly_mul(row_product_bound, common_denominator), column_numerator),
    )
    deficit_minus_gamma = poly_sub(
        deficit_numerator, poly_scale(common_denominator, gamma_n)
    )
    contradiction_numerator = poly_add(
        deficit_minus_gamma,
        poly_scale(
            poly_mul(poly_pow(one_minus_a, n), common_denominator), lower_bound
        ),
    )
    return trim(contradiction_numerator), trim(deficit_minus_gamma), trim(common_denominator)


def verify_n10() -> None:
    n = 10
    m = n - 2
    gamma_n = gamma(n)
    lower_bound = Fraction(36_719, 100_000_000)

    verify_two_zero_equalization(n)
    permanent_poly = two_zero_permanent_polynomial(n)
    if len(permanent_poly) - 1 != n:
        raise AssertionError("unexpected two-zero permanent-polynomial degree")
    face_left = Fraction(m - 2, m * m)
    face_right = Fraction(1, m)
    face_difference = poly_sub(permanent_poly, (lower_bound,))
    face_left_value = poly_eval(face_difference, face_left)
    face_right_value = poly_eval(face_difference, face_right)
    if not (face_left_value > 0 and face_right_value > 0):
        raise AssertionError("two-zero face endpoint failure")
    face_roots = sturm_root_count(face_difference, face_left, face_right)
    if face_roots != 0:
        raise AssertionError(f"P_10-L_10 has {face_roots} root(s) on its domain")
    if not lower_bound > gamma_n:
        raise AssertionError("L_10 must exceed gamma_10")

    contradiction, deficit_minus_gamma, common = marginal_numerator(n, lower_bound)
    if len(contradiction) - 1 != 20:
        raise AssertionError("unexpected marginal numerator degree")
    interval_left = Fraction(0)
    interval_right = Fraction(1, 50)
    linear_denominator_right = 1 - (2 - gamma_n) * interval_right
    if not linear_denominator_right > 0:
        raise AssertionError("marginal denominator is not positive on the interval")
    deficit_margin = poly_eval(deficit_minus_gamma, interval_right)
    if not deficit_margin > 0:
        raise AssertionError("D(1/50) does not exceed gamma_10")
    contradiction_left = poly_eval(contradiction, interval_left)
    contradiction_right = poly_eval(contradiction, interval_right)
    if not (contradiction_left > 0 and contradiction_right > 0):
        raise AssertionError("marginal contradiction endpoint failure")
    marginal_roots = sturm_root_count(contradiction, interval_left, interval_right)
    if marginal_roots != 0:
        raise AssertionError(f"marginal numerator has {marginal_roots} root(s)")

    proper_cuts = 0
    minimum_eta: Fraction | None = None
    minimum_case: tuple[int, int, int, Fraction] | None = None
    for u in range(1, n):
        for v in range(1, n - u):
            k = n - u - v
            staircase_permanent = gamma(n - u) * gamma(n - v) / gamma(k)
            supported_permutations = Fraction(
                factorial(n - v) * factorial(n - u), factorial(k)
            )
            permutation_weight = (
                Fraction(1, (n - v) ** u)
                * Fraction(1, (n - u) ** v)
                * Fraction(k**k, ((n - u) * (n - v)) ** k)
            )
            if supported_permutations * permutation_weight != staircase_permanent:
                raise AssertionError("staircase barycenter count failed")
            eta = (
                staircase_permanent
                - gamma_n
                - Fraction(n**3, 4 * k * k)
                * staircase_permanent
                * staircase_permanent
                / (1 - gamma_n)
            )
            if eta <= 0:
                raise AssertionError(f"proper-cut failure at u={u}, v={v}")
            proper_cuts += 1
            if minimum_eta is None or eta < minimum_eta:
                minimum_eta = eta
                minimum_case = (u, v, k, staircase_permanent)

    if minimum_eta is None or minimum_case is None:
        raise AssertionError("no proper cuts were checked")
    u, v, k, staircase_permanent = minimum_case
    h_zero = contradiction_left / poly_eval(common, interval_left)
    d_right_minus_gamma = deficit_margin / poly_eval(common, interval_right)

    print("Dittert n=10 exact dimension-specific verifier")
    print("gamma_10 =", gamma_n, "~=", decimal_string(gamma_n))
    print("two-zero L_10 =", lower_bound, "~=", decimal_string(lower_bound))
    print("two-zero face Sturm roots =", face_roots)
    print("marginal interval = [0, 1/50]")
    print("D(1/50)-gamma_10 =", d_right_minus_gamma)
    print("marginal Sturm roots =", marginal_roots)
    print("H(0) =", h_zero, "~=", decimal_string(h_zero))
    print("proper cuts checked =", proper_cuts)
    print(
        f"smallest proper eta at (u,v,k)=({u},{v},{k}), "
        f"nu={staircase_permanent}: {minimum_eta} ~= {decimal_string(minimum_eta)}"
    )
    print("CERTIFIED")


if __name__ == "__main__":
    verify_n10()
