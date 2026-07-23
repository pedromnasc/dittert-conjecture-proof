#!/usr/bin/env python3
"""Primary exact verifier for the n=8 Dittert working proof.

The accompanying proof note lists the external mathematical inputs.  Every
dimension-specific comparison below uses only integer and Fraction arithmetic.
Printed decimals are diagnostic and are never used in a proof decision.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial
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
    return trim(
        tuple(Fraction(index) * poly[index] for index in range(1, len(poly)))
    )


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
    while True:
        remainder = list(trim(remainder))
        if remainder == [Fraction(0)] or len(remainder) - 1 < divisor_degree:
            break
        shift = len(remainder) - 1 - divisor_degree
        coefficient = remainder[-1] / divisor[-1]
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
            return sequence
        sequence.append(poly_scale(remainder, Fraction(-1)))


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
    return sign_variations(poly_eval(item, left) for item in sequence) - sign_variations(
        poly_eval(item, right) for item in sequence
    )


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
    output: list[str] = []
    for _ in range(digits):
        remainder *= 10
        digit, remainder = divmod(remainder, value.denominator)
        output.append(str(digit))
    return f"{prefix}{whole}." + "".join(output)


def compose_affine(poly: Poly, left: Fraction, width: Fraction) -> Poly:
    result: Poly = (Fraction(0),)
    for coefficient in reversed(poly):
        result = poly_add(poly_mul(result, (left, width)), (coefficient,))
    return result


def power_to_bernstein(poly: Poly, left: Fraction, right: Fraction) -> Poly:
    degree = len(poly) - 1
    power = compose_affine(poly, left, right - left)
    power += (Fraction(0),) * (degree + 1 - len(power))
    return tuple(
        sum(
            power[k] * Fraction(comb(index, k), comb(degree, k))
            for k in range(index + 1)
        )
        for index in range(degree + 1)
    )


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
    m = n - 2
    one: BiPoly = {(0, 0): Fraction(1)}
    s_value: BiPoly = {(1, 0): Fraction(1)}
    p_value: BiPoly = {(0, 1): Fraction(1)}
    x_value = bipoly_scale(
        bipoly_add(one, bipoly_scale(s_value, Fraction(-1))), Fraction(1, m)
    )
    x1x2: BiPoly = {(0, 0): Fraction(1), (1, 0): Fraction(-m), (0, 1): Fraction(m * m)}
    mixed: BiPoly = {(2, 0): Fraction(1), (0, 1): Fraction(-2), (1, 1): Fraction(-m)}
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
    if Fraction(-1) + Fraction(2, m) + Fraction(4, m * m) >= 0:
        raise AssertionError("two-zero equalization sign failed")


def two_zero_permanent_polynomial(n: int) -> Poly:
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
    gamma_n = gamma(n)
    one_minus_a = linear(1, -1)
    row_bound = poly_mul(
        one_minus_a, poly_pow(linear(1, Fraction(1, n - 1)), n - 1)
    )
    denominator = linear(1, -(2 - gamma_n))
    h_numerator = poly_scale(monomial(1), 1 - gamma_n)
    column_numerator = poly_mul(
        poly_add(denominator, h_numerator),
        poly_pow(
            poly_sub(poly_scale(denominator, n - 1), h_numerator), n - 1
        ),
    )
    common = poly_scale(
        poly_pow(denominator, n), Fraction((n - 1) ** (n - 1))
    )
    deficit = poly_sub(
        poly_scale(common, 2),
        poly_add(poly_mul(row_bound, common), column_numerator),
    )
    deficit_minus_gamma = poly_sub(deficit, poly_scale(common, gamma_n))
    contradiction = poly_add(
        deficit_minus_gamma,
        poly_scale(poly_mul(poly_pow(one_minus_a, n), common), lower_bound),
    )
    return trim(contradiction), trim(deficit_minus_gamma), trim(common)


def support_base_polynomials(n: int) -> tuple[Poly, Poly, Poly, Poly, Poly]:
    gamma_n = gamma(n)
    e_value = linear(1, -(2 - gamma_n))
    h_numerator = poly_scale(monomial(1), 1 - gamma_n)
    t_value = poly_sub(e_value, poly_scale(h_numerator, Fraction(1, n - 1)))
    e_power = poly_pow(e_value, n - 1)
    q_numerator = poly_sub(
        poly_scale(e_power, 2 - gamma_n), poly_pow(t_value, n - 1)
    )
    return e_value, h_numerator, t_value, e_power, q_numerator


def subset_product_polynomial(n: int, subset_size: int) -> Poly:
    return poly_mul(
        poly_pow(linear(1, Fraction(1, subset_size)), subset_size),
        poly_pow(
            linear(1, Fraction(-1, n - subset_size)), n - subset_size
        ),
    )


def verify_n8() -> None:
    n = 8
    m = n - 2
    gamma_n = gamma(n)
    lower_bound = Fraction(12_249, 5_000_000)

    verify_two_zero_equalization(n)
    permanent_poly = two_zero_permanent_polynomial(n)
    if len(permanent_poly) - 1 != n:
        raise AssertionError("unexpected two-zero polynomial degree")
    face_left = Fraction(m - 2, m * m)
    face_right = Fraction(1, m)
    face_difference = poly_sub(permanent_poly, (lower_bound,))
    if not (
        poly_eval(face_difference, face_left) > 0
        and poly_eval(face_difference, face_right) > 0
    ):
        raise AssertionError("two-zero face endpoint failure")
    face_roots = sturm_root_count(face_difference, face_left, face_right)
    if face_roots:
        raise AssertionError(f"P_8-L_8 has {face_roots} root(s)")
    if lower_bound <= gamma_n:
        raise AssertionError("L_8 must exceed gamma_8")

    contradiction, deficit_minus_gamma, common = marginal_numerator(n, lower_bound)
    if len(contradiction) - 1 != 16:
        raise AssertionError("unexpected marginal numerator degree")
    feasible_right = Fraction(1, 20)
    risk_left = Fraction(1, 400)
    risk_right = Fraction(1, 75)
    if 1 - (2 - gamma_n) * feasible_right <= 0:
        raise AssertionError("marginal denominator is not positive")
    deficit_margin = poly_eval(deficit_minus_gamma, feasible_right)
    if deficit_margin <= 0:
        raise AssertionError("D(1/20) does not exceed gamma_8")
    safe_intervals = ((Fraction(0), risk_left), (risk_right, feasible_right))
    safe_roots: list[int] = []
    for left, right in safe_intervals:
        if not (poly_eval(contradiction, left) > 0 and poly_eval(contradiction, right) > 0):
            raise AssertionError("safe marginal endpoint failure")
        root_count = sturm_root_count(contradiction, left, right)
        if root_count:
            raise AssertionError(f"safe marginal interval has {root_count} root(s)")
        safe_roots.append(root_count)

    e_value, h_numerator, t_value, e_power, q_numerator = support_base_polynomials(n)
    h_right = (1 - gamma_n) * risk_right / (1 - (2 - gamma_n) * risk_right)
    if not 0 < h_right < n - 1:
        raise AssertionError("h leaves its monotonicity range")
    if poly_eval(poly_sub(q_numerator, e_power), risk_left) <= 0:
        raise AssertionError("q is not greater than one on the risk interval")
    support_minimum: Fraction | None = None
    support_cases = 0
    support_bounds: list[Fraction] = []
    one_minus_a = linear(1, -1)
    for zero_count in range(1, n - 1):
        support_size = n - zero_count
        staircase_bound = (
            gamma(n - zero_count) * gamma(n - 1) / gamma(n - zero_count - 1)
        )
        case_bound = max(lower_bound, staircase_bound)
        support_bounds.append(case_bound)
        r_numerator = poly_sub(
            poly_mul(
                poly_add(
                    (2 - gamma_n,),
                    poly_scale(poly_pow(one_minus_a, n), case_bound),
                ),
                poly_pow(e_value, n),
            ),
            poly_mul(poly_add(e_value, h_numerator), poly_pow(t_value, n - 1)),
        )
        weighted_q = poly_sub(
            poly_mul(linear(n - 1, 1), q_numerator),
            poly_scale(e_power, support_size - 1),
        )
        left_side = poly_scale(
            poly_mul(poly_pow(q_numerator, n - 1), r_numerator),
            Fraction(zero_count**zero_count),
        )
        right_side = poly_mul(
            poly_mul(
                one_minus_a,
                poly_pow(e_value, n + (n - 1) * (support_size - 1)),
            ),
            poly_pow(weighted_q, zero_count),
        )
        gap = poly_sub(left_side, right_side)
        if len(gap) - 1 != 65:
            raise AssertionError("unexpected support-gap degree")
        bernstein = power_to_bernstein(gap, risk_left, risk_right)
        case_minimum = min(bernstein)
        if case_minimum <= 0:
            raise AssertionError(f"support gap failed for z={zero_count}")
        support_cases += 1
        support_minimum = (
            case_minimum
            if support_minimum is None
            else min(support_minimum, case_minimum)
        )

    subset_constants = {
        1: Fraction(1, 2),
        2: Fraction(3, 10),
        3: Fraction(1, 4),
        4: Fraction(6, 25),
        5: Fraction(1, 4),
        6: Fraction(3, 10),
        7: Fraction(1, 2),
    }
    subset_minimum: Fraction | None = None
    subset_endpoint_margin: Fraction | None = None
    square = monomial(2)
    for subset_size, constant in subset_constants.items():
        product_bound = subset_product_polynomial(n, subset_size)
        deficit_difference = poly_sub(
            poly_sub((Fraction(1),), product_bound), poly_scale(square, constant)
        )
        quotient, remainder = poly_divmod(deficit_difference, square)
        if remainder != (0,) or len(quotient) - 1 != 6:
            raise AssertionError("subset-deficit factorization failed")
        coefficients = power_to_bernstein(
            quotient, Fraction(-1, 10), Fraction(1, 10)
        )
        if min(coefficients) <= 0:
            raise AssertionError(f"subset-deficit bound failed for k={subset_size}")
        subset_minimum = (
            min(coefficients)
            if subset_minimum is None
            else min(subset_minimum, *coefficients)
        )
        for endpoint in (Fraction(-1, 10), Fraction(1, 10)):
            margin = 1 - poly_eval(product_bound, endpoint) - gamma_n
            if margin <= 0:
                raise AssertionError("subset localization endpoint failed")
            subset_endpoint_margin = (
                margin
                if subset_endpoint_margin is None
                else min(subset_endpoint_margin, margin)
            )

    generic_cuts = 0
    special_cuts = 0
    minimum_eta: Fraction | None = None
    minimum_special: Fraction | None = None
    minimum_case: tuple[int, int, int, Fraction] | None = None
    for u in range(1, n):
        for v_value in range(1, n - u):
            k = n - u - v_value
            staircase_permanent = gamma(n - u) * gamma(n - v_value) / gamma(k)
            supported_permutations = Fraction(
                factorial(n - v_value) * factorial(n - u), factorial(k)
            )
            permutation_weight = (
                Fraction(1, (n - v_value) ** u)
                * Fraction(1, (n - u) ** v_value)
                * Fraction(k**k, ((n - u) * (n - v_value)) ** k)
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
            if eta > 0:
                generic_cuts += 1
                if minimum_eta is None or eta < minimum_eta:
                    minimum_eta = eta
                    minimum_case = (u, v_value, k, staircase_permanent)
                continue
            if k != 1:
                raise AssertionError(f"unexpected proper-cut obstruction at u={u}, v={v_value}")
            effective = (
                subset_constants[u]
                * subset_constants[v_value]
                / (subset_constants[u] + subset_constants[v_value])
            )
            special_poly = poly_add(
                poly_scale(poly_pow(one_minus_a, n), staircase_permanent),
                (-gamma_n, Fraction(0), effective),
            )
            if len(special_poly) - 1 != n:
                raise AssertionError("unexpected special-cut polynomial degree")
            coefficients: list[Fraction] = []
            for half in range(2):
                coefficients.extend(
                    power_to_bernstein(
                        special_poly, Fraction(half, 14), Fraction(half + 1, 14)
                    )
                )
            if min(coefficients) <= 0:
                raise AssertionError(f"special proper-cut failure at u={u}, v={v_value}")
            special_cuts += 1
            minimum_special = (
                min(coefficients)
                if minimum_special is None
                else min(minimum_special, *coefficients)
            )

    if 2401 * gamma_n >= 6:
        raise AssertionError("special-cut interval bound s<1/7 failed")
    if n * gamma_n >= 1 - gamma_n:
        raise AssertionError("generic proper-cut square-root bound failed")
    if any(
        value is None
        for value in (
            support_minimum,
            subset_minimum,
            subset_endpoint_margin,
            minimum_eta,
            minimum_special,
            minimum_case,
        )
    ):
        raise AssertionError("an exact check was skipped")

    u, v_value, k, staircase_permanent = minimum_case
    h_zero = poly_eval(contradiction, Fraction(0)) / poly_eval(common, Fraction(0))
    d_margin = deficit_margin / poly_eval(common, feasible_right)
    print("Dittert n=8 exact dimension-specific verifier")
    print("gamma_8 =", gamma_n, "~=", decimal_string(gamma_n))
    print("two-zero L_8 =", lower_bound, "~=", decimal_string(lower_bound))
    print("two-zero face Sturm roots =", face_roots)
    print("marginal feasible interval = [0, 1/20]")
    print("D(1/20)-gamma_8 =", d_margin)
    print("safe marginal Sturm roots =", safe_roots)
    print("marginal risk interval = [1/400, 1/75]")
    print("support cases checked =", support_cases)
    print("support permanent bounds =", support_bounds)
    print("smallest support Bernstein coefficient ~=", decimal_string(support_minimum))
    print("H(0) =", h_zero, "~=", decimal_string(h_zero))
    print("subset-deficit cases checked =", len(subset_constants))
    print("smallest subset Bernstein coefficient ~=", decimal_string(subset_minimum))
    print("generic proper cuts checked =", generic_cuts)
    print("special proper cuts checked =", special_cuts)
    print("smallest special Bernstein coefficient ~=", decimal_string(minimum_special))
    print(
        f"smallest generic eta at (u,v,k)=({u},{v_value},{k}), "
        f"nu={staircase_permanent}: {minimum_eta} ~= {decimal_string(minimum_eta)}"
    )
    print("CERTIFIED")


if __name__ == "__main__":
    verify_n8()
