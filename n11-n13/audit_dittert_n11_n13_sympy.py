#!/usr/bin/env python3
"""Independent exact audit for the n=11,12,13 Dittert proof.

This audit deliberately does not import the primary verifier.  It uses SymPy's
exact polynomial/Sturm implementation and a literal exact Ryser permanent
calculation at n+1 rational points to validate the two-zero-face polynomial.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import factorial

import sympy as sp

X = sp.symbols("x")
S, PVAR = sp.symbols("s p")

CASES = {
    11: sp.Rational(113, 800_000),
    12: sp.Rational(1083, 20_000_000),
    13: sp.Rational(10349, 500_000_000),
}


def gamma(n: int) -> sp.Rational:
    if n == 0:
        return sp.Rational(1)
    return sp.Rational(factorial(n), n**n)


def ryser_permanent_fraction(matrix: list[list[Fraction]]) -> Fraction:
    """Literal exact Ryser formula; suitable here for n<=13."""
    n = len(matrix)
    total = Fraction(0)
    for mask in range(1, 1 << n):
        bits = mask.bit_count()
        product = Fraction(1)
        for i in range(n):
            row_sum = Fraction(0)
            row = matrix[i]
            for j in range(n):
                if mask >> j & 1:
                    row_sum += row[j]
            product *= row_sum
            if product == 0:
                break
        total += product if (n - bits) % 2 == 0 else -product
    return total


def two_zero_matrix(n: int, x: Fraction) -> list[list[Fraction]]:
    m = n - 2
    a = (Fraction(2 - m) + m * m * x) / 2
    b = (Fraction(1) - m * x) / 2
    A = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    A[0][0] = a
    A[1][1] = a
    for j in range(2, n):
        A[0][j] = b
        A[1][j] = b
    for i in range(2, n):
        A[i][0] = b
        A[i][1] = b
        for j in range(2, n):
            A[i][j] = x
    return A


def two_zero_polynomial(n: int) -> sp.Poly:
    m = n - 2
    a = sp.Rational(1, 2) * (2 - m + m * m * X)
    b = sp.Rational(1, 2) * (1 - m * X)
    expression = sp.factorial(m) * X ** (m - 2) * (
        X**2 * a**2 + 2 * m * X * a * b**2 + m * (m - 1) * b**4
    )
    return sp.Poly(sp.expand(expression), X, domain=sp.QQ)


def verify_equalization(n: int) -> None:
    m = n - 2
    x = (1 - S) / m
    x1x2 = 1 - m * S + m * m * PVAR
    mixed = S**2 - (2 + m * S) * PVAR
    E = sp.expand(x**2 * x1x2 + m * x * mixed + m * (m - 1) * PVAR**2)
    derivative = sp.expand(sp.diff(E, PVAR))
    expected = 2 * m * (m - 1) * PVAR + (m + 1) * S**2 - m * S - 1
    if sp.expand(derivative - expected) != 0:
        raise AssertionError(f"n={n}: independent equalization identity failed")
    D = sp.expand(expected.subs(PVAR, S**2 / 4))
    if sp.factor(D - ((m * m + m + 2) * S**2 - 2 * m * S - 2) / 2) != 0:
        raise AssertionError(f"n={n}: independent envelope identity failed")
    if not (D.subs(S, 0) < 0 and D.subs(S, sp.Rational(2, m)) < 0):
        raise AssertionError(f"n={n}: independent equalization endpoint check failed")


def audit_case(n: int, L: sp.Rational) -> None:
    m = n - 2
    lo = sp.Rational(m - 2, m * m)
    hi = sp.Rational(1, m)
    poly = two_zero_polynomial(n)

    verify_equalization(n)

    # A permanent of an n x n matrix whose entries are affine in x has
    # degree at most n.  Agreement at n+1 exact points proves the formula.
    for j in range(n + 1):
        point = lo + (hi - lo) * sp.Rational(j, n)
        point_f = Fraction(int(point.p), int(point.q))
        direct = ryser_permanent_fraction(two_zero_matrix(n, point_f))
        formula = poly.eval(point)
        if sp.Rational(direct.numerator, direct.denominator) != formula:
            raise AssertionError(f"n={n}: Ryser mismatch at sample {j}")

    q = sp.Poly(poly.as_expr() - L, X, domain=sp.QQ)
    if q.eval(lo) <= 0 or q.eval(hi) <= 0:
        raise AssertionError(f"n={n}: endpoint positivity failed")
    roots = sp.polys.polytools.count_roots(q, lo, hi)
    if roots != 0:
        raise AssertionError(f"n={n}: SymPy Sturm found {roots} roots")

    g = gamma(n)
    eta_m = sp.factor(L - g - sp.Rational(n * (n - 1), 2) * L**2 / (1 - g))
    if eta_m <= 0:
        raise AssertionError(f"n={n}: marginal eta failed")

    min_eta = None
    min_data = None
    count = 0
    for u in range(1, n):
        for v in range(1, n - u):
            k = n - u - v
            nu = sp.factor(gamma(n - u) * gamma(n - v) / gamma(k))

            # Independent supported-permutation count for the two-step
            # staircase barycenter.
            supported = sp.Rational(factorial(n - v) * factorial(n - u), factorial(k))
            weight = (
                sp.Rational(1, (n - v) ** u)
                * sp.Rational(1, (n - u) ** v)
                * sp.Rational(k**k, ((n - u) * (n - v)) ** k)
            )
            if sp.factor(supported * weight - nu) != 0:
                raise AssertionError(f"n={n},u={u},v={v}: barycenter count mismatch")

            eta = sp.factor(nu - g - sp.Rational(n**3, 4 * k * k) * nu**2 / (1 - g))
            if eta <= 0:
                raise AssertionError(f"n={n},u={u},v={v}: proper eta failed")
            count += 1
            if min_eta is None or eta < min_eta:
                min_eta = eta
                min_data = (u, v, k, nu)

    print(f"n={n}: INDEPENDENTLY CERTIFIED")
    print(f"  exact Ryser samples = {n+1}")
    print(f"  SymPy Sturm roots = {roots}")
    print(f"  marginal eta = {eta_m}")
    print(f"  proper cuts = {count}, minimum at {min_data}, eta={min_eta}")


def main() -> None:
    for n, L in CASES.items():
        audit_case(n, L)
    print("INDEPENDENT AUDIT CERTIFIED")


if __name__ == "__main__":
    main()
