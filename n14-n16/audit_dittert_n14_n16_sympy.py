#!/usr/bin/env python3
"""Independent exact SymPy/Sturm audit for dimensions 14, 15, and 16.

This audit independently checks the two-parameter block permanent,
its reduction to equal exceptional parameters, the one-variable permanent
polynomial, the full-interval lower bound, and the final scalar inequality.
All symbolic domains are exact rationals.
"""

from math import factorial, gcd
import sympy as sp


def integer_lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def ryser_integer_permanent(matrix: list[list[int]]) -> int:
    """Exact Ryser permanent using a Gray-code subset traversal."""
    n = len(matrix)
    row_sums = [0] * n
    total = 0
    previous_gray = 0
    for counter in range(1, 1 << n):
        gray = counter ^ (counter >> 1)
        changed = gray ^ previous_gray
        column = (changed & -changed).bit_length() - 1
        direction = 1 if gray & changed else -1
        for row in range(n):
            row_sums[row] += direction * matrix[row][column]
        product = 1
        for value in row_sums:
            product *= value
        total += product if (n - gray.bit_count()) % 2 == 0 else -product
        previous_gray = gray
    return total


def ryser_permanent_of_B(n: int, x_value: sp.Rational) -> sp.Rational:
    """Evaluate per(B_m(x_value)) independently by exact integer Ryser."""
    m = n - 2
    a_value = (2 - m + m * m * x_value) / 2
    b_value = (1 - m * x_value) / 2
    denominators = [int(sp.denom(v)) for v in (a_value, b_value, x_value)]
    scale = 1
    for denominator in denominators:
        scale = integer_lcm(scale, denominator)

    matrix: list[list[int]] = []
    for row in range(n):
        current: list[int] = []
        for column in range(n):
            if (row, column) in ((0, 1), (1, 0)):
                value = sp.Rational(0)
            elif row < 2 and column < 2:
                value = a_value
            elif row < 2 or column < 2:
                value = b_value
            else:
                value = x_value
            current.append(int(value * scale))
        matrix.append(current)
    return sp.Rational(ryser_integer_permanent(matrix), scale**n)


x, s, p = sp.symbols("x s p")
a1, a2 = sp.symbols("a1 a2")

CASES = (
    (14, sp.Rational(789, 10**8)),
    (15, sp.Rational(3001, 10**9)),
    (16, sp.Rational(1139, 10**9)),
)

for n, L in CASES:
    m = n - 2

    # Independent reconstruction of the permanent of the averaged block form:
    # [x1,0,a1*1^T; 0,x2,a2*1^T; a1*1,a2*1,x*J_m].
    x_block = (1 - (a1 + a2)) / m
    x1 = 1 - m * a1
    x2 = 1 - m * a2
    E_direct = sp.expand(
        x_block**2 * x1 * x2
        + m * x_block * (x1 * a2**2 + x2 * a1**2)
        + m * (m - 1) * a1**2 * a2**2
    )
    E_sp = sp.expand(
        ((1 - s) / m) ** 2 * (1 - m * s + m**2 * p)
        + m * ((1 - s) / m) * (s**2 - (2 + m * s) * p)
        + m * (m - 1) * p**2
    )
    if sp.expand(E_direct - E_sp.subs({s: a1 + a2, p: a1 * a2})) != 0:
        raise AssertionError(f"n={n}: symmetric two-parameter permanent identity failed")

    derivative = sp.factor(sp.diff(E_sp, p))
    expected_derivative = 2 * m * (m - 1) * p + (m + 1) * s**2 - m * s - 1
    if sp.expand(derivative - expected_derivative) != 0:
        raise AssertionError(f"n={n}: derivative in p mismatch")

    # Since p <= s^2/4, the derivative is at most D_m(s).  D_m is convex,
    # so exact negativity at both endpoints proves strict negativity throughout.
    D = sp.factor(expected_derivative.subs(p, s**2 / 4))
    expected_D = ((m**2 + m + 2) * s**2 - 2 * m * s - 2) / 2
    if sp.expand(D - expected_D) != 0:
        raise AssertionError(f"n={n}: equalization upper bound mismatch")
    if not (sp.diff(D, s, 2) > 0 and D.subs(s, 0) < 0 and D.subs(s, sp.Rational(2, m)) < 0):
        raise AssertionError(f"n={n}: equalization negativity failed")

    # Set a1=a2=(1-mx)/2 and reconstruct the one-variable permanent directly
    # from the block count.  Compare it with the closed summation used by the
    # separate verifier.
    b = (1 - m * x) / 2
    a = 1 - m * b
    P_direct = sp.expand(
        factorial(m)
        * x ** (m - 2)
        * (x**2 * a**2 + 2 * m * x * a * b**2 + m * (m - 1) * b**4)
    )
    P_sum = sp.Rational(factorial(m) ** 2, 16) * sum(
        sp.binomial(2, i)
        * sp.Rational(2**i, factorial(m - 2 + i))
        * (2 - m + m * m * x) ** i
        * (1 - m * x) ** (4 - 2 * i)
        * x ** (m - 2 + i)
        for i in range(3)
    )
    if sp.expand(P_direct - P_sum) != 0:
        raise AssertionError(f"n={n}: one-variable permanent formulas disagree")
    P = sp.Poly(sp.expand(P_direct), x, domain=sp.QQ)
    if P.degree() != n:
        raise AssertionError(f"n={n}: unexpected permanent-polynomial degree")

    lo = sp.Rational(m - 2, m * m)
    hi = sp.Rational(1, m)

    # Independent exact check of the polynomial itself.  The permanent of a
    # matrix whose entries are affine in x has degree at most n.  Agreement at
    # n+1 distinct rational points therefore verifies the polynomial identity.
    for point_index in range(n + 1):
        test_x = lo + (hi - lo) * sp.Rational(point_index, n)
        if ryser_permanent_of_B(n, test_x) != P.eval(test_x):
            raise AssertionError(f"n={n}: exact Ryser polynomial check failed")
    Q = sp.Poly(P.as_expr() - L, x, domain=sp.QQ)
    root_count = Q.count_roots(lo, hi)
    if root_count != 0:
        raise AssertionError(f"n={n}: P_n-L_n has {root_count} roots in the domain")
    if Q.eval(lo) <= 0 or Q.eval(hi) <= 0:
        raise AssertionError(f"n={n}: P_n-L_n is not positive at both endpoints")

    gamma = sp.Rational(factorial(n), n**n)
    eta = sp.factor(L - gamma - sp.Rational(n**3, 4) * L**2 / (1 - gamma))
    if eta <= 0:
        raise AssertionError(f"n={n}: scalar inequality failed")
    if n * gamma >= 1 - gamma:
        raise AssertionError(f"n={n}: t<1 check failed")

    print(f"n={n}: STURM CERTIFIED")
    print("  two-parameter equalization reduction = exact")
    print(f"  exact Ryser polynomial checks = {n + 1}")
    print(f"  roots of P_n-L_n on [{lo}, {hi}] = {root_count}")
    print(f"  (P_n-L_n)({lo}) = {Q.eval(lo)}")
    print(f"  eta_n = {eta}")

print("INDEPENDENT AUDIT CERTIFIED")
