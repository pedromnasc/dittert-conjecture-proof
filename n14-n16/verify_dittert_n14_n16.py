#!/usr/bin/env python3
"""Exact verifier for the Dittert conjecture in dimensions 14, 15, and 16.

The verifier checks three finite ingredients using only Python's standard library
and exact rational arithmetic:

1. The two-parameter-to-one-parameter reduction for the permanent on the
   doubly stochastic face with two prescribed independent zeroes.
2. A lower bound L_n for the permanent on that face.
3. The exact scalar inequality that combines this lower bound with the
   joint-deficit scaling lemma.

No floating-point value is used in a correctness decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import Sequence

Poly = tuple[Fraction, ...]  # coefficients in increasing powers
BiPoly = dict[tuple[int, int], Fraction]  # coefficients of s^i p^j
Interval = tuple[Fraction, Fraction]


def trim(p: Sequence[Fraction]) -> Poly:
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return tuple(q)


def poly_add(p: Poly, q: Poly) -> Poly:
    n = max(len(p), len(q))
    out = [Fraction(0) for _ in range(n)]
    for i, value in enumerate(p):
        out[i] += value
    for i, value in enumerate(q):
        out[i] += value
    return trim(out)


def poly_mul(p: Poly, q: Poly) -> Poly:
    out = [Fraction(0) for _ in range(len(p) + len(q) - 1)]
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return trim(out)


def poly_scale(p: Poly, c: Fraction) -> Poly:
    return trim([c * value for value in p])


def poly_pow(p: Poly, exponent: int) -> Poly:
    if exponent < 0:
        raise ValueError("negative polynomial exponent")
    result: Poly = (Fraction(1),)
    base = p
    e = exponent
    while e:
        if e & 1:
            result = poly_mul(result, base)
        base = poly_mul(base, base)
        e >>= 1
    return result


def poly_derivative(p: Poly) -> Poly:
    if len(p) <= 1:
        return (Fraction(0),)
    return trim([Fraction(i) * p[i] for i in range(1, len(p))])


def poly_eval(p: Poly, x: Fraction) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(p):
        value = value * x + coefficient
    return value


def monomial(power: int) -> Poly:
    return tuple([Fraction(0)] * power + [Fraction(1)])


def linear(constant: int | Fraction, slope: int | Fraction) -> Poly:
    return (Fraction(constant), Fraction(slope))


def interval_add(a: Interval, b: Interval) -> Interval:
    return (a[0] + b[0], a[1] + b[1])


def interval_mul(a: Interval, b: Interval) -> Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(products), max(products))


def interval_horner(p: Poly, x_interval: Interval) -> Interval:
    value: Interval = (Fraction(0), Fraction(0))
    for coefficient in reversed(p):
        value = interval_add(
            interval_mul(value, x_interval),
            (coefficient, coefficient),
        )
    return value


def decimal_string(x: Fraction, digits: int = 18) -> str:
    """Return a non-rigorous decimal rendering for human-readable logs only."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    integer = x.numerator // x.denominator
    remainder = x.numerator % x.denominator
    decimals: list[str] = []
    for _ in range(digits):
        remainder *= 10
        decimals.append(str(remainder // x.denominator))
        remainder %= x.denominator
    return f"{sign}{integer}." + "".join(decimals)



def bipoly_add(a: BiPoly, b: BiPoly) -> BiPoly:
    out = dict(a)
    for monomial_key, coefficient in b.items():
        out[monomial_key] = out.get(monomial_key, Fraction(0)) + coefficient
        if out[monomial_key] == 0:
            del out[monomial_key]
    return out


def bipoly_scale(a: BiPoly, c: Fraction) -> BiPoly:
    return {key: c * value for key, value in a.items() if c * value != 0}


def bipoly_mul(a: BiPoly, b: BiPoly) -> BiPoly:
    out: BiPoly = {}
    for (is_, ip), av in a.items():
        for (js, jp), bv in b.items():
            key = (is_ + js, ip + jp)
            out[key] = out.get(key, Fraction(0)) + av * bv
    return {key: value for key, value in out.items() if value != 0}


def bipoly_diff_p(a: BiPoly) -> BiPoly:
    out: BiPoly = {}
    for (is_, ip), coefficient in a.items():
        if ip:
            out[(is_, ip - 1)] = coefficient * ip
    return out


def verify_equalization_reduction(n: int) -> None:
    """Verify the exact two-parameter permanent formula and equalization.

    Minc's averaging theorem reduces a permanent minimizer on the face with
    zeroes (1,2) and (2,1) to the block form

        [ x1  0   a1 1^T ]
        [ 0   x2  a2 1^T ]
        [ a1 1 a2 1  x J ]

    with x_i=1-m a_i and a1+a2+m x=1.  Put s=a1+a2 and
    p=a1*a2.  After removing the positive factor m! x^(m-2), the permanent
    is the bivariate polynomial E(s,p) checked below.  Its p-derivative is
    negative throughout the feasible region, so the minimum has
    p=s^2/4, equivalently a1=a2.
    """
    m = n - 2
    if m < 4:
        raise AssertionError("equalization proof is used only for m >= 4")

    one: BiPoly = {(0, 0): Fraction(1)}
    s_poly: BiPoly = {(1, 0): Fraction(1)}
    p_poly: BiPoly = {(0, 1): Fraction(1)}
    x_poly = bipoly_scale(bipoly_add(one, bipoly_scale(s_poly, Fraction(-1))), Fraction(1, m))

    # x1*x2 = 1-ms+m^2p and
    # x1*a2^2+x2*a1^2 = s^2-(2+ms)p.
    x1x2 = {(0, 0): Fraction(1), (1, 0): Fraction(-m), (0, 1): Fraction(m * m)}
    mixed = {(2, 0): Fraction(1), (0, 1): Fraction(-2), (1, 1): Fraction(-m)}

    E = bipoly_mul(bipoly_mul(x_poly, x_poly), x1x2)
    E = bipoly_add(E, bipoly_scale(bipoly_mul(bipoly_mul(x_poly, mixed), one), Fraction(m)))
    E = bipoly_add(E, bipoly_scale(bipoly_mul(p_poly, p_poly), Fraction(m * (m - 1))))

    dE_dp = bipoly_diff_p(E)
    expected_derivative: BiPoly = {
        (0, 1): Fraction(2 * m * (m - 1)),
        (2, 0): Fraction(m + 1),
        (1, 0): Fraction(-m),
        (0, 0): Fraction(-1),
    }
    if dE_dp != expected_derivative:
        raise AssertionError(f"n={n}: two-parameter permanent derivative mismatch")

    # Since p <= s^2/4 and dE/dp increases with p,
    # dE/dp <= D_m(s) on 0 <= s <= 2/m, where
    # D_m(s)=((m^2+m+2)s^2-2ms-2)/2.  This is convex, hence its maximum
    # on the interval is attained at an endpoint.
    D_at_0 = Fraction(-1)
    D_at_right = Fraction(-1) + Fraction(2, m) + Fraction(4, m * m)
    if not (D_at_0 < 0 and D_at_right < 0):
        raise AssertionError(f"n={n}: equalization derivative is not strictly negative")


def two_zero_permanent_polynomial(n: int) -> Poly:
    """Permanent of the Pula-Song-Wanless one-parameter minimizer family.

    Put m=n-2.  After a column permutation, two independent prescribed
    zeroes give the support V_{m,2}.  A minimizing matrix can be chosen as

        [ a  0  b 1^T ]
        [ 0  a  b 1^T ]
        [ b1 b1   x J ]

    where a=(2-m+m^2 x)/2, b=(1-mx)/2, and
    (m-2)/m^2 <= x <= 1/m.

    Counting permanent terms according to how many of the two entries a
    are used gives the exact polynomial below.
    """
    if n < 4:
        raise ValueError("n must be at least 4")
    m = n - 2
    x = monomial(1)
    a_numerator = linear(2 - m, m * m)  # 2a
    b_numerator = linear(1, -m)         # 2b

    total: Poly = (Fraction(0),)
    for i in range(3):
        coefficient = Fraction(
            factorial(m) ** 2 * comb(2, i) * (2 ** i),
            16 * factorial(m - 2 + i),
        )
        term = poly_mul(poly_pow(a_numerator, i), poly_pow(b_numerator, 4 - 2 * i))
        term = poly_mul(term, monomial(m - 2 + i))
        total = poly_add(total, poly_scale(term, coefficient))
    return trim(total)


@dataclass(frozen=True)
class Case:
    n: int
    L: Fraction
    reported_floor: Fraction
    bracket: Interval
    p_factor_constant: int
    p_x_power: int
    quartic: Poly
    d_factor_constant: int
    d_x_power: int
    linear_factor: Poly
    cubic: Poly
    cubic_derivative_discriminant: Fraction
    cubic_endpoint_values: tuple[Fraction, Fraction]
    expected_eta: Fraction


CASES = (
    Case(
        n=14,
        L=Fraction(789, 10**8),
        reported_floor=Fraction(7890634087, 10**15),
        bracket=(Fraction(70587793823672, 10**15), Fraction(70587793823673, 10**15)),
        p_factor_constant=119750400,
        p_x_power=10,
        quartic=(Fraction(33), Fraction(-1704), Fraction(33220), Fraction(-289728), Fraction(953856)),
        d_factor_constant=718502400,
        d_x_power=9,
        linear_factor=(Fraction(-5), Fraction(84)),
        cubic=(Fraction(-11), Fraction(440), Fraction(-5896), Fraction(26496)),
        cubic_derivative_discriminant=Fraction(-847616),
        cubic_endpoint_values=(Fraction(-1, 216), Fraction(1, 18)),
        expected_eta=Fraction(
            1801777553768587612545302097,
            957898641732692266551875000000000000,
        ),
    ),
    Case(
        n=15,
        L=Fraction(3001, 10**9),
        reported_floor=Fraction(3001151594, 10**15),
        bracket=(Fraction(65989565762012, 10**15), Fraction(65989565762013, 10**15)),
        p_factor_constant=1556755200,
        p_x_power=11,
        quartic=(Fraction(39), Fraction(-2171), Fraction(45582), Fraction(-427739), Fraction(1513733)),
        d_factor_constant=20237817600,
        d_x_power=10,
        linear_factor=(Fraction(-11), Fraction(195)),
        cubic=(Fraction(-3), Fraction(129), Fraction(-1857), Fraction(8957)),
        cubic_derivative_discriminant=Fraction(-71640),
        cubic_endpoint_values=(Fraction(-2, 2197), Fraction(2, 169)),
        expected_eta=Fraction(
            172396552001926445448721,
            24213708253338147072000000000000,
        ),
    ),
    Case(
        n=16,
        L=Fraction(1139, 10**9),
        reported_floor=Fraction(1139154849, 10**15),
        bracket=(Fraction(61946716891694, 10**15), Fraction(61946716891695, 10**15)),
        p_factor_constant=10897286400,
        p_x_power=12,
        quartic=(Fraction(91), Fraction(-5432), Fraction(122200), Fraction(-1227744), Fraction(4648336)),
        d_factor_constant=305124019200,
        d_x_power=11,
        linear_factor=(Fraction(-3), Fraction(56)),
        cubic=(Fraction(-13), Fraction(598), Fraction(-9204), Fraction(47432)),
        cubic_derivative_discriminant=Fraction(-1517568),
        cubic_endpoint_values=(Fraction(-1, 343), Fraction(2, 49)),
        expected_eta=Fraction(
            4164538286276424752117963756736721,
            1208924448418671074738176000000000000000000,
        ),
    ),
)


def verify_case(case: Case) -> None:
    n = case.n
    m = n - 2
    domain: Interval = (Fraction(m - 2, m * m), Fraction(1, m))
    a, b = case.bracket
    if not (domain[0] < a < b < domain[1]):
        raise AssertionError(f"n={n}: root bracket is not inside the admissible interval")

    # Verify the reduction from the averaged two-parameter block family to
    # the one-parameter family, then reconstruct its permanent polynomial.
    verify_equalization_reduction(n)
    P = two_zero_permanent_polynomial(n)

    expected_P = poly_scale(
        poly_mul(monomial(case.p_x_power), case.quartic),
        Fraction(case.p_factor_constant),
    )
    if P != expected_P:
        raise AssertionError(f"n={n}: permanent polynomial factorization mismatch")

    dP = poly_derivative(P)
    expected_dP = poly_scale(
        poly_mul(
            poly_mul(monomial(case.d_x_power), case.linear_factor),
            case.cubic,
        ),
        Fraction(case.d_factor_constant),
    )
    if dP != expected_dP:
        raise AssertionError(f"n={n}: derivative factorization mismatch")

    # The cubic is strictly increasing on R: its derivative is a positive
    # quadratic with negative discriminant.
    c0, c1, c2, c3 = case.cubic
    derivative_discriminant = (2 * c2) ** 2 - 4 * (3 * c3) * c1
    if derivative_discriminant != case.cubic_derivative_discriminant:
        raise AssertionError(f"n={n}: displayed cubic discriminant mismatch")
    if not (3 * c3 > 0 and derivative_discriminant < 0):
        raise AssertionError(f"n={n}: cubic monotonicity check failed")

    # The other nonzero derivative factors are positive on the full domain.
    if poly_eval(case.linear_factor, domain[0]) <= 0:
        raise AssertionError(f"n={n}: linear derivative factor is not positive")
    if domain[0] <= 0:
        raise AssertionError(f"n={n}: admissible x values must be positive")

    # The strictly increasing cubic has one root in the admissible interval,
    # and the supplied dyadic/decimal bracket encloses it exactly.
    endpoint_values = (poly_eval(case.cubic, domain[0]), poly_eval(case.cubic, domain[1]))
    if endpoint_values != case.cubic_endpoint_values:
        raise AssertionError(f"n={n}: displayed cubic endpoint values mismatch")
    if not (endpoint_values[0] < 0 < endpoint_values[1]):
        raise AssertionError(f"n={n}: cubic does not change sign on the domain")
    if not (poly_eval(case.cubic, a) < 0 < poly_eval(case.cubic, b)):
        raise AssertionError(f"n={n}: exact critical-point bracket failed")

    # dP is negative before that root and positive after it.  Hence the global
    # minimum occurs inside [a,b].  Exact interval Horner evaluation proves
    # P(x) > L throughout this bracket.
    P_interval = interval_horner(P, (a, b))
    if P_interval[0] <= case.reported_floor:
        raise AssertionError(f"n={n}: displayed interval lower bound failed")
    if case.reported_floor <= case.L:
        raise AssertionError(f"n={n}: displayed lower bound does not imply L_n")

    gamma = Fraction(factorial(n), n**n)

    # For every 0 < delta <= gamma, the joint-deficit dilation parameter
    # t=sqrt(n delta/(1-delta)) is strictly less than 1.
    if not (n * gamma < 1 - gamma):
        raise AssertionError(f"n={n}: dilation may fail to satisfy t<1")

    # Bernoulli plus completing the square reduces the boundary contradiction
    # to this single exact rational inequality.
    eta = case.L - gamma - Fraction(n**3, 4) * case.L**2 / (1 - gamma)
    if eta != case.expected_eta:
        raise AssertionError(f"n={n}: displayed eta value mismatch")
    if eta <= 0:
        raise AssertionError(f"n={n}: scalar boundary-exclusion inequality failed")

    print(f"n={n}: CERTIFIED")
    print("  two-parameter equalization reduction = exact")
    print(f"  admissible x interval = [{domain[0]}, {domain[1]}]")
    print(f"  critical point bracket = [{a}, {b}]")
    print(f"  exact interval-Horner lower endpoint = {P_interval[0]}")
    print(f"  displayed rational floor = {case.reported_floor}")
    print(f"  chosen L_n = {case.L}")
    print(f"  P_n-L_n margin > {decimal_string(P_interval[0] - case.L, 24)}")
    print(f"  scalar eta_n = {eta}")
    print(f"  eta_n ~= {decimal_string(eta, 24)}")


def main() -> None:
    for case in CASES:
        verify_case(case)
    print("ALL CASES CERTIFIED")


if __name__ == "__main__":
    main()
