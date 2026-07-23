#!/usr/bin/env python3
"""Exact verifier for a boundary-exclusion proof of Dittert's conjecture
in dimensions 11, 12, and 13.

The mathematical proof uses four ingredients external to this program:

* Hwang's theorem that a positive Dittert maximizer is the uniform matrix.
* Cheon--Wanless: a maximizer is fully indecomposable and its complete
  zero set is not one rectangular block.
* Li's cut characterization of doubly superstochastic matrices.
* Hwang's theorem that the barycenter minimizes the permanent on a
  staircase face of the Birkhoff polytope.

Everything dimension-specific is checked here using exact integer/rational
arithmetic only:

1. A strict lower bound L_n for the permanent of a doubly stochastic matrix
   with two prescribed independent zeroes.
2. The marginal-cut scalar inequality.
3. Every proper-cut staircase scalar inequality.

No floating-point value is used in a correctness decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import Iterable, Sequence

Poly = tuple[Fraction, ...]  # coefficients in increasing powers
BiPoly = dict[tuple[int, int], Fraction]  # s^i p^j


def trim(p: Sequence[Fraction]) -> Poly:
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return tuple(q) if q else (Fraction(0),)


def poly_add(p: Poly, q: Poly) -> Poly:
    out = [Fraction(0) for _ in range(max(len(p), len(q)))]
    for i, c in enumerate(p):
        out[i] += c
    for i, c in enumerate(q):
        out[i] += c
    return trim(out)


def poly_sub(p: Poly, q: Poly) -> Poly:
    return poly_add(p, tuple(-c for c in q))


def poly_mul(p: Poly, q: Poly) -> Poly:
    out = [Fraction(0) for _ in range(len(p) + len(q) - 1)]
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return trim(out)


def poly_scale(p: Poly, c: Fraction) -> Poly:
    return trim(tuple(c * a for a in p))


def poly_pow(p: Poly, exponent: int) -> Poly:
    if exponent < 0:
        raise ValueError("negative exponent")
    out: Poly = (Fraction(1),)
    base = p
    e = exponent
    while e:
        if e & 1:
            out = poly_mul(out, base)
        base = poly_mul(base, base)
        e >>= 1
    return out


def poly_derivative(p: Poly) -> Poly:
    if len(p) <= 1:
        return (Fraction(0),)
    return trim(tuple(Fraction(i) * p[i] for i in range(1, len(p))))


def poly_eval(p: Poly, x: Fraction) -> Fraction:
    value = Fraction(0)
    for c in reversed(p):
        value = value * x + c
    return value


def poly_divmod(p: Poly, q: Poly) -> tuple[Poly, Poly]:
    """Exact polynomial division over Q, increasing coefficient order."""
    p = trim(p)
    q = trim(q)
    if q == (0,):
        raise ZeroDivisionError("polynomial division by zero")
    if len(p) < len(q):
        return (Fraction(0),), p
    rem = list(p)
    quotient = [Fraction(0) for _ in range(len(p) - len(q) + 1)]
    qdeg = len(q) - 1
    qlead = q[-1]
    while len(rem) - 1 >= qdeg and any(rem):
        rem = list(trim(rem))
        if rem == [Fraction(0)] or len(rem) - 1 < qdeg:
            break
        shift = (len(rem) - 1) - qdeg
        coeff = rem[-1] / qlead
        quotient[shift] += coeff
        for j, qj in enumerate(q):
            rem[shift + j] -= coeff * qj
    return trim(quotient), trim(rem)


def sturm_sequence(p: Poly) -> list[Poly]:
    p = trim(p)
    if p == (0,):
        raise ValueError("zero polynomial has no Sturm sequence")
    seq = [p, poly_derivative(p)]
    if seq[1] == (0,):
        return seq[:1]
    while seq[-1] != (0,):
        _, rem = poly_divmod(seq[-2], seq[-1])
        if rem == (0,):
            break
        seq.append(poly_scale(rem, Fraction(-1)))
    return seq


def sign(x: Fraction) -> int:
    return (x > 0) - (x < 0)


def sign_variations(values: Iterable[Fraction]) -> int:
    signs = [sign(v) for v in values if v != 0]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def sturm_root_count(p: Poly, a: Fraction, b: Fraction) -> int:
    if not a < b:
        raise ValueError("Sturm interval must satisfy a < b")
    if poly_eval(p, a) == 0 or poly_eval(p, b) == 0:
        raise ValueError("Sturm endpoints must not be roots")
    seq = sturm_sequence(p)
    va = sign_variations(poly_eval(q, a) for q in seq)
    vb = sign_variations(poly_eval(q, b) for q in seq)
    return va - vb


def monomial(power: int) -> Poly:
    return tuple([Fraction(0)] * power + [Fraction(1)])


def linear(constant: int | Fraction, slope: int | Fraction) -> Poly:
    return (Fraction(constant), Fraction(slope))


def bipoly_add(a: BiPoly, b: BiPoly) -> BiPoly:
    out = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, Fraction(0)) + value
        if out[key] == 0:
            del out[key]
    return out


def bipoly_scale(a: BiPoly, c: Fraction) -> BiPoly:
    return {k: c * v for k, v in a.items() if c * v != 0}


def bipoly_mul(a: BiPoly, b: BiPoly) -> BiPoly:
    out: BiPoly = {}
    for (is_, ip), av in a.items():
        for (js, jp), bv in b.items():
            key = (is_ + js, ip + jp)
            out[key] = out.get(key, Fraction(0)) + av * bv
    return {k: v for k, v in out.items() if v != 0}


def bipoly_diff_p(a: BiPoly) -> BiPoly:
    return {(is_, ip - 1): c * ip for (is_, ip), c in a.items() if ip}


def gamma(n: int) -> Fraction:
    if n == 0:
        return Fraction(1)
    return Fraction(factorial(n), n**n)


def decimal_string(x: Fraction, digits: int = 24) -> str:
    sign_prefix = "-" if x < 0 else ""
    x = abs(x)
    whole, remainder = divmod(x.numerator, x.denominator)
    out: list[str] = []
    for _ in range(digits):
        remainder *= 10
        digit, remainder = divmod(remainder, x.denominator)
        out.append(str(digit))
    return f"{sign_prefix}{whole}." + "".join(out)


def verify_two_zero_equalization(n: int) -> None:
    """Verify the reduction from the averaged V_{m,2} face to one variable.

    Let m=n-2.  After averaging the m identical-support rows and columns,
    a minimizer has parameters a1,a2 and x=(1-a1-a2)/m.  Put
    s=a1+a2 and p=a1*a2.  Up to the positive factor m! x^(m-2), its
    permanent is E(s,p).  This routine reconstructs E and verifies

        dE/dp = 2m(m-1)p + (m+1)s^2 - ms - 1.

    Since p <= s^2/4 and 0 <= s <= 2/m, the derivative is bounded above
    by a convex quadratic whose values at both endpoints are negative.
    Hence the permanent decreases with p, and its minimum has a1=a2.
    """
    m = n - 2
    one: BiPoly = {(0, 0): Fraction(1)}
    s_poly: BiPoly = {(1, 0): Fraction(1)}
    p_poly: BiPoly = {(0, 1): Fraction(1)}
    x_poly = bipoly_scale(bipoly_add(one, bipoly_scale(s_poly, Fraction(-1))), Fraction(1, m))

    x1x2: BiPoly = {(0, 0): Fraction(1), (1, 0): Fraction(-m), (0, 1): Fraction(m * m)}
    mixed: BiPoly = {(2, 0): Fraction(1), (0, 1): Fraction(-2), (1, 1): Fraction(-m)}

    E = bipoly_mul(bipoly_mul(x_poly, x_poly), x1x2)
    E = bipoly_add(E, bipoly_scale(bipoly_mul(x_poly, mixed), Fraction(m)))
    E = bipoly_add(E, bipoly_scale(bipoly_mul(p_poly, p_poly), Fraction(m * (m - 1))))

    expected: BiPoly = {
        (0, 1): Fraction(2 * m * (m - 1)),
        (2, 0): Fraction(m + 1),
        (1, 0): Fraction(-m),
        (0, 0): Fraction(-1),
    }
    if bipoly_diff_p(E) != expected:
        raise AssertionError(f"n={n}: equalization derivative identity failed")

    # Upper envelope after p <= s^2/4:
    # D_m(s)=((m^2+m+2)s^2-2ms-2)/2.
    d0 = Fraction(-1)
    d1 = Fraction(-1) + Fraction(2, m) + Fraction(4, m * m)
    if not (d0 < 0 and d1 < 0):
        raise AssertionError(f"n={n}: equalization derivative is not negative")


def two_zero_permanent_polynomial(n: int) -> Poly:
    """Permanent on the equalized two-independent-zero face.

    With m=n-2, the matrix is

        [ a  0  b 1^T ]
        [ 0  a  b 1^T ]
        [ b1 b1   x J ]

    where a=(2-m+m^2 x)/2, b=(1-mx)/2 and
    (m-2)/m^2 <= x <= 1/m.
    """
    m = n - 2
    x = monomial(1)
    two_a = linear(2 - m, m * m)
    two_b = linear(1, -m)

    # Direct count according to whether 0, 1, or 2 diagonal a-entries
    # are selected by the permutation.
    a = poly_scale(two_a, Fraction(1, 2))
    b = poly_scale(two_b, Fraction(1, 2))
    bracket = poly_add(
        poly_mul(poly_mul(x, x), poly_mul(a, a)),
        poly_add(
            poly_scale(poly_mul(poly_mul(x, a), poly_mul(b, b)), Fraction(2 * m)),
            poly_scale(poly_pow(b, 4), Fraction(m * (m - 1))),
        ),
    )
    return poly_scale(poly_mul(monomial(m - 2), bracket), Fraction(factorial(m)))


@dataclass(frozen=True)
class Case:
    n: int
    L: Fraction


CASES = (
    Case(11, Fraction(113, 800_000)),
    Case(12, Fraction(1083, 20_000_000)),
    Case(13, Fraction(10349, 500_000_000)),
)


def verify_case(case: Case) -> None:
    n = case.n
    m = n - 2
    g = gamma(n)
    L = case.L
    domain_left = Fraction(m - 2, m * m)
    domain_right = Fraction(1, m)

    verify_two_zero_equalization(n)
    P = two_zero_permanent_polynomial(n)
    if len(P) - 1 > n:
        raise AssertionError(f"n={n}: permanent polynomial degree is too large")

    Q = poly_sub(P, (L,))
    q_left = poly_eval(Q, domain_left)
    q_right = poly_eval(Q, domain_right)
    if not (q_left > 0 and q_right > 0):
        raise AssertionError(f"n={n}: P_n-L_n is not positive at a domain endpoint")
    roots = sturm_root_count(Q, domain_left, domain_right)
    if roots != 0:
        raise AssertionError(f"n={n}: P_n-L_n has {roots} root(s) in the admissible interval")
    if not L > g:
        raise AssertionError(f"n={n}: the two-zero lower bound must exceed gamma_n")

    # Marginal-cut bound.  If lambda is a smallest row/column sum, then
    # lambda >= 1-U, U^2 <= 2(n-1)delta/[n(1-delta)].  Bernoulli and
    # completing the square reduce the contradiction to eta_marginal > 0.
    alpha = Fraction(2 * (n - 1), n)
    if not alpha * g < 1 - g:
        raise AssertionError(f"n={n}: marginal dilation can reach one")
    eta_marginal = L - g - Fraction(n * (n - 1), 2) * L * L / (1 - g)
    if eta_marginal <= 0:
        raise AssertionError(f"n={n}: marginal-cut inequality failed")

    # Proper cut.  A tight (I,J) cut with complement sizes u,v >= 1
    # forces the dominated doubly stochastic B to have a u x v zero block.
    # Hwang's staircase-face theorem gives
    #   per(B) >= nu = gamma_{n-u} gamma_{n-v} / gamma_k,
    # where k=n-u-v.  The scalar contradiction reduces to eta>0.
    if not n * g < 1 - g:
        raise AssertionError(f"n={n}: proper-cut dilation can reach one")

    checked = 0
    min_eta: Fraction | None = None
    min_data: tuple[int, int, int, Fraction] | None = None
    for u in range(1, n):
        for v in range(1, n - u):
            k = n - u - v
            nu = gamma(n - u) * gamma(n - v) / gamma(k)

            # Independent exact reconstruction from the barycenter count:
            # N=(n-v)!(n-u)!/k! supported permutations, each of weight
            # (n-v)^(-u)(n-u)^(-v)[k/((n-u)(n-v))]^k.
            supported = Fraction(factorial(n - v) * factorial(n - u), factorial(k))
            bary_weight = (
                Fraction(1, (n - v) ** u)
                * Fraction(1, (n - u) ** v)
                * Fraction(k**k, ((n - u) * (n - v)) ** k)
            )
            if supported * bary_weight != nu:
                raise AssertionError(f"n={n},u={u},v={v}: staircase barycenter formula failed")

            eta = nu - g - Fraction(n**3, 4 * k * k) * nu * nu / (1 - g)
            if eta <= 0:
                raise AssertionError(
                    f"n={n},u={u},v={v},k={k}: proper-cut inequality failed"
                )
            checked += 1
            if min_eta is None or eta < min_eta:
                min_eta = eta
                min_data = (u, v, k, nu)

    assert min_eta is not None and min_data is not None
    u, v, k, nu = min_data

    print(f"n={n}: CERTIFIED")
    print(f"  gamma_n = {g}")
    print(f"  two-zero L_n = {L} ~= {decimal_string(L)}")
    print(f"  Sturm roots of P_n-L_n on [{domain_left},{domain_right}] = {roots}")
    print(f"  endpoint margins = {q_left}, {q_right}")
    print(
        "  marginal eta = "
        f"{eta_marginal} ~= {decimal_string(eta_marginal)}"
    )
    print(f"  proper cuts checked = {checked}")
    print(
        f"  smallest proper eta at (u,v,k)=({u},{v},{k}), "
        f"nu={nu}: {min_eta} ~= {decimal_string(min_eta)}"
    )


def main() -> None:
    for case in CASES:
        verify_case(case)
    print("ALL CASES CERTIFIED")


if __name__ == "__main__":
    main()
