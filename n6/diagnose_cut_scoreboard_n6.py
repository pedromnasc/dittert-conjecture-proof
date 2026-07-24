#!/usr/bin/env python3
"""Exact scoreboard of the boundary-cut architecture for ``n=6``.

The audited ``n=8`` proof excludes every boundary maximizer of ``Phi`` by
classifying the maximal Li scaling into *whole*, *marginal*, and *proper*
tight cuts and killing each with an exact univariate inequality (Sturm root
counts and Bernstein coefficients).  This program applies that same
classification to ``n=6`` and reports the **exact margin of every cut**, so the
gaps that keep ``n=6`` open are explicit and individually trackable.

It is a diagnostic, not a proof.  Only exact rational arithmetic feeds the
PASS/OPEN decisions; no floating-point value is used in a decision.  A negative
margin means the corresponding ``n=8`` bound does *not* close that cut at
``n=6``.

Reuses the exact univariate/Sturm/Bernstein machinery of the audited ``n=8``
package.  Run with the discovery venv, e.g.::

    python n6/diagnose_cut_scoreboard_n6.py
"""
from __future__ import annotations

import sys
from fractions import Fraction
from math import isqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "n8"))
import verify_dittert_n8 as exact  # noqa: E402

N = 6
M = N - 2
GAMMA = exact.gamma(N)                       # 5/324
TWO_ZERO_L = Fraction(2, 125)                # per(B) lower bound on the face
# Marginal safe interval endpoints (from explore_li_scaling_n6.py):
STRENGTHENED_SAFE_RIGHT = Fraction(1, 125)   # basic gives 1/200; strengthened 1/125
RISK_LEFT = STRENGTHENED_SAFE_RIGHT
RISK_RIGHT = Fraction(3, 50)
FEASIBLE_RIGHT = Fraction(11, 100)
# n=6 subset-deficit constants  1 - F_d(x) >= c_d x^2  on [-1/4, 1/4]:
SUBSET_C = {1: Fraction(1, 2), 2: Fraction(7, 20), 3: Fraction(8, 25),
            4: Fraction(7, 20), 5: Fraction(1, 2)}
SUBSET_RADIUS = Fraction(1, 4)


def line(tag: str, verdict: str, detail: str) -> None:
    mark = {"PASS": "  ok ", "OPEN": ">>OPEN", "INFO": "  -- "}[verdict]
    print(f"{mark} {tag:<34} {detail}")


def rational_upper_sqrt(value: Fraction, denom: int = 10_000) -> Fraction:
    """Smallest ``k/denom`` whose square is >= ``value`` (exact)."""
    target = value.numerator * denom * denom
    root = isqrt(target // value.denominator)
    while Fraction(root * root, 1) < Fraction(target, value.denominator):
        root += 1
    result = Fraction(root, denom)
    assert result * result >= value
    return result


def gamma_ext(m: int) -> Fraction:
    return Fraction(1) if m <= 1 else exact.gamma(m)


# --------------------------------------------------------------------------- #
# 1. two independent zeroes: symmetric doubly stochastic face (EXACT, closed)
# --------------------------------------------------------------------------- #
def two_zero_block() -> bool:
    permanent = exact.two_zero_permanent_polynomial(N)
    left = Fraction(M - 2, M * M)
    right = Fraction(1, M)
    difference = exact.poly_sub(permanent, (TWO_ZERO_L,))
    roots = exact.sturm_root_count(difference, left, right)
    ends_ok = (exact.poly_eval(difference, left) > 0
               and exact.poly_eval(difference, right) > 0)
    gap = TWO_ZERO_L - GAMMA
    ok = roots == 0 and ends_ok and gap > 0
    line("two-zero permanent gap", "PASS" if ok else "OPEN",
         f"Sturm roots(P-L)={roots} on [{left},{right}], "
         f"L-gamma={gap} ~ {float(gap):.3g}")
    return ok


# --------------------------------------------------------------------------- #
# 2. marginal cut: safe outside a risk interval (EXACT); the interior is
#    handed to the support cuts.
# --------------------------------------------------------------------------- #
def strengthened_marginal_numerator(deficit_minus_gamma, common):
    one_minus_a = exact.linear(1, -1)
    q_minus_two = exact.linear(N - 3, -N)
    return exact.trim(exact.poly_add(
        exact.poly_mul(q_minus_two, deficit_minus_gamma),
        exact.poly_add(
            exact.poly_scale(common, -2 * GAMMA),
            exact.poly_scale(
                exact.poly_mul(exact.poly_pow(one_minus_a, N), common),
                (N - 1) * TWO_ZERO_L))))


def marginal_block() -> bool:
    contradiction, deficit_minus_gamma, common = exact.marginal_numerator(
        N, TWO_ZERO_L)
    strengthened = strengthened_marginal_numerator(deficit_minus_gamma, common)
    right_roots = exact.sturm_root_count(contradiction, RISK_RIGHT, FEASIBLE_RIGHT)
    left_roots = exact.sturm_root_count(strengthened, Fraction(0), RISK_LEFT)
    left_ends = (exact.poly_eval(strengthened, Fraction(0)) > 0
                 and exact.poly_eval(strengthened, RISK_LEFT) > 0)
    right_ends = (exact.poly_eval(contradiction, RISK_RIGHT) > 0
                  and exact.poly_eval(contradiction, FEASIBLE_RIGHT) > 0)
    ok = right_roots == 0 and left_roots == 0 and left_ends and right_ends
    line("marginal cut (outside risk)", "PASS" if ok else "OPEN",
         f"safe on [0,{RISK_LEFT}] u [{RISK_RIGHT},{FEASIBLE_RIGHT}], "
         f"roots={left_roots},{right_roots}")
    line("  -> marginal RISK interval", "INFO",
         f"[{RISK_LEFT},{RISK_RIGHT}] must be covered by support cuts below")
    return ok


# --------------------------------------------------------------------------- #
# 3. support cuts (the large-column bound that covers the marginal risk band)
#
# The large-column derivation is only valid where q(a) > 1: that is what caps
# every non-distinguished row sum below one, excludes a full-support column,
# and makes U_z a valid product upper bound (n=8 proof, eq. row-cap).  For n=6,
# q(1/125)-1 < 0, so q > 1 holds only on a strict sub-interval [a_q, 3/50];
# the remaining band [1/125, a_q] is NOT covered by this argument.
# --------------------------------------------------------------------------- #
def q_validity_left() -> Fraction:
    """Rational a_q >= the root of q-1 in the risk band, with q>1 to its right."""
    _, _, _, e_power, q_numerator = exact.support_base_polynomials(N)
    q_minus_one = exact.poly_sub(q_numerator, e_power)
    require = exact.sturm_root_count(q_minus_one, RISK_LEFT, RISK_RIGHT)
    if require != 1:
        raise AssertionError(f"expected one q=1 crossing, found {require}")
    lo, hi = RISK_LEFT, RISK_RIGHT
    for _ in range(80):
        mid = (lo + hi) / 2
        if exact.poly_eval(q_minus_one, mid) > 0:
            hi = mid
        else:
            lo = mid
    # clean rational upper bound on the root: smallest k/2000 that is >= hi
    scale = 2000
    a_q = Fraction(-(-(hi * scale).numerator // (hi * scale).denominator), scale)
    # a_q must have q>1 and no further crossing up to RISK_RIGHT
    if not (a_q >= hi and exact.poly_eval(q_minus_one, a_q) > 0
            and exact.sturm_root_count(q_minus_one, a_q, RISK_RIGHT) == 0):
        raise AssertionError("failed to certify the q>1 sub-interval")
    return a_q


def support_block() -> bool:
    e_value, h_numerator, t_value, e_power, q_numerator = (
        exact.support_base_polynomials(N))
    one_minus_a = exact.linear(1, -1)
    a_q = q_validity_left()
    line("support domain q>1", "OPEN",
         f"valid only on [{a_q}~{float(a_q):.4f}, {RISK_RIGHT}]; "
         f"band [{RISK_LEFT}, {a_q}] is UNCOVERED (q<1 there)")
    all_ok = False  # the uncovered band alone leaves the marginal cut open
    for zero_count in range(1, N - 1):
        support_size = N - zero_count
        staircase = (gamma_ext(N - zero_count) * gamma_ext(N - 1)
                     / gamma_ext(N - zero_count - 1))
        case_bound = max(TWO_ZERO_L, staircase)
        r_numerator = exact.poly_sub(
            exact.poly_mul(
                exact.poly_add((2 - GAMMA,),
                               exact.poly_scale(exact.poly_pow(one_minus_a, N),
                                                case_bound)),
                exact.poly_pow(e_value, N)),
            exact.poly_mul(exact.poly_add(e_value, h_numerator),
                           exact.poly_pow(t_value, N - 1)))
        weighted_q = exact.poly_sub(
            exact.poly_mul(exact.linear(N - 1, 1), q_numerator),
            exact.poly_scale(e_power, support_size - 1))
        left_side = exact.poly_scale(
            exact.poly_mul(exact.poly_pow(q_numerator, N - 1), r_numerator),
            Fraction(zero_count**zero_count))
        right_side = exact.poly_mul(
            exact.poly_mul(one_minus_a,
                           exact.poly_pow(e_value, N + (N - 1) * (support_size - 1))),
            exact.poly_pow(weighted_q, zero_count))
        gap = exact.poly_sub(left_side, right_side)
        # Bernstein enclosure over the VALID domain [a_q, RISK_RIGHT] only.
        splits = 8
        coeffs = []
        for i in range(splits):
            coeffs += exact.power_to_bernstein(
                gap, a_q + (RISK_RIGHT - a_q) * Fraction(i, splits),
                a_q + (RISK_RIGHT - a_q) * Fraction(i + 1, splits))
        bmin = min(coeffs)
        ok = bmin > 0
        line(f"support cut z={zero_count} (M={case_bound}) on [a_q,{RISK_RIGHT}]",
             "PASS" if ok else "OPEN",
             f"Bernstein-min={float(bmin):+.4g}")
    return all_ok


# --------------------------------------------------------------------------- #
# 4. subset-deficit constants (input to the proper-cut sharp bound)
# --------------------------------------------------------------------------- #
def subset_block() -> bool:
    all_ok = True
    square = exact.monomial(2)
    for d, constant in SUBSET_C.items():
        product = exact.subset_product_polynomial(N, d)
        difference = exact.poly_sub(
            exact.poly_sub((Fraction(1),), product),
            exact.poly_scale(square, constant))
        quotient, remainder = exact.poly_divmod(difference, square)
        coeffs = exact.power_to_bernstein(quotient, -SUBSET_RADIUS, SUBSET_RADIUS)
        ok = remainder == (0,) and min(coeffs) > 0
        all_ok &= ok
        line(f"subset-deficit d={d} (c={constant})", "PASS" if ok else "OPEN",
             f"Bernstein-min={float(min(coeffs)):+.4g}")
    return all_ok


# --------------------------------------------------------------------------- #
# 5. proper cuts: crude eta bound AND sharp K bound (both OPEN at n=6)
# --------------------------------------------------------------------------- #
def proper_block() -> bool:
    all_ok = True
    for u in range(1, N):
        for v in range(1, N - u):
            k = N - u - v
            nu = gamma_ext(N - u) * gamma_ext(N - v) / gamma_ext(k)
            # B lies on the two-zero face, so per(B) >= max(nu, 2/125); the n=8
            # proof never needed this because there nu > L on every sharp cut.
            perm_bound = max(nu, TWO_ZERO_L)
            c_uv = SUBSET_C[u] * SUBSET_C[v] / (SUBSET_C[u] + SUBSET_C[v])
            eta = nu - GAMMA - Fraction(N**3, 4 * k * k) * nu * nu / (1 - GAMMA)
            # sharp bound K(s) = perm_bound (1 - s/k)^N - gamma + c_uv s^2
            s_max = rational_upper_sqrt(GAMMA / c_uv)
            K = exact.poly_add(
                exact.poly_scale(
                    exact.poly_pow(exact.linear(1, Fraction(-1, k)), N), perm_bound),
                exact.poly_add((-GAMMA,), (Fraction(0), Fraction(0), c_uv)))
            splits = 8
            coeffs = []
            for i in range(splits):
                coeffs += exact.power_to_bernstein(
                    K, s_max * Fraction(i, splits), s_max * Fraction(i + 1, splits))
            k_min = min(coeffs)
            ok = eta > 0 or k_min > 0
            all_ok &= ok
            verdict = "PASS" if ok else "OPEN"
            note = "" if perm_bound == nu else f"  [uses per(B)>=2/125>{float(nu):.5f}]"
            line(f"proper cut (u,v,k)=({u},{v},{k})", verdict,
                 f"eta={float(eta):+.3g}  sharp-K-min={float(k_min):+.3g}{note}")
    return all_ok


def main() -> None:
    print(f"Dittert n=6 boundary-cut scoreboard   gamma_6={GAMMA} "
          f"~ {float(GAMMA):.10g}")
    print("(exact rational decisions; OPEN = the n=8 bound fails at n=6)\n")
    results = {
        "two-zero (symmetric face)": two_zero_block(),
        "marginal cut": marginal_block(),
        "support cuts": support_block(),
        "subset constants": subset_block(),
        "proper cuts": proper_block(),
    }
    print()
    closed = [name for name, ok in results.items() if ok]
    stuck = [name for name, ok in results.items() if not ok]
    print("CLOSED :", ", ".join(closed) if closed else "(none)")
    print("OPEN   :", ", ".join(stuck) if stuck else "(none)")
    if stuck:
        print("\nn=6 is NOT proved: the boundary-cut architecture has open cuts.")
    else:
        print("\nAll boundary cuts closed (subject to an exact end-to-end verifier).")


if __name__ == "__main__":
    main()
