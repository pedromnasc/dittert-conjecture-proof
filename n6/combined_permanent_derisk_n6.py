#!/usr/bin/env python3
"""Numerically de-risk the combined minimum-permanent lever for the open n=6
boundary cuts.

The audited n=8 proof bounds ``per(B)`` from below by the staircase barycenter
of the *cut's own* zero block (``nu`` for proper cuts, ``M_z`` for support
cuts).  But the dominated doubly stochastic ``B`` always inherits the two
independent zeros of ``A``.  This program measures, for each OPEN n=6 cut,

  * threshold   : the ``per(B)`` lower bound that would just close the cut, and
  * guaranteed  : the minimum permanent that is *guaranteed* for every
                  admissible B, i.e. over the WORST placement of the inherited
                  zeros relative to the cut's own zeros.

The worst-placement caveat is essential.  A ``u x v`` block can already contain
one (or, when ``u>=2`` and ``v>=2``, both) of the two independent zeros, and a
single zero column can contain at most one of them (they need distinct
columns).  So only the zeros that CANNOT be absorbed are guaranteed to add new
constraints; adding two free zeros unconditionally minimizes over a strictly
smaller face and overstates the bound.

This is a NUMERICAL diagnostic (SLSQP permanent minimization with an analytic
gradient), not a proof.  A cut only "closes" here if the guaranteed bound
exceeds the threshold, and even then it still needs an exact minimum-permanent
reduction (in the style of ``verify_dittert_n8.two_zero_permanent_polynomial``)
before it counts.

Findings (after correct absorption handling): NO open cut closes via the
combined bound.  The corner proper cut ``(1,1,4)`` is closed separately and
rigorously by ``per(B) >= 2/125`` alone (see ``diagnose_cut_scoreboard_n6.py``);
the earlier "closures" of ``(1,2,3),(2,1,3)`` and support ``z=2`` were artifacts
of adding two guaranteed-outside zeros. The small-``k`` proper cuts need a
different argument (folding the ``k=1`` cuts into the marginal large-line bound,
and/or a stronger deficit bound than ``c_{u,v} s^2``).

Run with the discovery venv (numpy + scipy)::

    python n6/combined_permanent_derisk_n6.py
"""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "n8"))
import verify_dittert_n8 as exact  # noqa: E402

N = 6
GAMMA = exact.gamma(N)
GAMMA_F = float(GAMMA)
L = Fraction(2, 125)
RISK_L, RISK_R = Fraction(1, 125), Fraction(3, 50)
SUBSET_C = {1: 0.5, 2: 0.35, 3: 0.32, 4: 0.35, 5: 0.5}
PERM = np.array(list(itertools.permutations(range(N))), dtype=np.int64)
PCOLS = np.arange(N)


# --------------------------------------------------------------------------- #
# minimum permanent over a Birkhoff face with a prescribed zero set (numeric)
# --------------------------------------------------------------------------- #
def _per_grad(D):
    sel = D[PCOLS[None, :], PERM]
    pre = np.ones_like(sel); suf = np.ones_like(sel)
    pre[:, 1:] = np.cumprod(sel[:, :-1], 1)
    suf[:, :-1] = np.cumprod(sel[:, :0:-1], 1)[:, ::-1]
    cof = pre * suf
    grad = np.zeros((N, N))
    for r in range(N):
        np.add.at(grad[r], PERM[:, r], cof[:, r])
    return float(np.prod(sel, 1).sum()), grad


def min_permanent(zeros, starts=22, seed=7):
    allowed = [(i, j) for i in range(N) for j in range(N) if (i, j) not in zeros]
    aidx = np.array(allowed); nv = len(allowed)
    amat = np.zeros((2 * N, nv))
    for k, (i, j) in enumerate(allowed):
        amat[i, k] = 1.0; amat[N + j, k] = 1.0
    bvec = np.ones(2 * N)

    def pg(x):
        D = np.zeros((N, N)); D[aidx[:, 0], aidx[:, 1]] = x
        p, g = _per_grad(D)
        return p, g[aidx[:, 0], aidx[:, 1]]

    cons = {"type": "eq", "fun": lambda x: amat @ x - bvec, "jac": lambda x: amat}
    rng = np.random.default_rng(seed); best = np.inf
    for _ in range(starts):
        M = np.zeros((N, N)); M[aidx[:, 0], aidx[:, 1]] = rng.random(nv) + 1e-2
        for _ in range(150):
            M /= M.sum(1, keepdims=True); M /= M.sum(0, keepdims=True)
        res = optimize.minimize(
            lambda x: pg(x)[0], M[aidx[:, 0], aidx[:, 1]], jac=lambda x: pg(x)[1],
            method="SLSQP", bounds=[(0, 1)] * nv, constraints=cons,
            options={"ftol": 1e-13, "maxiter": 400})
        if abs(amat @ res.x - bvec).max() < 1e-7 and res.x.min() > -1e-7:
            best = min(best, pg(res.x)[0])
    return best


# --------------------------------------------------------------------------- #
# proper cuts
# --------------------------------------------------------------------------- #
def proper_threshold(u, v, k):
    c_uv = SUBSET_C[u] * SUBSET_C[v] / (SUBSET_C[u] + SUBSET_C[v])
    s_max = (GAMMA_F / c_uv) ** 0.5
    ss = np.linspace(0, s_max, 3000)
    lo, hi = 0.0, 0.1
    for _ in range(70):
        mid = (lo + hi) / 2
        if (mid * (1 - ss / k) ** N - GAMMA_F + c_uv * ss * ss).min() > 0:
            hi = mid
        else:
            lo = mid
    return hi


L_FLOAT = 2 / 125  # the two-zero bound per(B) >= 2/125 (always valid)


def worst_one_extra(fixed_zeros, excl_rows, excl_cols, starts=6):
    """Guaranteed bound when the cut pattern can absorb only ONE inherited zero:
    place the single guaranteed-outside zero (independent of the pattern:
    outside excl_rows/excl_cols) in the worst position and take the minimum
    permanent.  Skips degenerate (near-decomposable) placements."""
    best = float("inf")
    for i in range(N):
        for j in range(N):
            if i in excl_rows or j in excl_cols:
                continue
            p = min_permanent(fixed_zeros | {(i, j)}, starts=starts)
            if p > 5e-3:                       # ignore decomposable placements
                best = min(best, p)
    return best


def proper_report():
    print("PROPER CUTS  (sharp bound K(s)=nu(1-s/k)^6 - gamma + c_uv s^2 > 0)")
    print("guaranteed per(B) accounts for the block absorbing an inherited zero")
    print(f"{'(u,v,k)':>9} {'nu':>8} {'need':>8} {'guar.':>8} {'absorb':>7} {'closes':>7}")
    for u in range(1, N):
        for v in range(1, N - u):
            k = N - u - v
            g = lambda m: exact.gamma(m) if m > 1 else 1  # noqa: E731
            nu = float(g(N - u) * g(N - v) / g(k))
            need = proper_threshold(u, v, k)
            brows, bcols = list(range(N - u, N)), list(range(N - v, N))
            block = frozenset((i, j) for i in brows for j in bcols)
            # a u x v block holds two independent zeros iff u>=2 AND v>=2
            if u >= 2 and v >= 2:
                absorb, guar = 2, max(nu, L_FLOAT)      # no zero guaranteed outside
            else:
                absorb = 1
                guar = max(nu, L_FLOAT, worst_one_extra(block, set(brows), set(bcols)))
            print(f"{f'({u},{v},{k})':>9} {nu:>8.5f} {need:>8.5f} {guar:>8.5f} "
                  f"{absorb:>7} {str(guar >= need):>7}")


# --------------------------------------------------------------------------- #
# support cuts (cover the marginal risk band)
# --------------------------------------------------------------------------- #
_E, _H, _T, _EPOW, _Q = exact.support_base_polynomials(N)
_OMA = exact.linear(1, -1)


def _support_gap(case_bound: Fraction, z: int):
    ssz = N - z
    r_num = exact.poly_sub(
        exact.poly_mul(exact.poly_add(
            (2 - GAMMA,), exact.poly_scale(exact.poly_pow(_OMA, N), case_bound)),
            exact.poly_pow(_E, N)),
        exact.poly_mul(exact.poly_add(_E, _H), exact.poly_pow(_T, N - 1)))
    wq = exact.poly_sub(exact.poly_mul(exact.linear(N - 1, 1), _Q),
                        exact.poly_scale(_EPOW, ssz - 1))
    left = exact.poly_scale(exact.poly_mul(exact.poly_pow(_Q, N - 1), r_num), Fraction(z**z))
    right = exact.poly_mul(exact.poly_mul(_OMA, exact.poly_pow(_E, N + (N - 1) * (ssz - 1))),
                           exact.poly_pow(wq, z))
    return exact.poly_sub(left, right)


def support_threshold(z):
    xs = [RISK_L + (RISK_R - RISK_L) * Fraction(i, 400) for i in range(401)]
    lo, hi = 0.0, 0.25
    for _ in range(45):
        mid = (lo + hi) / 2
        gap = _support_gap(Fraction(mid).limit_denominator(10**8), z)
        if min(float(exact.poly_eval(gap, x)) for x in xs) > 0:
            hi = mid
        else:
            lo = mid
    return hi


def support_report():
    print("\nSUPPORT CUTS  (current M_z = L = 2/125 for z=1,2)")
    print("a single zero column can absorb only ONE inherited zero (they need")
    print("distinct columns), so only one extra zero is guaranteed outside it.")
    print(f"{'z':>3} {'need':>8} {'col':>8} {'guar.':>8} {'closes':>7}")
    for z in (1, 2):
        need = support_threshold(z)
        column = frozenset((i, 0) for i in range(z))
        col = min_permanent(column)
        guar = max(col, L_FLOAT,
                   worst_one_extra(column, set(range(z)), {0}))
        print(f"{z:>3} {need:>8.5f} {col:>8.5f} {guar:>8.5f} {str(guar >= need):>7}")


def main():
    print(f"n=6 gamma={GAMMA_F:.7g}  (numerical de-risk, not a proof)\n")
    proper_report()
    support_report()
    print("\nWith correct absorption handling, no open cut closes via the")
    print("combined bound. (1,1,4) is closed separately by per(B)>=2/125 alone.")


if __name__ == "__main__":
    main()
