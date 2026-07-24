#!/usr/bin/env python3
"""Numerically de-risk the combined minimum-permanent lever for the open n=6
boundary cuts.

The audited n=8 proof bounds ``per(B)`` from below by the staircase barycenter
of the *cut's own* zero block (``nu`` for proper cuts, ``M_z`` for support
cuts).  But the dominated doubly stochastic ``B`` always inherits the two
independent zeros of ``A``.  This program measures, for each OPEN n=6 cut,

  * threshold  : the ``per(B)`` lower bound that would just close the cut, and
  * combined   : the minimum permanent over doubly stochastic 6x6 matrices that
                 carry the cut's zeros PLUS two independent zeros,

so we know exactly which cuts the combined bound can close.

This is a NUMERICAL diagnostic (SLSQP permanent minimization with an analytic
gradient), not a proof.  A cut that "closes" here still needs an exact
minimum-permanent reduction (in the style of
``verify_dittert_n8.two_zero_permanent_polynomial``) before it counts.

Findings (default settings): the combined bound closes support ``z=2`` and the
three "corner" proper cuts ``(1,1,4),(1,2,3),(2,1,3)``; support ``z=1`` sits on
the threshold; the seven small-``k`` proper cuts need a ~50% permanent lift the
combined bound cannot supply and require a different argument (folding the
``k=1`` cuts into the marginal large-line bound, and/or a stronger deficit
bound than ``c_{u,v} s^2``).

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


def proper_report():
    print("PROPER CUTS  (sharp bound K(s)=nu(1-s/k)^6 - gamma + c_uv s^2 > 0)")
    print(f"{'(u,v,k)':>9} {'nu_now':>8} {'need':>8} {'blk':>8} {'+2z':>8} "
          f"{'swallow':>7} {'closes':>7}")
    for u in range(1, N):
        for v in range(1, N - u):
            k = N - u - v
            g = lambda m: exact.gamma(m) if m > 1 else 1  # noqa: E731
            nu_now = float(g(N - u) * g(N - v) / g(k))
            need = proper_threshold(u, v, k)
            brows, bcols = list(range(N - u, N)), list(range(N - v, N))
            block = frozenset((i, j) for i in brows for j in bcols)
            blk = min_permanent(block)
            free_r = [i for i in range(N) if i not in brows]
            free_c = [j for j in range(N) if j not in bcols]
            if len(free_r) >= 2 and len(free_c) >= 2:
                extra = {(free_r[0], free_c[0]), (free_r[1], free_c[1])}
            else:  # u>=2 and v>=2: two independent zeros fit inside the block
                extra = set()
            comb = min_permanent(block | frozenset(extra)) if extra else blk
            swallow = (u >= 2 and v >= 2)
            usable = blk if swallow else comb
            print(f"{f'({u},{v},{k})':>9} {nu_now:>8.5f} {need:>8.5f} {blk:>8.5f} "
                  f"{comb:>8.5f} {str(swallow):>7} {str(usable >= need):>7}")


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
    print(f"{'z':>3} {'need':>8} {'col':>8} {'+2z':>8} {'closes':>7}")
    for z in (1, 2):
        need = support_threshold(z)
        col = min_permanent(frozenset((i, 0) for i in range(z)))
        comb = min_permanent(frozenset((i, 0) for i in range(z)) | {(z, 1), (z + 1, 2)})
        print(f"{z:>3} {need:>8.5f} {col:>8.5f} {comb:>8.5f} {str(comb >= need):>7}")


def main():
    print(f"n=6 gamma={GAMMA_F:.7g}  (numerical de-risk, not a proof)\n")
    proper_report()
    support_report()
    print("\nCombined bound closes: support z=2 and proper (1,1,4),(1,2,3),(2,1,3).")
    print("Open after it: support z=1 (borderline) and proper k=1,2 cuts.")


if __name__ == "__main__":
    main()
