#!/usr/bin/env python3
"""Exact/numerical diagnostics for adapting the n=8 Li-scaling proof to n=6.

This is an exploratory program.  It imports the exact univariate arithmetic
from the audited n=8 package, but it reports failed inequalities instead of
claiming a certificate.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "n8"))
import verify_dittert_n8 as exact  # noqa: E402


N = 6
M = N - 2
GAMMA = exact.gamma(N)
LOWER_BOUND = Fraction(2, 125)
RISK_LEFT = Fraction(1, 200)
RISK_RIGHT = Fraction(3, 50)
STRENGTHENED_RISK_LEFT = Fraction(1, 125)
FEASIBLE_RIGHT = Fraction(11, 100)


def as_float(poly: exact.Poly, value: float) -> float:
    return float(exact.poly_eval(poly, Fraction(value)))


def bernstein_minimum(poly: exact.Poly, left: Fraction, right: Fraction) -> Fraction:
    return min(exact.power_to_bernstein(poly, left, right))


def strengthened_marginal_numerator(
    deficit_minus_gamma: exact.Poly, common: exact.Poly
) -> exact.Poly:
    """Clear denominators in the first-variation Li contradiction.

    For a marginal tight cut, ``lambda = 1-a`` and ``k = 1``.  The
    strengthened inequality implies

        (n*lambda-3) delta
            <= (n*lambda-1) gamma_n - (n-1) lambda^n L_n.

    Combining this with ``delta >= D(a)`` gives the strict contradiction

        (n*lambda-3) D(a) - (n*lambda-1) gamma_n
            + (n-1) lambda^n L_n > 0.

    ``deficit_minus_gamma/common`` is ``D(a)-gamma_n``.  The returned
    polynomial is therefore the numerator after multiplication by the
    positive common denominator.
    """
    one_minus_a = exact.linear(1, -1)
    q_minus_two = exact.linear(N - 3, -N)
    return exact.trim(
        exact.poly_add(
            exact.poly_mul(q_minus_two, deficit_minus_gamma),
            exact.poly_add(
                exact.poly_scale(common, -2 * GAMMA),
                exact.poly_scale(
                    exact.poly_mul(exact.poly_pow(one_minus_a, N), common),
                    (N - 1) * LOWER_BOUND,
                ),
            ),
        )
    )


def two_zero_diagnostic() -> None:
    permanent = exact.two_zero_permanent_polynomial(N)
    left = Fraction(M - 2, M * M)
    right = Fraction(1, M)
    difference = exact.poly_sub(permanent, (LOWER_BOUND,))
    roots = exact.sturm_root_count(difference, left, right)
    minimum = optimize.minimize_scalar(
        lambda value: as_float(permanent, value),
        bounds=(float(left), float(right)),
        method="bounded",
        options={"xatol": 1e-15},
    )
    print(
        "two-zero:",
        f"interval=[{left},{right}]",
        f"roots(P-L)={roots}",
        f"numeric-min={minimum.fun:.15g}",
        f"at={minimum.x:.15g}",
        f"L-gamma={float(LOWER_BOUND - GAMMA):.15g}",
    )


def marginal_diagnostic() -> None:
    contradiction, deficit_minus_gamma, common = exact.marginal_numerator(
        N, LOWER_BOUND
    )
    strengthened = strengthened_marginal_numerator(deficit_minus_gamma, common)
    safe_intervals = (
        (Fraction(0), RISK_LEFT),
        (RISK_RIGHT, FEASIBLE_RIGHT),
    )
    print(
        "marginal:",
        "safe-roots=",
        [exact.sturm_root_count(contradiction, *interval) for interval in safe_intervals],
        "safe-endpoints=",
        [
            float(exact.poly_eval(contradiction, point) / exact.poly_eval(common, point))
            for interval in safe_intervals
            for point in interval
        ],
        "D(right)-gamma=",
        float(
            exact.poly_eval(deficit_minus_gamma, FEASIBLE_RIGHT)
            / exact.poly_eval(common, FEASIBLE_RIGHT)
        ),
    )
    strengthened_interval = (Fraction(0), STRENGTHENED_RISK_LEFT)
    strengthened_roots = exact.sturm_root_count(strengthened, *strengthened_interval)
    strengthened_endpoints = [
        exact.poly_eval(strengthened, point) for point in strengthened_interval
    ]
    if strengthened_roots or min(strengthened_endpoints) <= 0:
        raise AssertionError("strengthened Li safe interval was not certified")
    first_strengthened_root = optimize.brentq(
        lambda value: as_float(strengthened, value),
        float(STRENGTHENED_RISK_LEFT),
        Fraction(1, 100),
        xtol=1e-15,
    )
    print(
        "strengthened marginal:",
        f"safe=[0,{STRENGTHENED_RISK_LEFT}]",
        f"roots={strengthened_roots}",
        f"first-numeric-root={first_strengthened_root:.15g}",
        f"combined-risk=[{STRENGTHENED_RISK_LEFT},{RISK_RIGHT}]",
    )

    e_value, h_numerator, t_value, e_power, q_numerator = (
        exact.support_base_polynomials(N)
    )
    q_minus_one = exact.poly_sub(q_numerator, e_power)
    print(
        "support q(left)-1=",
        float(
            exact.poly_eval(q_minus_one, RISK_LEFT)
            / exact.poly_eval(e_power, RISK_LEFT)
        ),
    )
    one_minus_a = exact.linear(1, -1)
    for zero_count in range(1, N - 1):
        support_size = N - zero_count
        staircase_bound = (
            exact.gamma(N - zero_count)
            * exact.gamma(N - 1)
            / exact.gamma(N - zero_count - 1)
        )
        case_bound = max(LOWER_BOUND, staircase_bound)
        r_numerator = exact.poly_sub(
            exact.poly_mul(
                exact.poly_add(
                    (2 - GAMMA,),
                    exact.poly_scale(exact.poly_pow(one_minus_a, N), case_bound),
                ),
                exact.poly_pow(e_value, N),
            ),
            exact.poly_mul(
                exact.poly_add(e_value, h_numerator),
                exact.poly_pow(t_value, N - 1),
            ),
        )
        weighted_q = exact.poly_sub(
            exact.poly_mul(exact.linear(N - 1, 1), q_numerator),
            exact.poly_scale(e_power, support_size - 1),
        )
        left_side = exact.poly_scale(
            exact.poly_mul(
                exact.poly_pow(q_numerator, N - 1), r_numerator
            ),
            Fraction(zero_count**zero_count),
        )
        right_side = exact.poly_mul(
            exact.poly_mul(
                one_minus_a,
                exact.poly_pow(
                    e_value, N + (N - 1) * (support_size - 1)
                ),
            ),
            exact.poly_pow(weighted_q, zero_count),
        )
        gap = exact.poly_sub(left_side, right_side)
        coefficients = exact.power_to_bernstein(gap, RISK_LEFT, RISK_RIGHT)
        numerical = optimize.minimize_scalar(
            lambda value: as_float(gap, value),
            bounds=(float(RISK_LEFT), float(RISK_RIGHT)),
            method="bounded",
            options={"xatol": 1e-15},
        )
        print(
            f"support z={zero_count} M={case_bound} degree={len(gap) - 1}",
            f"bernstein-min={float(min(coefficients)):.15g}",
            f"negative={sum(value <= 0 for value in coefficients)}",
            f"numeric-min={numerical.fun:.15g}@{numerical.x:.8g}",
            f"endpoints=({as_float(gap, float(RISK_LEFT)):.8g},"
            f"{as_float(gap, float(RISK_RIGHT)):.8g})",
        )


def proper_cut_diagnostic() -> None:
    print("proper generic eta:")
    failures: list[tuple[int, int, int, Fraction, Fraction]] = []
    for u in range(1, N):
        for v_value in range(1, N - u):
            core = N - u - v_value
            staircase = exact.gamma(N - u) * exact.gamma(N - v_value) / exact.gamma(
                core
            )
            eta = (
                staircase
                - GAMMA
                - Fraction(N**3, 4 * core * core)
                * staircase
                * staircase
                / (1 - GAMMA)
            )
            print(
                f"  (u,v,k)=({u},{v_value},{core})",
                f"nu={float(staircase):.12g}",
                f"eta={float(eta):.12g}",
            )
            if eta <= 0:
                failures.append((u, v_value, core, staircase, eta))
    print("generic failures=", [(u, v, core) for u, v, core, _, _ in failures])

    constants = {
        1: Fraction(1, 2),
        2: Fraction(7, 20),
        3: Fraction(8, 25),
        4: Fraction(7, 20),
        5: Fraction(1, 2),
    }
    radius = Fraction(1, 4)
    print(f"subset diagnostics on [-{radius},{radius}]:")
    for subset_size, constant in constants.items():
        product = exact.subset_product_polynomial(N, subset_size)
        difference = exact.poly_sub(
            exact.poly_sub((Fraction(1),), product),
            exact.poly_scale(exact.monomial(2), constant),
        )
        quotient, remainder = exact.poly_divmod(difference, exact.monomial(2))
        coefficients = exact.power_to_bernstein(quotient, -radius, radius)
        endpoint_margin = min(
            1 - exact.poly_eval(product, endpoint) - GAMMA
            for endpoint in (-radius, radius)
        )
        print(
            f"  d={subset_size} c={constant} remainder={remainder}",
            f"bernstein-min={float(min(coefficients)):.12g}",
            f"endpoint-gamma={float(endpoint_margin):.12g}",
        )

    print("failed-cut direct numerical minima:")
    for u, v_value, core, staircase, _ in failures:
        effective = constants[u] * constants[v_value] / (
            constants[u] + constants[v_value]
        )
        upper = float(np.sqrt(float(GAMMA / effective)))

        def special(value: float) -> float:
            return (
                float(staircase) * (1 - value) ** N
                - float(GAMMA)
                + float(effective) * value * value
            )

        minimum = optimize.minimize_scalar(
            special,
            bounds=(0, upper),
            method="bounded",
            options={"xatol": 1e-15},
        )
        print(
            f"  (u,v,k)=({u},{v_value},{core}) ceff={effective}",
            f"range=[0,{upper:.12g}] min={minimum.fun:.12g}",
            f"at={minimum.x:.12g}",
        )


def main() -> None:
    print(f"n={N} gamma={GAMMA} ~ {float(GAMMA):.15g}")
    exact.verify_two_zero_equalization(N)
    two_zero_diagnostic()
    marginal_diagnostic()
    proper_cut_diagnostic()


if __name__ == "__main__":
    main()
