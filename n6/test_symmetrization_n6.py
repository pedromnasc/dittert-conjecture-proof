#!/usr/bin/env python3
"""Numerically stress-test whether full face averaging increases F."""
from __future__ import annotations

import argparse

import numpy as np
from scipy import optimize

import optimize_face_n6 as face
import search_sos_n6 as base


SCALE = 1e8


def variable_orbits() -> list[np.ndarray]:
    group = base.variable_group()
    result: list[np.ndarray] = []
    seen: set[int] = set()
    for variable in range(base.NV):
        if variable in seen:
            continue
        orbit = np.unique(group[:, variable]).astype(np.int32)
        result.append(orbit)
        seen.update(int(item) for item in orbit)
    if sorted(len(orbit) for orbit in result) != [2, 16, 16]:
        raise AssertionError("unexpected variable orbits")
    return result


ORBITS = variable_orbits()


def average(vector: np.ndarray) -> np.ndarray:
    result = vector.copy()
    for orbit in ORBITS:
        result[orbit] = vector[orbit].mean()
    return result


def gap_gradient(vector: np.ndarray) -> tuple[float, np.ndarray]:
    averaged = average(vector)
    averaged_value, averaged_gradient = face.value_gradient(averaged)
    value, gradient = face.value_gradient(vector)
    projected_gradient = average(averaged_gradient)
    return averaged_value - value, projected_gradient - gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument("--minimum-value", type=float)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    best: tuple[float, np.ndarray] | None = None
    concentrations = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 10.0)
    for index in range(args.starts):
        concentration = concentrations[index % len(concentrations)]
        random_point = rng.dirichlet(np.full(base.NV, concentration))
        if args.minimum_value is None:
            start = random_point
        else:
            hard_point = face.symmetric_two_zero_start()
            mixing = rng.uniform(0, 0.5)
            start = (1 - mixing) * hard_point + mixing * random_point
            while face.value_gradient(start)[0] < args.minimum_value:
                mixing /= 2
                start = (1 - mixing) * hard_point + mixing * random_point
        constraints = [
            {
                "type": "eq",
                "fun": lambda vector: vector.sum() - 1,
                "jac": lambda vector: np.ones(base.NV),
            }
        ]
        if args.minimum_value is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda vector: face.value_gradient(vector)[0]
                    - args.minimum_value,
                    "jac": lambda vector: face.value_gradient(vector)[1],
                }
            )
        result = optimize.minimize(
            lambda vector: SCALE * gap_gradient(vector)[0],
            start,
            jac=lambda vector: SCALE * gap_gradient(vector)[1],
            method="SLSQP",
            bounds=optimize.Bounds(np.zeros(base.NV), np.ones(base.NV)),
            constraints=constraints,
            options={"ftol": 1e-13, "maxiter": args.maxiter, "disp": False},
        )
        candidate = np.maximum(result.x, 0)
        candidate /= candidate.sum()
        value, _ = face.value_gradient(candidate)
        if args.minimum_value is not None and value < args.minimum_value - 1e-10:
            continue
        gap, gradient = gap_gradient(candidate)
        if best is None or gap < best[0]:
            best = (gap, candidate.copy())
        if index % 10 == 0 or gap < -1e-12:
            positive = candidate > 1e-9
            print(
                f"start={index} success={result.success} iterations={result.nit} "
                f"gap={gap:.16g} positive={positive.sum()} "
                f"gradient-spread={np.ptp(gradient[positive]):.6g}",
                flush=True,
            )
        if gap < -1e-10:
            print("COUNTEREXAMPLE")
            print(np.array2string(face.matrix_from_vector(candidate), precision=12))
            return
    assert best is not None
    print(f"smallest optimized gap={best[0]:.17g}")
    print(np.array2string(face.matrix_from_vector(best[1]), precision=12))


if __name__ == "__main__":
    main()
