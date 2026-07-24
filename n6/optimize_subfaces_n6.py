#!/usr/bin/env python3
"""Numerically maximize F after imposing one additional zero of each type."""
from __future__ import annotations

import argparse

import numpy as np
from scipy import optimize

import optimize_face_n6 as face
import search_sos_n6 as base


def orbit_representatives() -> list[tuple[int, int]]:
    group = base.variable_group()
    seen: set[int] = set()
    result = []
    for variable, cell in enumerate(base.ALLOWED):
        if variable in seen:
            continue
        orbit = set(int(item) for item in group[:, variable])
        seen.update(orbit)
        result.append((variable, len(orbit)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    for representative, orbit_size in orbit_representatives():
        lower = np.zeros(base.NV)
        upper = np.ones(base.NV)
        upper[representative] = 0
        best: tuple[float, np.ndarray] | None = None
        for index in range(args.starts):
            if index == 0:
                start = face.symmetric_two_zero_start()
            else:
                start = rng.dirichlet(np.full(base.NV, 0.2 + index % 5))
            start[representative] = 0
            start /= start.sum()
            result = optimize.minimize(
                lambda vector: -face.OBJECTIVE_SCALE * face.value_gradient(vector)[0],
                start,
                jac=lambda vector: -face.OBJECTIVE_SCALE
                * face.value_gradient(vector)[1],
                method="SLSQP",
                bounds=optimize.Bounds(lower, upper),
                constraints={
                    "type": "eq",
                    "fun": lambda vector: vector.sum() - 1,
                    "jac": lambda vector: np.ones(base.NV),
                },
                options={"ftol": 1e-13, "maxiter": 3000, "disp": False},
            )
            if (
                abs(float(result.x.sum()) - 1) > 1e-8
                or result.x.min() < -1e-9
                or result.x[representative] > 1e-9
            ):
                continue
            value, _ = face.value_gradient(result.x)
            if best is None or value > best[0]:
                best = (value, result.x.copy())
        assert best is not None
        matrix = face.matrix_from_vector(best[1])
        print(
            f"extra-zero={base.ALLOWED[representative]} orbit={orbit_size} "
            f"F={best[0]:.17g} gap={base.TARGET_CONSTANT - best[0]:.17g} "
            f"positive={np.count_nonzero(best[1] > 1e-9)}"
        )
        print("row-sums", np.array2string(matrix.sum(axis=1), precision=10))
        print("col-sums", np.array2string(matrix.sum(axis=0), precision=10))
        print(np.array2string(matrix, precision=8, suppress_small=True))


if __name__ == "__main__":
    main()
