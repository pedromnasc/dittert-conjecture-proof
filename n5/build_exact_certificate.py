#!/usr/bin/env python3
"""Round a successful numerical n=5 search result to an exact certificate.

This is a discovery/reconstruction utility.  The output is accepted as a
proof only after ``verify_primary.py`` checks the resulting polynomial
identity and all residual coefficients using arbitrary-precision integers.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np


DENOMINATOR_EXPONENT = 28
DIAGONAL_SHIFT = 1e-8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("search_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = np.load(args.search_result, allow_pickle=False)
    allowed = np.asarray(source["allowed"], dtype=np.int8)
    group = np.asarray(source["group"], dtype=np.int16)
    vertex_representatives = np.asarray(source["vertex_representatives"], dtype=np.int16)
    m2 = np.asarray(
        list(itertools.combinations_with_replacement(range(len(allowed)), 2)),
        dtype=np.int16,
    )
    factors = []
    scale = 1 << DENOMINATOR_EXPONENT
    for number in range(len(vertex_representatives)):
        q_orbit = source[f"vertex_q_orbit_{number}"]
        q_values = source[f"vertex_q_values_{number}"]
        gram = q_values[q_orbit] + DIAGONAL_SHIFT * np.eye(len(m2))
        factor = np.linalg.cholesky(gram)
        integer_factor = np.rint(scale * factor).astype(np.int64)
        assert np.all(np.triu(integer_factor, 1) == 0)
        assert np.all(np.diag(integer_factor) > 0)
        factors.append(integer_factor)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        denominator_exponent=np.asarray(DENOMINATOR_EXPONENT, dtype=np.int16),
        pattern=np.asarray("10000/01000/00000/00000/00000"),
        allowed=allowed,
        group=group,
        vertex_representatives=vertex_representatives,
        m2=m2,
        Lvertex_num=np.asarray(factors, dtype=np.int64),
    )
    print(args.output)


if __name__ == "__main__":
    main()
