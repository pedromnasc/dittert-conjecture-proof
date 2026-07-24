#!/usr/bin/env python3
"""Round a numerical n=6 SDP solution to an exact dyadic SOS certificate.

This is a reconstruction utility, not a verifier.  It projects the numerical
Gram matrices onto the positive-semidefinite cone, factors them, and rounds
the factors.  ``verify_primary.py`` subsequently checks the resulting
polynomial identity using Python integers.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import search_pair_sos_n6 as pair_search
import search_sos_n6 as base


DEFAULT_EXPONENT = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def full_coordinate_index() -> np.ndarray:
    upper_index = {
        (int(i), int(j)): index
        for index, (i, j) in enumerate(pair_search.LINEAR_UPPER)
    }
    result = np.empty((base.NV, base.NV), dtype=np.int32)
    for i in range(base.NV):
        for j in range(base.NV):
            result[i, j] = upper_index[tuple(sorted((i, j)))]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("search_result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exponent", type=int, default=DEFAULT_EXPONENT)
    parser.add_argument(
        "--eigenvalue-floor",
        type=float,
        default=0.0,
        help="nonnegative floor applied before Gram factorization",
    )
    args = parser.parse_args()
    if args.exponent <= 0 or args.exponent >= 62:
        raise ValueError("the dyadic exponent must lie between 1 and 61")
    if args.eigenvalue_floor < 0:
        raise ValueError("the eigenvalue floor must be nonnegative")

    source_path = args.search_result.resolve()
    source = np.load(source_path, allow_pickle=False)
    expected = {
        "allowed",
        "group",
        "multipliers",
        "degree_six_representatives",
        "degree_six_sizes",
        "target",
        "residual",
        "margin",
        "minimum_eigenvalues",
        "gram_coordinates",
    }
    if set(source.files) != expected:
        raise ValueError(f"unexpected search archive fields: {set(source.files)}")

    allowed = np.asarray(source["allowed"], dtype=np.int8)
    group = np.asarray(source["group"], dtype=np.int16)
    multipliers = np.asarray(source["multipliers"], dtype=np.int16)
    coordinates = np.asarray(source["gram_coordinates"], dtype=float)
    if allowed.shape != (base.NV, 2):
        raise ValueError("unexpected allowed-variable array")
    if group.shape != (2304, base.NV):
        raise ValueError("unexpected symmetry-group array")
    if multipliers.shape != (302, 4):
        raise ValueError("unexpected multiplier array")
    if coordinates.shape != (302, len(pair_search.LINEAR_UPPER)):
        raise ValueError("unexpected Gram-coordinate array")

    grams = coordinates[:, full_coordinate_index()]
    scale = 1 << args.exponent
    factors = []
    numerical_minima = []
    maximum_factor_entry = 0
    for gram in grams:
        eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        numerical_minima.append(float(eigenvalues[0]))
        clipped = np.maximum(eigenvalues, args.eigenvalue_floor)
        factor = eigenvectors * np.sqrt(clipped)
        integer_factor = np.rint(scale * factor).astype(np.int64)
        factors.append(integer_factor)
        maximum_factor_entry = max(
            maximum_factor_entry, int(np.max(np.abs(integer_factor)))
        )

    factor_array = np.asarray(factors, dtype=np.int64)
    if factor_array.shape != (302, base.NV, base.NV):
        raise AssertionError("unexpected rounded-factor shape")
    if maximum_factor_entry >= (1 << 62):
        raise OverflowError("rounded factor does not fit safely in int64")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        denominator_exponent=np.asarray(args.exponent, dtype=np.int16),
        pattern=np.asarray("100000/010000/000000/000000/000000/000000"),
        allowed=allowed,
        group=group,
        multipliers=multipliers,
        factor_numerators=factor_array,
        search_result_sha256=np.asarray(sha256(source_path)),
    )
    print(args.output)
    print("search SHA-256:", sha256(source_path))
    print("dyadic denominator: 2^", args.exponent, sep="")
    print("minimum numerical Gram eigenvalue:", min(numerical_minima))
    print("largest factor numerator:", maximum_factor_entry)


if __name__ == "__main__":
    main()
