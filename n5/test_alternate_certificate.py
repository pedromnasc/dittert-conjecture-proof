#!/usr/bin/env python3
"""Check that the exact verifier accepts a non-bit-identical valid certificate."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    directory = Path(__file__).resolve().parent
    source_path = directory / "dittert_n5_exact_certificate.npz"
    with np.load(source_path, allow_pickle=False) as source:
        payload = {name: source[name].copy() for name in source.files}

    # A tiny deterministic perturbation models platform-dependent changes in
    # rounded Cholesky factors.  The new archive has a different digest and
    # extrema, while its exact residual coefficients remain strictly positive.
    factors = payload["Lvertex_num"]
    factors += np.sign(factors) * (np.abs(factors) // 1_000_000)

    with tempfile.TemporaryDirectory(prefix="dittert-n5-alternate-") as temporary:
        alternate = Path(temporary) / "alternate_certificate.npz"
        np.savez_compressed(alternate, **payload)
        completed = subprocess.run(
            [sys.executable, str(directory / "verify_primary.py"), str(alternate)],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    assert "bundled extrema regression: skipped (alternate valid certificate)" in completed.stdout
    assert completed.stdout.rstrip().endswith("CERTIFIED")
    print("ALTERNATE CERTIFICATE ACCEPTED")


if __name__ == "__main__":
    main()
