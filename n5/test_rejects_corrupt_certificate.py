#!/usr/bin/env python3
"""Check that the exact verifier rejects a structurally valid corrupt certificate."""
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

    # Preserve the archive schema, integer dtype, shape, triangularity, and
    # positive diagonal while making the exact residual polynomial invalid.
    payload["Lvertex_num"][0, 0, 0] += 1 << 20

    with tempfile.TemporaryDirectory(prefix="dittert-n5-corrupt-") as temporary:
        corrupt = Path(temporary) / "corrupt_certificate.npz"
        np.savez_compressed(corrupt, **payload)
        completed = subprocess.run(
            [sys.executable, str(directory / "verify_primary.py"), str(corrupt)],
            check=False,
            capture_output=True,
            text=True,
        )
    assert completed.returncode != 0
    assert "CERTIFIED" not in completed.stdout
    assert "assert minimum is not None and minimum > 0" in completed.stderr
    print("CORRUPT CERTIFICATE REJECTED")


if __name__ == "__main__":
    main()
