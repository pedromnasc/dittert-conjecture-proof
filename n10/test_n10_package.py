#!/usr/bin/env python3
"""Integration and safety tests for the two n=10 verifiers."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
PRIMARY = DIRECTORY / "verify_dittert_n10.py"
AUDIT = DIRECTORY / "audit_dittert_n10_stdlib.py"


def run_script(path: Path, optimized: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-I"]
    if optimized:
        command.append("-O")
    command.append(str(path))
    return subprocess.run(
        command,
        cwd=DIRECTORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


class N10PackageTests(unittest.TestCase):
    def test_primary_and_audit_certify(self) -> None:
        primary = run_script(PRIMARY)
        self.assertEqual(primary.returncode, 0, primary.stderr)
        self.assertEqual(primary.stdout.splitlines()[-1], "CERTIFIED")
        self.assertIn("proper cuts checked = 36", primary.stdout)

        audit = run_script(AUDIT)
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertEqual(audit.stdout.splitlines()[-1], "INDEPENDENT AUDIT CERTIFIED")
        self.assertIn("exact Ryser interpolation samples = 11", audit.stdout)
        self.assertIn("two-zero Bernstein subintervals = 256", audit.stdout)

    def test_optimized_mode_has_identical_output(self) -> None:
        for script in (PRIMARY, AUDIT):
            with self.subTest(script=script.name):
                ordinary = run_script(script)
                optimized = run_script(script, optimized=True)
                self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
                self.assertEqual(optimized.returncode, 0, optimized.stderr)
                self.assertEqual(optimized.stdout, ordinary.stdout)

    def test_inflated_face_bound_is_rejected(self) -> None:
        mutations = (
            (
                PRIMARY,
                "lower_bound = Fraction(36_719, 100_000_000)",
                "lower_bound = Fraction(40_000, 100_000_000)",
            ),
            (
                AUDIT,
                "LOWER_BOUND = Fraction(36_719, 100_000_000)",
                "LOWER_BOUND = Fraction(40_000, 100_000_000)",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="dittert-n10-test-") as temporary:
            temp_directory = Path(temporary)
            for source_path, old, new in mutations:
                with self.subTest(script=source_path.name):
                    source = source_path.read_text(encoding="utf-8")
                    self.assertEqual(source.count(old), 1)
                    mutant = temp_directory / source_path.name
                    mutant.write_text(source.replace(old, new), encoding="utf-8")
                    result = run_script(mutant)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("\nCERTIFIED\n", f"\n{result.stdout}\n")
                    self.assertNotIn("INDEPENDENT AUDIT CERTIFIED", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
