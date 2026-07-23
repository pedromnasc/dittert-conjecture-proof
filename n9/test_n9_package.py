#!/usr/bin/env python3
"""Integration, optimized-mode, and mutation tests for the n=9 package."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
PRIMARY = DIRECTORY / "verify_dittert_n9.py"
AUDIT = DIRECTORY / "audit_dittert_n9_stdlib.py"


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
        timeout=90,
    )


class N9PackageTests(unittest.TestCase):
    def test_primary_and_audit_certify(self) -> None:
        primary = run_script(PRIMARY)
        self.assertEqual(primary.returncode, 0, primary.stderr)
        self.assertEqual(primary.stdout.splitlines()[-1], "CERTIFIED")
        self.assertIn("support cases checked = 7", primary.stdout)
        self.assertIn("proper cuts checked = 28", primary.stdout)

        audit = run_script(AUDIT)
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertEqual(audit.stdout.splitlines()[-1], "INDEPENDENT AUDIT CERTIFIED")
        self.assertIn("exact Ryser interpolation samples = 10", audit.stdout)
        self.assertIn("exact samples per support polynomial = 83", audit.stdout)

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
                "lower_bound = Fraction(47_533, 50_000_000)",
                "lower_bound = Fraction(48_000, 50_000_000)",
            ),
            (
                AUDIT,
                "LOWER_BOUND = Fraction(47_533, 50_000_000)",
                "LOWER_BOUND = Fraction(48_000, 50_000_000)",
            ),
        )
        self._require_mutations_rejected(mutations)

    def test_weakened_support_accounting_is_rejected(self) -> None:
        mutations = (
            (
                PRIMARY,
                "poly_scale(e_eighth, support_size - 1)",
                "poly_scale(e_eighth, support_size - 2)",
            ),
            (
                AUDIT,
                "(support_size - 1) * e_value**8",
                "(support_size - 2) * e_value**8",
            ),
        )
        self._require_mutations_rejected(mutations)

    def _require_mutations_rejected(
        self, mutations: tuple[tuple[Path, str, str], ...]
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="dittert-n9-test-") as temporary:
            temp_directory = Path(temporary)
            for source_path, old, new in mutations:
                with self.subTest(script=source_path.name, mutation=old):
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
