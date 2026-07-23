# Exact working proof for Dittert's conjecture in dimension 9

**Status:** exact, reproducible working proof; not yet independently peer
reviewed.

For a nonnegative `9 x 9` matrix `A` whose entries sum to `9`, this package
proves

\[
\Phi(A)=\prod_i r_i+\prod_j c_j-\operatorname{per}(A)
\le 2-\frac{9!}{9^9},
\]

with equality only at the uniform matrix.

## Reproduce

Python 3.10 or newer is sufficient; no third-party package is needed.

```bash
python3 -I verify_dittert_n9.py
python3 -I audit_dittert_n9_stdlib.py
python3 -I test_n9_package.py
```

The verifier final lines must be `CERTIFIED` and `INDEPENDENT AUDIT
CERTIFIED`, and the tests must report `OK`. The test suite confirms identical
ordinary and `python -O` output and requires both implementations to reject
false face-bound and support-accounting mutations.

After the manifest has been generated, verify all packaged files with:

```bash
sha256sum -c SHA256SUMS
```

## Files

- `dittert_n9_exact_proof_2026-07-23.pdf` and `.tex` — formal proof note.
- `verify_dittert_n9.py` — primary exact standard-library verifier.
- `audit_dittert_n9_stdlib.py` — independent exact standard-library audit.
- `test_n9_package.py` — integration, optimized-mode, and mutation tests.
- `verify_dittert_n9.log` and `audit_dittert_n9_stdlib.log` — captured output.
- `SOURCE_NOTES.md` — dependency and verification-scope notes.
- `SHA256SUMS` — package integrity manifest.

## Proof architecture

The maximal Li scaling has a whole, marginal, or proper tight cut. Whole cuts
reduce to the doubly stochastic equality case. The 28 proper cuts are excluded
by exact staircase-face inequalities.

For a marginal cut, write the minimum row as `1-a`. Stationarity forces a
column of sum at least `1+h(a)` that contains this minimum row. Cofactor
nonnegativity then bounds every other row meeting that column. Full
indecomposability leaves seven possible nontrivial column support sizes; exact
degree-82 Bernstein certificates exclude all seven on the only interval where
the simpler scalar deficit inequality fails.

The argument needs only the exact two-independent-zero bound

\[
\operatorname{per}(B)>\frac{47533}{50000000}>\gamma_9.
\]

It does **not** assume a three-zero permanent-minimum theorem.

## Independent implementation

The audit imports no code from the primary verifier. It reconstructs the
two-zero polynomial from ten literal exact Ryser evaluations, reconstructs the
degree-18 marginal numerator from exact values, and independently reconstructs
each degree-82 support polynomial from 83 exact scalar values. It proves all
required positivity statements through exact Bernstein coefficients, whereas
the primary verifier uses Sturm sequences for the face and safe marginal
intervals and direct polynomial expansions for the support cases.

The proof note and `SOURCE_NOTES.md` state the published structural inputs and
the exact scope of the computer checks.
