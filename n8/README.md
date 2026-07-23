# Exact working proof for Dittert's conjecture in dimension 8

**Status:** exact, reproducible working proof; not yet independently peer
reviewed.

For a nonnegative `8 x 8` matrix `A` whose entries sum to `8`, this package
proves

\[
\Phi(A)=\prod_i r_i+\prod_j c_j-\operatorname{per}(A)
\le 2-\frac{8!}{8^8},
\]

with equality only at the uniform matrix.

## Reproduce

Python 3.10 or newer is sufficient; no third-party package is needed.

```bash
python3 -I verify_dittert_n8.py
python3 -I audit_dittert_n8_stdlib.py
python3 -I test_n8_package.py
```

The verifier final lines must be `CERTIFIED` and `INDEPENDENT AUDIT
CERTIFIED`, and the tests must report `OK`. The test suite confirms identical
ordinary and `python -O` output and requires the implementations to reject
three mutations targeting the face bound, marginal support bound, and
exceptional proper-cut estimate.

Verify all packaged files with:

```bash
sha256sum -c SHA256SUMS
```

## Files

- `dittert_n8_exact_proof_2026-07-23.pdf` and `.tex` — formal proof note.
- `verify_dittert_n8.py` — primary exact standard-library verifier.
- `audit_dittert_n8_stdlib.py` — independent exact standard-library audit.
- `test_n8_package.py` — integration, optimized-mode, and mutation tests.
- `verify_dittert_n8.log` and `audit_dittert_n8_stdlib.log` — captured output.
- `SOURCE_NOTES.md` — dependency and verification-scope notes.
- `SHA256SUMS` — package integrity manifest.

## Proof architecture

The maximal Li scaling has a whole, marginal, or proper tight cut. Whole cuts
reduce to the doubly stochastic equality case.

The general staircase estimate excludes 15 of the 21 proper cuts. The six
cuts with a one-dimensional core require a sharper argument: exact quadratic
deficit estimates for grouped row and column sums reduce them to three
degree-eight Bernstein certificates (the other three follow by
transposition).

For a marginal cut, write the minimum row as `1-a`. Stationarity forces a
large column that contains this row. Cofactor nonnegativity bounds every other
row meeting that column. The zero entries in the column also put the
dominated doubly stochastic matrix on a smaller staircase face. Combining
these two effects yields six degree-65 Bernstein certificates, one for each
possible nontrivial support size.

The argument needs only the exact two-independent-zero bound

\[
\operatorname{per}(B)>\frac{12249}{5000000}>\gamma_8.
\]

It does **not** assume a four-zero permanent-minimum theorem.

## Independent implementation

The audit imports no code from the primary verifier. It uses literal exact
Ryser evaluations for the two-zero face and reconstructs the marginal,
support, subset-deficit, and exceptional-cut polynomials from exact scalar
values. It then uses an independently implemented exact Bernstein
conversion. The primary instead uses Sturm sequences for the two-zero face
and safe marginal intervals and direct polynomial expansions for the
remaining certificates.

The proof note and `SOURCE_NOTES.md` state the published structural inputs and
the exact scope of the computer checks.
