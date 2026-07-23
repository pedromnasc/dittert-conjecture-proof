# Exact working proof for Dittert's conjecture in dimension 10

**Status:** exact, reproducible working proof; not yet independently peer
reviewed.

For a nonnegative `10 x 10` matrix `A` whose entries sum to `10`, this package
proves

\[
\Phi(A)=\prod_i r_i+\prod_j c_j-\operatorname{per}(A)
\le 2-\frac{10!}{10^{10}},
\]

with equality only at the uniform matrix.

## Files

- `dittert_n10_exact_proof_2026-07-23.pdf` — formal proof note.
- `dittert_n10_exact_proof_2026-07-23.tex` — LaTeX source.
- `verify_dittert_n10.py` — primary exact verifier, standard library only.
- `audit_dittert_n10_stdlib.py` — independent exact verifier, standard
  library only.
- `test_n10_package.py` — integration, optimized-mode, and mutation tests.
- `verify_dittert_n10.log` and `audit_dittert_n10_stdlib.log` — captured clean
  outputs.
- `SOURCE_NOTES.md` — dependency and supplied-artifact provenance notes.
- `SHA256SUMS` — integrity manifest.

## Reproduce

Python 3.10 or newer is sufficient. No third-party package is needed.

```bash
python3 -I verify_dittert_n10.py
python3 -I audit_dittert_n10_stdlib.py
python3 -I test_n10_package.py
```

The verifier final lines must be:

```text
CERTIFIED
INDEPENDENT AUDIT CERTIFIED
```

The tests must report `OK`. Both verifiers deliberately make proof decisions
with explicit conditionals rather than executable `assert` statements. The
test suite confirms that ordinary and `python -O` runs have identical output.

To verify the packaged files first, run:

```bash
sha256sum -c SHA256SUMS
```

## Proof architecture

The proof assumes a boundary global maximizer and applies the largest Li
scaling for which the matrix becomes doubly superstochastic.

- A whole tight cut would make the matrix doubly stochastic and is excluded
  by the equality case of the van der Waerden theorem.
- A proper tight cut forces a zero rectangle in a dominated doubly stochastic
  matrix. Hwang's staircase-face theorem and 36 exact rational inequalities
  exclude all such cuts.
- A marginal tight cut equals a minimum row or column sum `1-a`. The support
  stationarity equation forces an opposite line sum of at least `1+h(a)`.
  The resulting shared row/column product deficit confines `a` to `[0,1/50)`
  and yields one exact degree-20 positivity check.

The permanent gap used in the marginal case is

\[
\operatorname{per}(B)>\frac{36719}{10^8}
>\gamma_{10}
\]

for every doubly stochastic `B` with two prescribed zeroes in distinct rows
and columns.

## Independent implementation

The audit does not import the primary verifier.

| Check | Primary verifier | Independent audit |
| --- | --- | --- |
| Two-zero polynomial | closed block count | 11 literal exact Ryser evaluations plus Lagrange interpolation |
| Face positivity | exact rational Sturm sequence | degree-10 Bernstein coefficients on 256 subintervals |
| Marginal numerator | direct rational polynomial expansion | exact value interpolation at 21 points |
| Marginal positivity | exact rational Sturm sequence | one degree-26 Bernstein expansion |
| Proper cuts | exact `Fraction` loop | separately written exact `Fraction` loop |

The test suite also inflates the asserted face lower bound to a false value
and requires both implementations to reject it.

## Verification scope

The programs verify all finite dimension-specific algebra: equalization of
the two-zero face, permanent reconstruction, both positivity certificates,
the staircase barycenter counts, and all proper-cut inequalities. They do not
reprove the published structural theorems of Hwang, Cheon–Wanless, Li, Minc,
or van der Waerden. Those dependencies are stated precisely in the proof note
and `SOURCE_NOTES.md`.

This package proves only dimension `10`. In the repository as a whole, the
remaining unresolved dimensions are `6,7,8,9`.
