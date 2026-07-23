# Exact working proof for Dittert's conjecture in dimensions 14, 15, and 16

Date: 23 July 2026

## Statement

For each `n` in `{14,15,16}` and every nonnegative `n x n` matrix `A` whose
entries sum to `n`, the bundle proves

    Phi(A) <= 2 - n!/n^n,

with equality only at the matrix whose entries are all `1/n`.

This is an exact, reproducible working proof. It has not yet undergone formal
peer review.

## Proof architecture

The proof has four parts.

1. Published boundary results imply that a boundary maximizer must contain two
   zero entries in distinct rows and columns.
2. A joint-deficit estimate scales any putative maximizer to a doubly
   superstochastic matrix by the factor

       t = sqrt(n*delta/(1-delta)),

   where `delta = n!/n^n - per(A)`.
3. Minc's averaging theorem reduces the minimum permanent on the corresponding
   two-zero doubly stochastic face to a two-parameter block matrix. A direct
   exact lemma in the note proves that its two exceptional parameters must be
   equal at a minimum, leaving one variable.
4. Exact rational lower bounds for the three resulting permanent polynomials
   make the final boundary contradiction strictly positive.

The exact margins in the last comparison are approximately

- `n=14`: `1.880968899287138e-9`
- `n=15`: `7.119791408990794e-9`
- `n=16`: `3.444829237860010e-9`

## Files

- `dittert_n14_n16_proof_2026-07-23.pdf` - proof note.
- `dittert_n14_n16_proof_2026-07-23.tex` - LaTeX source.
- `verify_dittert_n14_n16.py` - standard-library-only verifier using exact
  `fractions.Fraction` arithmetic and rational interval Horner evaluation.
- `audit_dittert_n14_n16_sympy.py` - independent exact SymPy audit using
  exact Ryser evaluations at `n+1` rational points and Sturm root counts on the
  full admissible intervals.
- `*.log` - captured successful runs.
- `SOURCE_NOTES.md` - dependency and citation audit.
- `MANIFEST.sha256` - hashes of the bundle contents, excluding the manifest.

## Run the exact checks

Python 3.10 or newer is sufficient for the primary verifier:

```bash
python verify_dittert_n14_n16.py
```

The independent audit additionally requires SymPy:

```bash
python audit_dittert_n14_n16_sympy.py
```

The expected final lines are respectively:

```text
ALL CASES CERTIFIED
INDEPENDENT AUDIT CERTIFIED
```

Neither program uses a floating-point number in a correctness decision.
Decimal output is for readability only.

## Scope of the programs

The programs verify the finite algebraic part of the proof: the block
permanent, equalization derivative, polynomial identities and factorizations,
critical-point isolation, permanent lower bounds, dilation precondition, and
final rational inequalities. They do not reprove the cited structural results
of Hwang, Cheon-Wanless, Li, Minc, or the van der Waerden theorem.
