# Source, dependency, and provenance notes

## Published mathematical inputs

The proof note relies on these established results:

1. S.-G. Hwang, *A note on a conjecture on permanents*, Linear Algebra and Its
   Applications 76 (1986), 31–44: a positive global maximizer of the Dittert
   functional is the uniform matrix.
2. G.-S. Cheon and I. M. Wanless, *Some results towards the Dittert conjecture
   on permanents*, Linear Algebra and Its Applications 436 (2012), 791–801:
   partly decomposable matrices cannot maximize, a maximizer's complete zero
   set cannot be one proper rectangle, and Li's doubly-superstochastic
   criterion is recorded as Lemma 2.2.
3. C.-K. Li, *On certain convex matrix sets*, Discrete Mathematics 79 (1990),
   323–326: the cut characterization of doubly superstochastic matrices.
4. S.-G. Hwang, *Minimum permanent on faces of staircase type of the polytope
   of doubly stochastic matrices*, Linear and Multilinear Algebra 18 (1985),
   271–306: the staircase-face barycenter minimizes the permanent.
5. H. Minc, *Minimum permanents of doubly stochastic matrices with prescribed
   zero entries*, Linear and Multilinear Algebra 15 (1984), 225–243: the
   averaging theorem giving the repeated-support block form. K. Pula,
   S.-Z. Song, and I. M. Wanless, Linear Algebra and Its Applications 434
   (2011), 232–238, provide the face notation and an explicit use of this
   averaging reduction.
6. The van der Waerden permanent theorem and its equality case.

The support-stationarity equation used for the new marginal argument is
proved directly from first variations and Euler's homogeneous identity in the
proof note. It is not treated as an external theorem.

As in the `n14-n16` package, only Minc's averaging step is imported for the
two-exceptional-zero face. The equalization of the two exceptional parameters
is supplied and checked explicitly.

## Supplied artifact provenance

The initial dimension-specific verifier and its captured output were supplied
on 23 July 2026 at these SHA-256 hashes:

```text
4888906c48c37c71d3e11f117a34ae6966c3b9714408c400f2d6053e5999c75e  verify_dittert_n10.py
26fe91a3896e16280f22977c3485fedb949c326d177aafa85b4f4f44e78a96d1  verify_dittert_n10.log
```

The packaged primary verifier preserves the supplied mathematics while adding
explicit degree and denominator checks, removing its sole executable `assert`
statement, and expanding the exact diagnostic output. The independent
standard-library verifier, proof note, tests, documentation, PDF, logs, and
manifest were created during repository packaging.

## Status

These materials are an exact computer-assisted working proof, not a claim of
completed journal peer review. The current repository coverage is dimensions
`4,5,9,10,11,12,13,14,15,16`; dimensions `6,7,8` remain open within this
project.
