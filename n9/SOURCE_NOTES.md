# Source and dependency notes

## Published mathematical inputs

The proof note uses the same established structural results as the dimension
10 package:

1. Hwang's positive-maximizer theorem for the Dittert functional.
2. Cheon and Wanless's exclusions of partly decomposable maximizers and of a
   complete zero set consisting of one proper rectangle.
3. Li's cut characterization of doubly superstochastic matrices.
4. Hwang's permanent minimum at the barycenter of a staircase face.
5. Minc's averaging theorem for the doubly stochastic face with two
   independent prescribed zeroes.
6. The van der Waerden permanent theorem and equality case.

Full citations are in the proof note. The support-stationarity equation and
the new large-column support-count argument are proved directly there.

## Computational scope

The primary and independent standard-library programs check the two-zero
equalization and permanent bound, the complete marginal intervals, all seven
large-column support cases, the staircase barycenter counts, and all 28 proper
tight cuts with exact rational arithmetic. The independent program imports no
primary-verifier code and reconstructs the key polynomials from exact values.

No floating-point number is used in a proof decision. The printed decimal
approximations are diagnostic only.

## Status

These materials are an exact computer-assisted working proof, not a claim of
completed journal peer review. With this package, the repository covers
dimensions `4,5,9,10,11,12,13,14,15,16`; dimensions `6,7,8` remain open
within this project.
