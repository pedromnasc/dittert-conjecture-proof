# Independent audit of the exact `n=4` Dittert certificate

**Audit date:** 23 July 2026  
**Scope:** the Dittert conjecture in dimension `n=4`, not the conjecture in every dimension.

## Verdict

The supplied certificate gives a valid exact computer-assisted proof of the following statement, conditional only on the cited published structural results of Hwang and Cheon-Wanless:

> For every nonnegative `4 x 4` matrix `A` whose entries sum to `4`,
> 
> `phi(A) <= 61/32`,
> 
> with equality only at the matrix whose sixteen entries are all `1/4`.

No floating-point result is used in a proof decision. Three verifier implementations were rerun successfully. The third implementation uses only the Python standard library and parses the NPZ/NPY encoding itself.

## Proof dependency graph

1. Compactness gives a global maximizer.
2. Hwang's positive-support theorem says that a positive global maximizer must be the uniform matrix.
3. Cheon-Wanless exclude a global maximizer whose complete zero set is a single rectangular block; they also exclude partly decomposable maximizers.
4. If a boundary maximizer had no pair of zeroes in different rows and different columns, all its zeroes would lie in one row or one column. Such a zero set is either a full zero row/column or a single rectangular block. Therefore a boundary maximizer has two independent zeroes, which can be moved to `(1,1)` and `(2,2)` by row and column permutations.
5. The exact certificate proves strict inequality on the entire face `a_11=a_22=0`.
6. Thus no boundary maximizer exists, so the maximizer is positive and Hwang's theorem identifies it with the uniform matrix.

## Face polynomial and scaling

Put `A=4X`. On the face `x_11=x_22=0`, fourteen variables remain and their sum is `S=1`. Direct expansion gives

`phi(A) = 256 F(x)`,

where `F` is the sum of the 274 square-free quartic monomials whose four cells meet all rows or all columns. Hence the desired strict inequality is

`G(x) = (61/8192) S^4 - F(x) > 0`.

The certificate proves the exact identity

`2^60 G = SOS_0 + weighted_SOS + residual`,

where every square term has an integer numerator, every monomial weight is nonnegative on the nonnegative orthant, and every one of the 2,380 degree-four residual coefficients is a strictly positive integer.

The smallest residual coefficient is

`72,694,203,872 / 2^60 = 6.3052171012101255e-8`.

Because a nonzero nonnegative vector has some positive fourth-power monomial, the strictly positive residual coefficients imply `G(x)>0` on the whole nonzero nonnegative cone, not merely at sampled points.

## Verifier results

All three programs reported:

- certificate SHA-256: `d76533bd1c5566ea8d96aa3b58a0b6a8bf3310eb438776ebd99a6bd88d5a11f6`
- hyperedges in `F`: `274`
- quartic monomials checked: `2380`
- minimum residual numerator: `72694203872`
- maximum residual numerator: `11312950144595`
- final status: `CERTIFIED`

The primary verifier builds Gram matrices and checks the symmetry and hypergraph data. The second verifier expands every saved factor as literal squares and constructs the target from row products, column products, and permanent terms. The third verifier repeats the literal-square check without NumPy and includes an independent NPZ/NPY parser.

## What this does not prove

This certificate does not settle dimensions `5` through `16`. A June 2026 preprint claims the conjecture for every `n >= 17`; its own status statement leaves `4 <= n <= 16` open. If the present `n=4` certificate is accepted, the unresolved finite range becomes `5 <= n <= 16`.

The result has not yet undergone independent peer review or journal publication. The certificate is easy to check exactly, but the literature reduction and the software should still be reviewed by other mathematicians before the result is cited as established.

## Reproduction

Run without `python -O`:

```bash
python verify_primary.py dittert_n4_exact_certificate.npz
python verify_literal_square_numpy.py dittert_n4_exact_certificate.npz
python verify_literal_square_stdlib.py dittert_n4_exact_certificate.npz
```

Each program must terminate with `CERTIFIED`.

## References used by the reduction

- S.-G. Hwang, “A note on a conjecture on permanents,” *Linear Algebra and its Applications* 76 (1986), 31-44.
- G.-S. Cheon and I. M. Wanless, “Some results towards the Dittert conjecture on permanents,” *Linear Algebra and its Applications* 436 (2012), 791-801.
- Z. Pang, “Proof of Dittert's conjecture for dimensions n >= 17,” arXiv:2606.01531 (2026), preprint.
