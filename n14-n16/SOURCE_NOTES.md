# Source and dependency notes

The proof note relies on the following established ingredients.

1. S.-G. Hwang, *A note on a conjecture on permanents*, Linear Algebra and Its
   Applications 76 (1986), 31-44: the only possible positive global maximizer
   of the Dittert functional is the uniform matrix.
2. G.-S. Cheon and I. M. Wanless, *Some results towards the Dittert conjecture
   on permanents*, Linear Algebra and Its Applications 436 (2012), 791-801:
   partly decomposable matrices cannot maximize; a maximizer's complete zero
   set cannot be one proper rectangular block; Li's doubly-superstochastic
   criterion is recorded as Lemma 2.2.
3. H. Minc, *Minimum permanents of doubly stochastic matrices with prescribed
   zero entries*, Linear and Multilinear Algebra 15 (1984), 225-243,
   DOI `10.1080/03081088408817592`: Theorem 1's averaging of repeated-support
   rows and columns of a permanent minimizer, and the two-exceptional-zero face.
4. K. Pula, S.-Z. Song, and I. M. Wanless, *Minimum permanents on two faces of
   the polytope of doubly stochastic matrices*, Linear Algebra and Its
   Applications 434 (2011), 232-238: notation for the face and an explicit
   statement of how Minc's averaging theorem gives the block form.

Citation audit note: Section 2 of the Pula-Song-Wanless paper assumes both of
its block parameters exceed 2. Therefore its Theorem 2.1 is not invoked for
our `V_{m,2}` case. The proof note instead supplies a direct exact equalization
lemma for the two exceptional parameters. Only Minc's general averaging step
and the block notation are imported.

Recent context, not needed for the proof itself:

- Z. Pang, arXiv:2606.01531v1 (2026), gives a proof for dimensions `n >= 17`.
- B. Kafidov, arXiv:2607.19439v1 (2026), independently gives dimension 16 via
  the joint-deficit scaling lemma and the one-zero boundary estimate.

The present working note uses the stronger two-independent-zero face gap to
cover dimensions 14 and 15 as well as 16.
