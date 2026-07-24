# `n=6` proof search: exact reduction and current status

## Status

There is **not yet an `n=6` proof certificate** in this directory.  The files
here record a reproducible search, including several certificate cones that
were tested and found numerically insufficient.  Floating-point solver output
is never treated as a proof.

The exact verifier scaffold, `verify_primary.py`, accepts only a rounded
integer certificate whose reconstructed residual coefficients are all
strictly positive.  No search result produced so far passes it.

Two working fronts exist.  (1) The global 34-variable SOS/coefficient-domination
line (`verify_primary.py` and the `search_*` cone programs) is **structurally
capped**: the face minimizer is an interior, irrational, near-degenerate point
`x*` with `G6(x*) ~ 1.2e-8`, and a strictly-positive residual forces the SOS
part to reproduce `G6` at `x*` to within `~1e-8`, i.e. the optimal Gram sits on
the boundary of the PSD cone and cannot be rounded to a strictly-positive
integer certificate.  Adding rays or degree does not change this.  (2) The
**boundary-cut recursion** (the exact Li-scaling architecture that proves
`n=8`) is the more promising line; `diagnose_cut_scoreboard_n6.py` maps exactly
which cuts remain open (see the scoreboard section below).

## Exact face reduction

The same published structural reduction used in the `n=5` proof reduces a
boundary maximizer to the face

```text
x[0,0] = x[1,1] = 0.
```

After writing `X=A/6`, let the 34 remaining entries be `x_v`, and put
`S=sum_v x_v`.  On this face,

```text
F(x) = product(row sums) + product(column sums) - permanent(X)
```

is the square-free hypergraph polynomial whose edges meet every row or every
column.  It has 64,296 edges.  Since

```text
2 - 6!/6^6 = 643/324,
```

the required strict face inequality is

```text
G6(x) = (643/15116544) S^6 - F(x) > 0       (x >= 0, S=1).
```

The automorphism group of the two-zero face has order 2,304.  Its action has
the following exact orbit counts:

| Objects | Total | Orbits |
| --- | ---: | ---: |
| degree-2 monomials | 595 | 16 |
| degree-3 monomials | 7,140 | 65 |
| degree-4 monomials | 66,045 | 302 |
| degree-6 monomials | 3,262,623 | 5,605 |
| degree-7 monomials | 18,643,560 | 23,287 |

All coefficient maps in the search programs are constructed from these exact
integer orbit tables.

## Certificate families tested

### Degree-four multipliers times linear squares

The first full cone was

```text
sum_mu sum_gamma x^(gamma mu) (ell(gamma x)^T Q_mu ell(gamma x)),
Q_mu >= 0,
```

with one block for each of the 302 degree-four multiplier orbits.  The raw
model has 179,690 Gram coordinates.  Stabilizer averaging reduces this to
50,818 coordinates without changing the represented polynomial.

A coarse SCS run appeared to have positive margin, but exact PSD projection
destroyed it.  Tighter direct and indirect runs both converged near margin
`-9.34e-5`.  Thus the coarse positive value was a feasibility artifact, not a
certificate.  `search_degree4_invariant_sos_n6.py` is the preferred search
implementation for this cone.

### Adding degree-two multipliers times quadratic squares

There are 16 such 595-by-595 Gram blocks.  Full matrices are too large for a
naive SDP, so two reductions were tested:

- spectral column generation in
  `search_degree2_quadratic_column_n6.py`;
- obstruction-generated quadratic subspaces in
  `search_degree2_subspace_sos_n6.py`.

The subspace model had 27,178 coefficient coordinates, 13 PSD blocks of sizes
at most 107, and 864,057 nonzero coefficient-map entries.  Its reduced map was
cross-checked against the unreduced map on random symmetric matrices.  It
again converged to a negative margin close to `-9.4e-5`; this addition did not
remove the degree-six obstruction.

### Cubic sums of squares

Three increasingly strong cubic experiments were tried:

- transversal binomials `(x^A-x^B)^2`;
- negative-hyperedge obstruction eigenvectors;
- dual-priced eigenvectors of the full 7,140-by-7,140 cubic slack matrix.

The first two did not materially change the margin.  Full cubic pricing did
improve it, but successive master margins remained negative (approximately
`-1.20e-4`, `-1.15e-4`, `-1.11e-4`, `-1.10e-4`, `-1.09e-4`).  The relevant
programs are `search_cubic_binomial_n6.py`,
`search_cubic_obstruction_rays_n6.py`, and `search_cubic_column_n6.py`.

These are numerical search results, not impossibility theorems for the exact
cones.

The previous attempt suggested a more structured way to expose nontrivial
representation components: for one anchor in each of the three cell orbits,
form cubic orbit sums under its stabilizer and average their Gram squares over
the full group.  The exact feature counts are

| anchor cell orbit | cubic features | symmetric Gram coordinates |
| --- | ---: | ---: |
| exceptional off-diagonal | 117 | 6,903 |
| exceptional-to-ordinary | 440 | 97,020 |
| ordinary-to-ordinary | 272 | 37,128 |

`search_anchored_cubic_n6.py` implements these three blocks by PSD column
generation.  It compresses each full cubic dual slack matrix to dense blocks
of sizes 117, 440, and 272, then adds the most-negative rank-one rays.  This
avoids materializing the full 141,051-coordinate coefficient map and writes a
restartable checkpoint after every iteration.

This anchored cone is not larger than the full group-averaged cubic SOS cone
priced by `search_cubic_column_n6.py`: every anchored square is already a
group average of an ordinary cubic square.  Its advantage is computational,
because it adds directions from three targeted stabilizer-fixed subspaces in
parallel.  A seven-master diagnostic run improved the margin from
`-0.000120205628876` to `-0.000109996046095`; its checkpoint contains 48
priced rays.  The reduced slacks were still indefinite, so the run had not
exhausted the anchored cones, but the rate of improvement was already
slowing.  This is useful search progress, not evidence of a positive
certificate.

## Persistent degree-seven continuation

The degree-six searches share a pure-vertex cost obstruction: after PSD
projection, almost every residual orbit is negative and even the coefficients
of `x_v^6` are about `-9e-5`.  Multiplying that particular residual by a power
of `S` cannot repair its pure coefficients.

The next hierarchy level is therefore to search for a new identity for
`S*G6` rather than merely lift the old residual.  A natural cone is

```text
sum_mu sum_gamma x^(gamma mu)
    m2(gamma x)^T Q_mu m2(gamma x),
Q_mu >= 0,
```

where `mu` runs over the 65 degree-three multiplier orbits.  The lifted target
has 23,287 coefficient orbits.

`search_degree7_column_n6.py` now keeps one HiGHS master model alive through
SciPy's bundled HiGHS interface.  It scales columns, reuses simplex bases or
uses IPM with crossover, prunes only old zero-weight rays, rejects duplicate
eigendirections, reconstructs full columns before accepting a result, and
writes restartable checkpoints.

The persistent search was run through master iteration 8.  The latest solved
model had 1,580 active quadratic rays and margin

```text
-0.000654998425835
```

before adding the next 220 priced rays.  The checkpoint contains 1,691 rays.
This improves on the initial margin `-0.000660763278985`, but is far too small
to indicate a path to zero.  Spectral violations also remain substantial.
Thus this degree-seven cone has not been proved insufficient, but merely
extending the same bundle is not the best next use of computation.

## Li-scaling diagnostic

The proof architecture used for dimensions 8--10 was specialized to `n=6`
in `explore_li_scaling_n6.py`.  One part is especially promising and exact.
On the doubly stochastic face with two independent zeroes, the permanent is
the univariate polynomial

```text
18 x^2 - 336 x^3 + 2520 x^4 - 8832 x^5 + 12288 x^6,
1/8 <= x <= 1/4.
```

An exact Sturm calculation proves that this polynomial is greater than
`2/125`, while

```text
2/125 - gamma_6 = 23/40500 > 0.
```

The later dimension-8 inequalities do not survive unchanged:

- the scalar marginal inequality has a risk interval roughly from `0.006`
  to `0.056`;
- the large-column support bounds fail for distinguished columns with one or
  two zero entries;
- every crude proper-cut `eta` comparison is negative.

This rules out a direct substitution of `n=6` into the dimension-8 proof,
but the exact two-zero permanent gap remains useful input for a stronger
maximality argument.

### First-variation refinement of Li scaling

There is a valid strengthening that keeps the first variation of the
permanent.  Write

```text
A = lambda B + E,       E >= 0.
```

For a tight Li cut of denominator `k`, tightness forces `E` to vanish on the
tight block.  Hence `p(t)=per(lambda B+tE)` has degree at most `n-k`, and its
nonnegative coefficients give

```text
P - lambda^n per(B) >= p'(1)/(n-k).
```

Support stationarity and `sum(E)=n(1-lambda)` then give

```text
(n lambda-k) P >= (n-k) lambda^n per(B) - lambda(E_r+E_c),

E_r = R (sum_i 1/r_i-n),       E_c = C (sum_j 1/c_j-n).
```

Since `lambda` is no larger than any line sum,

```text
lambda E_r <= -2R log R <= 2(1-R),
lambda E_c <= -2C log C <= 2(1-C).
```

Together with the shared deficit this proves

```text
(n lambda-k) P >= (n-k) lambda^n per(B) - 2 delta.
```

For the `n=6` marginal cut (`lambda=1-a`, `k=1`), combining this with
`delta>=D(a)` and `per(B)>=2/125` yields a new exact safe interval
`0<=a<=1/125`.  A Sturm check in `explore_li_scaling_n6.py` proves positivity
there; the first numerical root is approximately `0.008138350996`.  Used
together with the basic Li estimate, this narrows the current rational risk
enclosure from `[1/200,3/50]` to `[1/125,3/50]`.  It does not improve the
right endpoint, so it is useful but not by itself decisive.

The first-variation inequality contains more information than the basic Li
estimate, but after replacing its error by `2 delta` neither resulting scalar
upper bound dominates the other for every `lambda`.  Both bounds must be
retained.

## Exact boundary-cut scoreboard (`diagnose_cut_scoreboard_n6.py`)

The audited `n=8` proof excludes every boundary maximizer by classifying the
maximal Li scaling into whole, marginal, and proper tight cuts and killing each
with an exact univariate inequality.  `diagnose_cut_scoreboard_n6.py` runs that
same classification for `n=6` in exact rational arithmetic and reports the
margin of every cut.  Current state:

| Cut | Status | Exact margin |
| --- | --- | --- |
| two-zero permanent gap (symmetric face) | closed | `L-gamma = 23/40500` |
| marginal cut outside risk `[1/125, 3/50]` | closed | Sturm, 0 roots |
| subset-deficit constants `c_d` | closed | Bernstein `> 0` |
| support-cut validity band `q>1` | **open** | valid only on `[31/2000, 3/50]`; `[1/125, 31/2000]` UNCOVERED |
| support cut `z=1` on `[a_q, 3/50]` | **open** | Bernstein-min `-1.8e-4` |
| support cut `z=2` on `[a_q, 3/50]` | **open** | Bernstein-min `-5.3e-4` |
| support cuts `z=3,4` on `[a_q, 3/50]` | closed | Bernstein `> 0` |
| proper cut `(1,1,4)` | **closed** | via `per(B)>=2/125`; sharp-`K`-min `+1.17e-5` |
| proper cuts `(u,v,k)`, other 9 | **open** | sharp-`K`-min `-3.7e-4` .. `-3.8e-3` |

The support-cut caveat is a correctness fix flagged in review: the
large-column derivation requires `q(a) > 1` (that is what caps every
non-distinguished row sum below one and makes `U_z` a valid product upper
bound).  For `n=6`, `q(1/125)-1 = -0.00745 < 0`; `q` first exceeds `1` only at
`a_q ~ 0.01529`.  So the support argument is valid only on `[a_q, 3/50]`, and
the band `[1/125, a_q]` of the marginal risk interval is **uncovered by any
current argument** — a real gap in the marginal case beyond the `z=1,2`
failures.  (`z=1,2` fail on the valid domain too; `z=3,4` pass there.)

Two `n=8` bounds were tested on the proper cuts and both fail at `n=6`:

- the crude `eta = nu - gamma - n^3 nu^2 / (4 k^2 (1-gamma))` (which `verify_n8`
  only needs for `k>=2`); and
- the sharp `K(s) = nu (1 - s/k)^n - gamma + c_{u,v} s^2`, generalized from the
  `n=8` `k=1` special cut using the cut identity `lambda >= 1 - s/k` (so
  `P >= nu (1 - s/k)^n`, which is *stronger* for `k>=2`).

The sharp bound is a two-sided squeeze `c_{u,v} s^2 <= delta <= gamma - nu
(1-s/k)^n` that stays consistent by a small margin near small-to-moderate `s`.
The marginal-style first-variation refinement (upper bound `delta` via
`(n lambda - k) P >= (n-k) lambda^n nu - 2 delta`) gives **no** improvement: a
numeric `(s, lambda)` scan shows the worst point sits at `lambda` near `1`
(for `k>=2`) or `lambda = 1 - s` (for `k=1`), where that refinement is vacuous.
Even the theoretically optimal near-zero subset constants
`c_d = (1/d + 1/(n-d))/2` cannot cover the worst `k=1` cuts (they would need
`c_{u,v}` roughly doubled).

### The shared improvement lever (measured)

Both open families under-use one fact: the dominated doubly stochastic `B`
always inherits the two independent zeros of `A`, yet the open permanent bounds
`M_z` (support) and `nu` (proper) are computed from the cut's own zero block
alone.  A **combined minimum-permanent bound** — cut zero block *plus* the
inherited zeros — could raise the open margins.  But it must respect
**absorption** (correctness fix flagged in review): a `u x v` block can already
contain one of the inherited zeros (both, when `u>=2` and `v>=2`), and a single
zero column can contain at most one (the two need distinct columns).  So only
the zeros that cannot be absorbed are *guaranteed* to add new constraints;
adding two free zeros unconditionally minimizes over a strictly smaller face and
overstates the bound.  `combined_permanent_derisk_n6.py` now computes the
guaranteed bound over the worst placement.  Result:

| Open cut | `per(B)` needed | guaranteed | verdict |
| --- | ---: | ---: | --- |
| proper `(1,2,3)`, `(2,1,3)` (absorb 1) | `0.01662` | `0.01645` | no |
| proper `(1,3,2)`, `(3,1,2)` (absorb 1) | `0.01836` | `0.01728` | no |
| proper `(1,4,1)`, `(4,1,1)` (absorb 1) | `0.02901` | `0.01922` | no |
| proper `(2,2,2)` (absorb 2) | `0.01872` | `0.01758` (`nu`) | no |
| proper `(2,3,1)`, `(3,2,1)` (absorb 2) | `0.03337` | `0.02083` (`nu`) | no |
| support `z=1` (absorb 1) | `0.01652` | `0.01604` | no |
| support `z=2` (absorb 1) | `0.01660` | `0.01645` | no |

**After correct absorption handling, no open cut closes via the combined
bound.**  The earlier "closures" of `(1,2,3),(2,1,3)` and support `z=2` were
artifacts of adding two guaranteed-outside zeros; with only the one
non-absorbed zero the guaranteed bound drops below threshold.

**The one rigorous closure** stands independently: `B` lies on the two-zero
face, so `per(B) >= max(nu, 2/125)`, and for `(1,1,4)` the block barycenter
`nu = 0.01573` is *below* `2/125 = 0.016`.  With `per(B) >= 2/125` the sharp
bound `K(s)` has exact Bernstein-min `+1.17e-5 > 0` on `[0, s_max]` (checked in
`diagnose_cut_scoreboard_n6.py`, which now applies `max(nu, 2/125)`).  This uses
only the two inherited zeros, so absorption does not affect it.  It drops the
open proper cuts from ten to nine.  A 3-independent-zero reduction is
**unnecessary** — `(1,1,4)` is the only single-cell-block cut and it is already
closed by the two-zero bound; every other open cut has `nu > 2/125` and gains
nothing from it.

### Remaining frontier for the hard cuts

The `k=1,2` proper cuts need a different handle than a better `per(B)`:

- **Fold the `k=1` proper cuts into the marginal large-line argument.**  A
  `k=1` proper cut has `u+v = n-1`, the maximal proper zero block, so `B` is one
  step from decomposable and `A` carries a distinguished large line.  The
  marginal section's large-column support bound (`U_z`, `R_{0,z}`, `G_z`) is far
  sharper than the crude `K(s)` and is the natural tool; the `k=1` cuts should
  be routed through it rather than the proper-cut `K`.
- **Strengthen the deficit lower bound** beyond `delta >= c_{u,v} s^2` for the
  `k=2` cuts (a linear-in-`s` deficit term, or a cut-specific Pinsker bound that
  keeps more than the quadratic).

Each closed cut still needs an *exact* minimum-permanent reduction (in the style
of `two_zero_permanent_polynomial`) before it counts; the de-risk tool only
tells us which reductions are worth deriving.  Even so, this route is far more
tractable than the 34-variable global SOS: each cut is one low-dimensional
permanent minimization plus a Sturm/Bernstein check, and the open set is now
seven cuts, all of one structural type.

## Numerical geometry of the hard face

`optimize_face_n6.py` evaluates the permanent and its gradient directly and
uses a scaled constrained objective.  Twenty random starts, plus symmetric
starts, all converged to the same relative-interior stationary point.  It is
doubly stochastic after scaling by 6 and has the three cell-orbit values

```text
special cross entries       0.038267539249...
top/lower cross entries     0.032099781855...
lower 4-by-4 entries        0.025616775739...
```

At this point

```text
F = 0.0000425238864527176...
target = 0.0000425361775813308...
gap = 0.0000000122911286131...
```

The Hessian restricted to the simplex tangent space is numerically negative
definite; its largest eigenvalue is about `-1.42495e-5`.  Imposing one
additional zero of each of the three variable-orbit types again led
numerically to doubly stochastic stationary points, with target gaps between
`1.78e-8` and `2.69e-8`.

These calculations strongly suggest that the global face maximizer is the
doubly stochastic permanent minimizer, but they do not prove it.
`test_symmetrization_n6.py` also records an important negative result: full
group averaging does not always increase `F`, even in a fairly high-value
region.  Any valid reduction to the symmetric point must therefore use the
first- or second-order conditions of a maximizer, not averaging alone.

## Best next proof search

The most focused next route is a higher KKT/Jacobian certificate for the
two-zero face.  At a constrained maximizer of the homogeneous degree-six form,

```text
x_v * (S * partial_v F - 6 F) = 0
```

for every allowed coordinate.  Also

```text
6 F - S * partial_v F >= 0
```

at a maximizer, with equality on its positive support.
`search_kkt_seed_n6.py` tested the two independent constant symmetry sums of
the complementarity equations at degree seven; they changed the projected
seed margin only from `-0.0006607633` to `-0.0006606883`.
`search_kkt_inequality_n6.py` tested the three degree-six orbit sums of the
derivative inequalities; the optimizer assigned all three zero weight.  The
coefficient map in the latter program was independently checked against a
direct gradient evaluation to `2.8e-19`.

Thus useful KKT information will require polynomial multipliers, equivalently
the next Jacobian relaxation, rather than constant orbit sums.  This still
targets only stationary supports and avoids forcing a very weak
coefficient-positive certificate over the entire orthant.  The exact
univariate permanent gap above can then certify the symmetric stationary
case.  A second computational option is to combine the full cubic-SOS and
degree-two/quadratic-SOS cones in one persistent master; those cones have so
far only been priced separately.

If either route produces a positive numerical margin, the exact-certificate
steps are already scaffolded:

1. round explicit factors on a dyadic grid;
2. reconstruct every coefficient with Python integers;
3. require a strictly positive exact residual;
4. add an independent verifier and only then write the proof note.

## Reproduction environment

Create a disposable environment and install the pinned discovery packages:

```bash
python3 -m venv /tmp/dittert-n6-venv
/tmp/dittert-n6-venv/bin/pip install -r n6/requirements-search.txt
```

For example, the stabilizer-reduced degree-six search is

```bash
/tmp/dittert-n6-venv/bin/python n6/search_degree4_invariant_sos_n6.py \
  --eps 1e-6 --max-iters 3000 --output /tmp/n6_degree4.npz
```

and the degree-seven continuation is

```bash
/tmp/dittert-n6-venv/bin/python n6/search_degree7_column_n6.py \
  /tmp/n6_degree4.npz --solver ipm \
  --checkpoint /tmp/n6_degree7.checkpoint.npz \
  --output /tmp/n6_degree7.npz
```

Restart it with

```bash
/tmp/dittert-n6-venv/bin/python n6/search_degree7_column_n6.py \
  /tmp/n6_degree4.npz --solver ipm \
  --resume /tmp/n6_degree7.checkpoint.npz \
  --checkpoint /tmp/n6_degree7.checkpoint.npz \
  --output /tmp/n6_degree7.npz
```

Run or resume the anchored-cubic search from a degree-four archive with

```bash
/tmp/dittert-n6-venv/bin/python n6/search_anchored_cubic_n6.py \
  /tmp/n6_degree4.npz --iterations 10 --rays-per-block 3 \
  --output /tmp/n6_anchored_cubic.npz

/tmp/dittert-n6-venv/bin/python n6/search_anchored_cubic_n6.py \
  /tmp/n6_degree4.npz --resume /tmp/n6_anchored_cubic.npz \
  --iterations 10 --rays-per-block 3 \
  --output /tmp/n6_anchored_cubic.npz
```

Run the structural diagnostics with

```bash
/tmp/dittert-n6-venv/bin/python n6/explore_li_scaling_n6.py
/tmp/dittert-n6-venv/bin/python n6/optimize_face_n6.py --random-starts 20
/tmp/dittert-n6-venv/bin/python n6/optimize_subfaces_n6.py --starts 20
/tmp/dittert-n6-venv/bin/python n6/search_kkt_seed_n6.py /tmp/n6_degree4.npz
/tmp/dittert-n6-venv/bin/python n6/search_kkt_inequality_n6.py /tmp/n6_degree4.npz
```

Neither command is a proof check.  A proof exists only when an exact archive
passes `verify_primary.py` and prints `CERTIFIED`.
