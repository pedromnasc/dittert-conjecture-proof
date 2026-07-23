# Exact working proofs for Dittert's conjecture in dimensions 4, 14, 15, and 16

> **Status:** These are exact, reproducible working proofs. They have not yet
> undergone formal peer review. This repository covers only `n=4` and
> `n=14,15,16`; it does not by itself settle the intervening dimensions.

For a nonnegative `n x n` matrix `A` whose entries sum to `n`, define

$$
\Phi(A)=\prod_{i=1}^n r_i+\prod_{j=1}^n c_j-\operatorname{per}(A),
$$

where the $r_i$ and $c_j$ are the row and column sums. Dittert's conjecture
asserts that

$$
\Phi(A)\leq 2-\frac{n!}{n^n},
$$

with equality only at the matrix whose entries are all `1/n`.

This repository contains proof notes, exact certificates, verification code,
captured audit output, and checksums for the cases listed above.

## Proofs and audit material

| Dimensions | Proof note | Audit and reproduction notes |
| --- | --- | --- |
| `n=4` | [PDF](n4/dittert_n4_exact_proof.pdf) · [LaTeX](n4/dittert_n4_exact_proof.tex) | [Independent audit report](n4/AUDIT_REPORT.md) · [bundle README](n4/README.md) |
| `n=14,15,16` | [PDF](n14-n16/dittert_n14_n16_proof_2026-07-23.pdf) · [LaTeX](n14-n16/dittert_n14_n16_proof_2026-07-23.tex) | [reproduction guide](n14-n16/README.md) · [source and citation notes](n14-n16/SOURCE_NOTES.md) |

## Reproduce the exact checks

Python 3.10 or newer is sufficient for the standard-library verifiers. NumPy
is used by two additional `n=4` checks, and SymPy is used by the independent
Sturm audit for `n=14,15,16`.

```bash
git clone https://github.com/pedromnasc/dittert-conjecture-proof.git
cd dittert-conjecture-proof

# Optional dependencies for every available verifier
python3 -m pip install numpy sympy

# n=4: check file integrity and run three independent certificate readers
cd n4
sha256sum -c SHA256SUMS
python3 verify_primary.py dittert_n4_exact_certificate.npz
python3 verify_literal_square_numpy.py dittert_n4_exact_certificate.npz
python3 verify_literal_square_stdlib.py dittert_n4_exact_certificate.npz

# n=14,15,16: check file integrity and run both exact audits
cd ../n14-n16
sha256sum -c MANIFEST.sha256
python3 verify_dittert_n14_n16.py
python3 audit_dittert_n14_n16_sympy.py
```

The expected final status is `CERTIFIED` for each `n=4` verifier,
`ALL CASES CERTIFIED` for the primary `n=14,15,16` verifier, and
`INDEPENDENT AUDIT CERTIFIED` for the SymPy audit. No floating-point value is
used in a correctness decision.

## Verification scope

The programs verify the finite algebraic parts of the arguments: certificate
identities, polynomial calculations, exact rational bounds, and final strict
inequalities. They do not reprove the published structural theorems used in
the reductions. Those dependencies and citations are identified in the proof
notes, the [`n=4` audit report](n4/AUDIT_REPORT.md), and the
[`n=14,15,16` source notes](n14-n16/SOURCE_NOTES.md).

Independent scrutiny of both the mathematical reductions and the software is
welcome. Please use [GitHub Issues](https://github.com/pedromnasc/dittert-conjecture-proof/issues)
for errors, questions, or independently reproduced results.

## Bundle provenance

The repository was assembled on 23 July 2026 from the two original bundles
below. Their extracted contents are preserved under `n4/` and `n14-n16/`, and
their internal checksum manifests verify unchanged.

```text
6b42ad7ebe73fc1686f7442c093ef872c1b25f30dfab1510da27be6000260e01  dittert_n4_audited_bundle_2026-07-23.zip
3536bf1bcb6ab1b141b1bea74c7ad9ab41b307b3dfa6beb13f883e3a5739149a  dittert_n14_n16_exact_proof_bundle_2026-07-23.zip
```
