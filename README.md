# Exact working proofs for Dittert's conjecture in dimensions 4, 5, and 11–16

[![Exact verification](https://github.com/pedromnasc/dittert-conjecture-proof/actions/workflows/verify.yml/badge.svg)](https://github.com/pedromnasc/dittert-conjecture-proof/actions/workflows/verify.yml)

> **Status:** These are exact, reproducible working proofs. They have not yet
> undergone formal peer review. This repository covers only `n=4,5` and
> `n=11,12,13,14,15,16`; it does not by itself settle dimensions `6` through
> `10`.

For a nonnegative `n x n` matrix `A` whose entries sum to `n`, define

$$
\Phi(A)=\prod_{i=1}^n r_i+\prod_{j=1}^n c_j-\mathrm{per}(A),
$$

where the $r_i$ and $c_j$ are the row and column sums. Dittert's conjecture
asserts that

$$
\Phi(A)\leq 2-\frac{n!}{n^n},
$$

with equality only at the matrix whose entries are all `1/n`.

This repository contains proof notes, exact certificates, verification code,
captured audit output, and checksums for the cases listed above.

Author: **Pedro Paulo Marques do Nascimento** (`pedromnasc`).

## Proofs and audit material

| Dimensions | Proof note | Audit and reproduction notes |
| --- | --- | --- |
| `n=4` | [PDF](n4/dittert_n4_exact_proof.pdf) · [LaTeX](n4/dittert_n4_exact_proof.tex) | [Independent audit report](n4/AUDIT_REPORT.md) · [bundle README](n4/README.md) |
| `n=5` | [PDF](n5/dittert_n5_exact_proof.pdf) · [LaTeX](n5/dittert_n5_exact_proof.tex) | [exact verifier and reproduction guide](n5/README.md) |
| `n=11,12,13` | [PDF](n11-n13/dittert_n11_n13_exact_proof_2026-07-23.pdf) · [LaTeX](n11-n13/dittert_n11_n13_exact_proof_2026-07-23.tex) | [exact verifier and independent-audit guide](n11-n13/README.md) |
| `n=14,15,16` | [PDF](n14-n16/dittert_n14_n16_proof_2026-07-23.pdf) · [LaTeX](n14-n16/dittert_n14_n16_proof_2026-07-23.tex) | [reproduction guide](n14-n16/README.md) · [source and citation notes](n14-n16/SOURCE_NOTES.md) |

## Reproduce the exact checks

Python 3.10 or newer is sufficient for the standard-library verifiers. NumPy
is used by two additional `n=4` checks, and SymPy is used by the independent
Sturm audits for `n=11` through `16`. Exact dependency versions used in
automated verification are recorded in `requirements-audit.txt`.

```bash
git clone https://github.com/pedromnasc/dittert-conjecture-proof.git
cd dittert-conjecture-proof

# Dependencies for every available verifier
python3 -m pip install -r requirements-audit.txt

# n=4: check file integrity and run three independent certificate readers
cd n4
sha256sum -c SHA256SUMS
python3 verify_primary.py dittert_n4_exact_certificate.npz
python3 verify_literal_square_numpy.py dittert_n4_exact_certificate.npz
python3 verify_literal_square_stdlib.py dittert_n4_exact_certificate.npz

# n=5: check file integrity and the exact quintic certificate
cd ../n5
sha256sum -c SHA256SUMS
python3 verify_primary.py dittert_n5_exact_certificate.npz

# n=11,12,13: check file integrity and run both exact audits
cd ../n11-n13
sha256sum -c CHECKSUMS.sha256
python3 -I verify_dittert_n11_n13.py
python3 -I audit_dittert_n11_n13_sympy.py

# n=14,15,16: check file integrity and run both exact audits
cd ../n14-n16
sha256sum -c MANIFEST.sha256
python3 verify_dittert_n14_n16.py
python3 audit_dittert_n14_n16_sympy.py
```

Run the `n=4` and `n=5` programs without Python's `-O` option. Each assertion-based
verifier rejects optimized mode explicitly so that proof checks cannot be
silently disabled. The same commands run automatically on every push and pull
request through GitHub Actions.

The expected final status is `CERTIFIED` for each `n=4` verifier and the
`n=5` verifier, `ALL CASES CERTIFIED` for each multi-dimension primary
verifier, and `INDEPENDENT AUDIT CERTIFIED` for each SymPy audit. No
floating-point value is used in a correctness decision.

## Verification scope

The programs verify the finite algebraic parts of the arguments: certificate
identities, polynomial calculations, exact rational bounds, and final strict
inequalities. They do not reprove the published structural theorems used in
the reductions. Those dependencies and citations are identified in the proof
notes, the [`n=4` audit report](n4/AUDIT_REPORT.md), the
[`n=5` proof note](n5/dittert_n5_exact_proof.tex), the
[`n=11,12,13` proof note](n11-n13/dittert_n11_n13_exact_proof_2026-07-23.tex), and the
[`n=14,15,16` source notes](n14-n16/SOURCE_NOTES.md).

Independent scrutiny of both the mathematical reductions and the software is
welcome. Please use [GitHub Issues](https://github.com/pedromnasc/dittert-conjecture-proof/issues)
for errors, questions, or independently reproduced results.

## Citation

Repository citation metadata is provided in [`CITATION.cff`](CITATION.cff).
Please retain the working-proof and peer-review status when discussing these
results.

## License

The papers, LaTeX sources, documentation, certificates, audit material, and
other non-software research content are licensed under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt). The verification code and supporting
software infrastructure are licensed under the [MIT License](LICENSES/MIT.txt).
See [`LICENSE.md`](LICENSE.md) for the precise scope, including archived bundle
contents.

## Bundle provenance

The repository was assembled on 23 July 2026 from the three original bundles
below; the `n=5` certificate was subsequently developed in this repository on
the same date. The untouched archives are retained under `original-bundles/`;
the reviewed working directories have their own current checksum manifests.

- [`n=4` original bundle](original-bundles/dittert_n4_audited_bundle_2026-07-23.zip)
- [`n=11,12,13` original bundle](original-bundles/dittert_n11_n13_exact_proof_bundle_2026-07-23.zip)
- [`n=14,15,16` original bundle](original-bundles/dittert_n14_n16_exact_proof_bundle_2026-07-23.zip)

```text
6b42ad7ebe73fc1686f7442c093ef872c1b25f30dfab1510da27be6000260e01  original-bundles/dittert_n4_audited_bundle_2026-07-23.zip
bfe6f67cd2eb5a6428bd3476236332dfb83ece6b6950db7ef238e96c7a1c0270  original-bundles/dittert_n11_n13_exact_proof_bundle_2026-07-23.zip
3536bf1bcb6ab1b141b1bea74c7ad9ab41b307b3dfa6beb13f883e3a5739149a  original-bundles/dittert_n14_n16_exact_proof_bundle_2026-07-23.zip
```
