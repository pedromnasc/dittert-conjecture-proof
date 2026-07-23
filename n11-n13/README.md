# Exact certificate for Dittert's conjecture in dimensions 11, 12, and 13

**Status:** reproducible working proof; not peer reviewed.

The proof note establishes that, for `n = 11, 12, 13`, the Dittert functional

\[
\Phi(A)=\prod_i r_i+\prod_j c_j-\operatorname{per}(A)
\]

on nonnegative `n x n` matrices with total entry sum `n` is uniquely maximized by the uniform matrix.

## Files

- `dittert_n11_n13_exact_proof_2026-07-23.pdf` - proof note.
- `dittert_n11_n13_exact_proof_2026-07-23.tex` - LaTeX source.
- `verify_dittert_n11_n13.py` - primary exact verifier; standard library only.
- `audit_dittert_n11_n13_sympy.py` - independent exact SymPy/Sturm/Ryser audit.
- `verify_dittert_n11_n13.log` - clean-run primary output.
- `audit_dittert_n11_n13_sympy.log` - clean-run independent output.
- `dittert_n11_n13_preflight.json` - PDF preflight report.
- `CHECKSUMS.sha256` - SHA-256 hashes for bundle contents.

## Reproduce

Python 3.11 or later is recommended. The recorded clean run used Python 3.13.5 and SymPy 1.14.0.

```bash
python3 -I verify_dittert_n11_n13.py
python3 -I audit_dittert_n11_n13_sympy.py
```

The final lines must be:

```text
ALL CASES CERTIFIED
INDEPENDENT AUDIT CERTIFIED
```

The primary program makes every correctness decision with exact `Fraction` arithmetic. The independent audit reconstructs the permanent polynomial by exact Ryser evaluations at `n+1` rational points and uses SymPy's independent exact Sturm implementation.

## Published mathematical dependencies

The proof note states the dependencies precisely. They are Hwang's positive-support theorem, Cheon-Wanless boundary exclusions, Li's doubly-superstochastic cut criterion, Hwang's staircase-face permanent theorem, the Minc/Pula-Song-Wanless averaging reduction for the two-independent-zero face, and the van der Waerden permanent theorem.
