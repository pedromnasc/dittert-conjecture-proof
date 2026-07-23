# Dittert `n=5` exact proof bundle

This directory contains an exact computer-assisted proof of Dittert's
conjecture in dimension `n=5`.

Run the proof check from this directory, without Python's `-O` option:

```bash
sha256sum -c SHA256SUMS
python3 verify_primary.py dittert_n5_exact_certificate.npz
```

The verifier reads the saved integer Cholesky factors, reconstructs the full
quintic certificate, and checks all 80,730 coefficients with
arbitrary-precision integer arithmetic. NumPy is used only to read the `.npz`
archive; it is not used for an arithmetic proof decision. The expected final
line is `CERTIFIED`.

`search_sos_n5.py` and `build_exact_certificate.py` record the numerical
discovery and rounding route. They are not part of the proof. The search
requires CVXPY, SciPy, and an SCS installation; the exact verifier requires
only Python 3.10 or newer and NumPy.

The optional discovery route can be reproduced in a separate virtual
environment:

```bash
python3 -m pip install -r requirements-search.txt
python3 search_sos_n5.py --output n5_search.npz
python3 build_exact_certificate.py n5_search.npz rebuilt_certificate.npz
python3 verify_primary.py rebuilt_certificate.npz
```

For the bundled certificate, the verifier additionally checks its known
minimum and maximum residuals as regression values. Rebuilt certificates may
have different extrema; they are accepted whenever their independently
reconstructed residual coefficients are all strictly positive.

The proof note has not yet undergone independent peer review. The certificate
is exact and reproducible, but both the mathematical reduction and verifier
should receive independent scrutiny before the result is cited as established.
