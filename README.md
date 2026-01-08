# Bruna-SchurSymmetry-alpha--1-Computation

This repository contains a single, self-contained Python script,
`FSC_AllInOne.py`, which reproduces and validates the numerical results
reported in the paper

**Michael A. Bruna**,  
*Schur Curvature, Dihedral Symmetry, and a Locked Dimensionless Invariant.*

The script implements the full computational pipeline underlying the
dihedral \(D_N\) folded-spectrum construction that yields a locked value
of the fine-structure constant \(\alpha\), together with robustness,
detuning, and renormalization diagnostics.

No fitting, data-driven optimization, or external numerical input is
used. All quantities are determined by symmetry-fixed constructions
described in the paper.

---

## Contents

- `FSC_AllInOne.py` — all-in-one validation and diagnostic script
- `README.md` — this document

The script is intentionally monolithic to ensure transparency and
reproducibility.

---

## Dependencies

- Python 3.9 or later
- `mpmath` (arbitrary-precision arithmetic)

Install the dependency with:
```bash
pip install mpmath
