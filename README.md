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

What the script computes
Folded dihedral moments

For a given dihedral order 
𝑁
N and weight parameter 
𝑞
q, the script
computes the folded moments 
(
𝐼
1
,
𝐼
2
)
(I
1
	​

,I
2
	​

) in two independent ways:

closed-form analytic expressions for the geometric sums,

explicit folded sums over normalized weights 
𝑥
𝑟
∝
𝑞
𝑟
x
r
	​

∝q
r
.

Agreement between these methods is asserted numerically to high
precision, certifying the folded-zone moments used throughout the
construction.

Band-normalized curvature 
𝜅
κ

The script computes a stiffness (curvature) quantity 
𝜅
κ
associated with the transverse band of the folded spectrum. Several
modes are implemented:

Rayleigh-quotient Hessian evaluation,

Fisher-metric normalization,

Schur-complement removal of the collective (gauge) mode,

harmonic-window projections with 
∣
𝑘
∣
≤
𝐾
∣k∣≤K.

The default and recommended mode is the band-normalized
Hessian–Fisher–Schur construction with a finite harmonic window.

Mapping to the fine-structure constant

Given 
(
𝐼
1
,
𝜅
)
(I
1
	​

,κ), the effective quadratic form

𝑓
e
f
f
2
f
eff
2
	​

 is constructed as described in the paper, and the
plateau value of the fine-structure constant is computed via

𝛼
−
1
=
16
𝜋
3
(
1
−
𝐼
1
𝑓
e
f
f
2
)
2
.
α
−1
=16π
3
(1−
f
eff
2
	​

	​

I
1
	​

	​

)
2
.

At the lock point 
𝑞
=
𝜑
−
2
q=φ
−2
 and 
𝑚
𝜌
2
=
2
m
ρ
2
	​

=2, the resulting
value agrees with the CODATA 2022 value at sub–parts-per-billion
precision.

Detuning and robustness checks

To demonstrate that the result is not numerological, the script performs
a detuning sweep in the metric parameter 
𝑚
𝜌
2
m
ρ
2
	​

, recording the
deviation 
Δ
(
𝛼
−
1
)
Δ(α
−1
) from CODATA and identifying approximate
zero crossings.

Additional robustness diagnostics include:

variation of harmonic window size 
𝐾
K,

variation of Schur ridge parameter 
𝜏
τ,

𝐾
→
∞
K→∞ extrapolation of 
𝜅
(
𝐾
)
κ(K),

arithmetic precision sweeps.

Renormalization diagnostics

The script computes derivatives of 
𝛼
−
1
(
𝑞
)
α
−1
(q) at the lock point
and maps them to derivatives with respect to 
ln
⁡
𝜇
lnμ.

The universal one-loop QED coefficient is used to fix the linear
mapping between 
ln
⁡
𝑞
lnq and 
ln
⁡
𝜇
lnμ.

The local curvature determines the implied two-loop coefficient.

A diagnostic quantity c_required is reported, corresponding to the
second-order curvature of the scale map required to enforce the
universal two-loop coefficient.

This quantity is reported for analysis and is not used to tune the
plateau value.

Optional toy-model validation

An optional Gaussian 
𝐷
𝑁
D
N
	​

 toy model is included to verify internal
linear-algebra consistency:

equality of the Fisher matrix and the Hessian of 
log
⁡
𝑍
logZ,

diagonalization in the Fourier basis,

equivalence of projector-based and Schur-based removal of the zero
mode.

This check is purely diagnostic and does not enter the physical
construction.

Running the script

Basic execution:

python FSC_AllInOne.py


Common examples:

python FSC_AllInOne.py --N 12 --kappa-mode hessian-fisher-schur-pk --pK 2
python FSC_AllInOne.py --mrho2 2.0,2.2,2.4
python FSC_AllInOne.py --auto-c


All results are written to a user-specified output directory (default:
outputs/) in CSV or plain-text format.

Reproducibility

All numerical results reported in the associated manuscript can be
reproduced exactly using this script and the configurations specified in
the paper.

No hidden parameters, auxiliary data files, or external inputs are
required.
