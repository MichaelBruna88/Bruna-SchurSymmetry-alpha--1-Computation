#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSC_AllInOne.py
v2.3.1 

What's included
---------------
• Folded D_N moments (I1, I2) with closed-form checks
• Band-normalized κ via Hessian/Fisher/Schur with harmonic windows (|k| ≤ K)
• Optional unit calibration (single constant factor) before mapping to α
• Detuning sweep in m_rho^2 and zero-crossing locator
• One-loop slope check fixed by universal QED coefficient (sets d ln q / d ln μ)
• Two-loop extraction from local curvature at the lock point
• Scale-map curvature diagnostic: compute c_required so universal two-loop holds
• Optional application of user-specified map curvature c (or auto=required)
• Tightening suite: ridge τ and K-sweep with K→∞ extrapolation
• Optional toy Gaussian D_N validation (ψ″ = Fisher, Schur vs projector)

Usage examples
--------------
python FSC_AllInOne.py
python FSC_AllInOne.py --N 12 --kappa-mode hessian-fisher-schur-pK --pK 2 --schur-tau 1e-3
python FSC_AllInOne.py --map-c -3.07e-6 --apply-map-c
python FSC_AllInOne.py --auto-c      # apply c_required that enforces universal two-loop

Dependencies:
  pip install mpmath
"""

from mpmath import mp
import argparse, csv, os
from typing import List, Tuple, Optional

# -----------------------
# Globals / constants
# -----------------------
DEFAULT_DPS = 100
PHI = (1 + mp.sqrt(5)) / 2
CODATA_ALPHA_INV = mp.mpf("137.035999084")
KAPPA_SCHUR_GEOM_DEFAULT = mp.mpf("0.793231802269")  # geometry-only constant

# =======================
# I/O helpers
# =======================
def _makedirs(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_csv(path, rows, header=None):
    _makedirs(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        w.writerows(rows)

def save_text(path, text: str):
    _makedirs(path)
    with open(path, "w") as f:
        f.write(text)

# =======================
# S0: Gaussian D_N toy (Ward + Fisher = ψ″ + Schur=projector)
# =======================
def toy_gaussian_dn(N: int = 12, m2: mp.mpf = mp.mpf("0.5")):
    # Build Laplacian L (ring), gradient D, K = m2 I + L
    L = [[mp.mpf(0) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        L[i][i] = 2
        L[i][(i+1)%N] = -1
        L[i][(i-1)%N] = -1

    D = [[mp.mpf(0) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        D[i][i] = 1
        D[i][(i-1)%N] = -1

    K = [[mp.mpf(0) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        K[i][i] = m2 + 2
        K[i][(i+1)%N] = -1
        K[i][(i-1)%N] = -1

    # Invert K (naive Gauss–Jordan)
    Kinv = gauss_jordan_inverse(K)
    # Fisher = Σ_JJ = D K^{-1} D^T
    DK = matmul(D, Kinv)
    Fisher = matmul(DK, transpose(D))

    # ψ(A) = 1/2 A^T Σ_JJ A → Hessian = Σ_JJ (finite-diff check)
    def logZ(A):
        return mp.mpf("0.5") * quad_form(Fisher, A)
    H_fd = hessian_fd(logZ, N, eps=mp.mpf("1e-6"))
    frob = frob_norm(diff(Fisher, H_fd))

    # k-basis diagonalization (discrete Fourier)
    F_k = to_fourier_basis(Fisher)
    offdiag_max = max_offdiag_abs(F_k)
    k0 = F_k[0][0]
    F_k_diag = [mp.re(F_k[i][i]) for i in range(N)]

    # Schur vs projector (remove k=0)
    F_T_site, F_schur_site = transverse_vs_schur(Fisher)
    frob_T_vs_S = frob_norm(diff(F_T_site, F_schur_site))
    evals_T = sorted(real_sym_eigs(F_T_site))
    kappa_T = evals_T[-1] if evals_T else mp.mpf("0")

    out = []
    out.append(f"||Fisher - Hessian_fd||_F = {mp.nstr(frob, 18)}")
    out.append(f"Max |off-diagonal in k-basis| = {mp.nstr(offdiag_max, 18)}")
    out.append(f"F_k[0,0] (k=0) = {mp.nstr(k0, 18)}")
    out.append(f"||F_T_site - F_schur_site||_F = {mp.nstr(frob_T_vs_S, 18)}")
    out.append(f"Top transverse eigenvalue (stiffness κ_T) = {mp.nstr(kappa_T, 18)}")
    out.append("F_k diagonal (real parts) by mode k:")
    out.append(", ".join(mp.nstr(v, 9) for v in F_k_diag))
    return "\n".join(out)

# --- linear algebra utils for S0 ---
def matmul(A, B):
    n = len(A); m = len(B[0]); p = len(B)
    C = [[mp.mpf(0) for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = mp.mpf(0)
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def transpose(A):
    return [list(row) for row in zip(*A)]

def gauss_jordan_inverse(A):
    n = len(A)
    M = [A[i][:] + [mp.mpf(1) if i==j else mp.mpf(0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = M[col][col]
        if mp.fabs(piv) < mp.mpf("1e-30"):
            for r in range(col+1, n):
                if mp.fabs(M[r][col]) > mp.mpf("1e-30"):
                    M[col], M[r] = M[r], M[col]
                    piv = M[col][col]; break
        invp = 1/piv
        M[col] = [invp*t for t in M[col]]
        for r in range(n):
            if r == col: continue
            fac = M[r][col]
            if fac == 0: continue
            M[r] = [M[r][c] - fac*M[col][c] for c in range(2*n)]
    return [row[n:] for row in M]

def quad_form(A, v):
    N = len(v)
    s = mp.mpf(0)
    for i in range(N):
        t = mp.mpf(0)
        for j in range(N):
            t += A[i][j]*v[j]
        s += v[i]*t
    return s

def hessian_fd(logZ, N, eps=mp.mpf("1e-6")):
    H = [[mp.mpf(0) for _ in range(N)] for _ in range(N)]
    e = [[mp.mpf(0) for _ in range(N)] for _ in range(N)]
    for i in range(N): e[i][i] = mp.mpf(1)
    for i in range(N):
        for j in range(N):
            A_pp = [eps*(e[i][k] + e[j][k]) for k in range(N)]
            A_pm = [eps*(e[i][k] - e[j][k]) for k in range(N)]
            A_mp = [eps*(-e[i][k] + e[j][k]) for k in range(N)]
            A_mm = [eps*(-e[i][k] - e[j][k]) for k in range(N)]
            H[i][j] = (logZ(A_pp) - logZ(A_pm) - logZ(A_mp) + logZ(A_mm)) / (4*eps*eps)
    return H

def frob_norm(A):
    return mp.sqrt(mp.fsum(A[i][j]*A[i][j] for i in range(len(A)) for j in range(len(A[0]))))

def diff(A, B):
    return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def to_fourier_basis(A):
    N = len(A)
    W = [[mp.e ** (-2j*mp.pi*i*j/N) for j in range(N)] for i in range(N)]
    Winv = [[mp.conj(W[j][i])/N for j in range(N)] for i in range(N)]
    return matmul(matmul(W, A), Winv)

def max_offdiag_abs(M):
    N = len(M)
    m = mp.mpf(0)
    for i in range(N):
        for j in range(N):
            if i==j: continue
            val = mp.fabs(M[i][j])
            if val > m: m = val
    return m

def real_sym_eigs(A):
    try:
        F = to_fourier_basis(A)
        return [mp.re(F[i][i]) for i in range(len(A))]
    except Exception:
        return [mp.mpf(0) for _ in range(len(A))]

def dot(a,b): return mp.fsum(a[i]*b[i] for i in range(len(a)))
def apply(A,v): return [mp.fsum(A[i][j]*v[j] for j in range(len(v))) for i in range(len(v))]

def orthonormal_complement(u):
    """GS-orthonormal basis of the complement to span{u}. Returns (u_normed, Q)."""
    N = len(u)
    nu = mp.sqrt(mp.fsum(t*t for t in u))
    if nu == 0: return [mp.mpf(0)]*N, []
    u = [t/nu for t in u]
    # Remove DC and re-normalize u
    dc = [mp.mpf(1)/mp.sqrt(N) for _ in range(N)]
    udot = mp.fsum(u[i]*dc[i] for i in range(N))
    u = [u[i] - udot*dc[i] for i in range(N)]
    un = mp.sqrt(mp.fsum(t*t for t in u))
    if un > mp.mpf("0"):
        u = [u[i]/un for i in range(N)]
    # Build orthonormal Q (complement)
    Q = []
    for k in range(N):
        e = [mp.mpf("0")]*N; e[k]=mp.mpf("1")
        c = dot(e,u); w = [e[i]-c*u[i] for i in range(N)]
        for q in Q:
            c2 = dot(w,q); w = [w[i]-c2*q[i] for i in range(N)]
        nw = mp.sqrt(mp.fsum(t*t for t in w))
        if nw > mp.mpf("1e-12"):
            Q.append([w[i]/nw for i in range(N)])
    return u, Q

def transverse_vs_schur(F):
    # Remove k=0 mode by projector in k-basis
    N = len(F)
    W = [[mp.e ** (-2j*mp.pi*i*j/N) for j in range(N)] for i in range(N)]
    Winv = [[mp.conj(W[j][i])/N for j in range(N)] for i in range(N)]
    Fk = matmul(matmul(W, F), Winv)
    for j in range(N): Fk[0][j] = 0
    for i in range(N): Fk[i][0] = 0
    F_T_site = matmul(matmul(Winv, Fk), W)

    # Schur: eliminate span{u=(1,...,1)/sqrt(N)}
    u = [mp.mpf(1)/mp.sqrt(N) for _ in range(N)]
    u, Q = orthonormal_complement(u)
    Au = apply(F, u)
    if not Q:
        return F_T_site, F_T_site
    m = len(Q)
    C = [[dot(Q[i], apply(F, Q[j])) for j in range(m)] for i in range(m)]
    b = [dot(Q[i], Au) for i in range(m)]
    tau = mp.mpf("1e-6")
    for i in range(m):
        C[i][i] += tau
    Cinv = gauss_jordan_inverse(C)
    y = [mp.fsum(Cinv[i][j]*b[j] for j in range(m)) for i in range(m)]
    F_schur = [[mp.mpf(0) for _ in range(N)] for _ in range(N)]
    for a in range(m):
        for b2 in range(m):
            coeff = C[a][b2] - y[a]*b[b2]
            for i in range(N):
                for j in range(N):
                    F_schur[i][j] += coeff * Q[a][i] * Q[b2][j]
    return F_T_site, F_schur

# =======================
# Core folded moments / operators
# =======================
def n(x, digits=12): return mp.nstr(x, digits)

def S0_closed_form(N: int, q: mp.mpf) -> mp.mpf:
    return q * (1 - q**N) / (1 - q)

def S1_closed_form(N: int, q: mp.mpf) -> mp.mpf:
    num = q * (1 - (N + 1) * q**N + N * q**(N + 1))
    den = (1 - q) ** 2
    return num / den

def S2_closed_form(N: int, q: mp.mpf) -> mp.mpf:
    num = (
        q * (1 + q)
        - (N + 1) ** 2 * q ** (N + 1)
        + (2 * N ** 2 + 2 * N - 1) * q ** (N + 2)
        - (N ** 2) * q ** (N + 3)
    )
    den = (1 - q) ** 3
    return num / den

def I1I2_closed(N: int, q: mp.mpf):
    S0 = S0_closed_form(N, q)
    S1 = S1_closed_form(N, q)
    S2 = S2_closed_form(N, q)
    return S1 / S0, S2 / S0

def I1I2_folded(N: int, q: mp.mpf):
    S0 = S0_closed_form(N, q)
    x = [mp.mpf("0")] + [q ** r / S0 for r in range(1, N + 1)]
    den = mp.fsum(x[1:])
    I1 = mp.fsum(mp.mpf(r) * x[r] for r in range(1, N + 1)) / den
    I2 = mp.fsum(mp.mpf(r * r) * x[r] for r in range(1, N + 1)) / den
    return I1, I2, x

def _cot_first_row(N: int):
    l = [mp.mpf("0")] * N
    for j in range(1, N):
        l[j] = mp.cos(mp.pi * j / N) / mp.sin(mp.pi * j / N)
    for j in range(1, N):
        l[N - j] = -l[j]
    return l

def _circulant_from_first_row(l):
    N = len(l)
    C = [[mp.mpf("0") for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            C[i][j] = l[(i - j) % N]
    return C

def _eigenvalue_k_of_circulant(l, k: int):
    N = len(l); w = mp.e ** (-2j * mp.pi / N)
    wk = mp.mpf("1"); s = mp.mpf("0"); step = w**k
    for j in range(N):
        s += l[j] * wk
        wk *= step
    return s

def _apply(A, vec):
    N = len(vec)
    return [mp.fsum(A[i][j]*vec[j] for j in range(N)) for i in range(N)]

def _largest_eigval_sym(A, iters=60):
    N = len(A)
    if N == 0: return mp.mpf("0")
    v = [mp.mpf("1")/mp.sqrt(N) for _ in range(N)]
    for _ in range(iters):
        w = _apply(A, v)
        nw = mp.sqrt(mp.fsum(t*t for t in w))
        if nw == 0: break
        v = [t/nw for t in w]
    Av = _apply(A, v)
    return mp.fabs(mp.fsum(v[i]*Av[i] for i in range(N)))

def _rayleigh_quotient(A, v):
    num = mp.mpf("0"); den = mp.mpf("0")
    for i in range(len(v)):
        den += v[i] * v[i]
        s = mp.mpf("0")
        for j in range(len(v)):
            s += A[i][j] * v[j]
        num += v[i] * s
    return num / den if den != 0 else mp.mpf("0")

def _proj_k_real_u(N: int, k: int):
    s = mp.sqrt(2/mp.mpf(N))
    u_cos = [s * mp.cos(2*mp.pi * k * i / N) for i in range(N)]
    u_sin = [s * mp.sin(2*mp.pi * k * i / N) for i in range(N)]
    return u_cos, u_sin

def _proj_k_le_K_real(N: int, K: int):
    P = [[mp.mpf("0") for _ in range(N)] for _ in range(N)]
    Ucols = []
    for k in range(1, K+1):
        uc, us = _proj_k_real_u(N, k)
        Ucols.append(uc); Ucols.append(us)
        for i in range(N):
            for j in range(N):
                P[i][j] += uc[i]*uc[j] + us[i]*us[j]
    return P, Ucols

# =======================
# κ modes (band-normalized)
# =======================
def kappa_mode_hessian_rayleigh(N: int, q: mp.mpf):
    I1, I2, x = I1I2_folded(N, q)
    l = _cot_first_row(N); H = _circulant_from_first_row(l)
    lam1 = _eigenvalue_k_of_circulant(l, 1)
    lam1_abs2 = (mp.re(lam1)**2 + mp.im(lam1)**2)
    HHt = matmul(H, transpose(H))
    S = [[HHt[i][j] / lam1_abs2 for j in range(N)] for i in range(N)]
    v = [mp.sqrt(x[r]) * (mp.mpf(r) - I1) for r in range(1, N+1)]
    return _rayleigh_quotient(S, v)

def kappa_mode_hessian_fisher(N: int, q: mp.mpf):
    I1, I2, x = I1I2_folded(N, q)
    sigma = mp.sqrt(I2 - I1*I1)
    l = _cot_first_row(N); H = _circulant_from_first_row(l)
    Ghs = [mp.sqrt(x[r]) for r in range(1, N+1)]
    Ht = [[Ghs[i]*H[i][j]*Ghs[j] for j in range(N)] for i in range(N)]
    lam1 = _eigenvalue_k_of_circulant(l, 1)
    lam1_abs2 = (mp.re(lam1)**2 + mp.im(lam1)**2)
    HtHtT = matmul(Ht, transpose(Ht))
    S_tilde = [[HtHtT[i][j]/lam1_abs2 for j in range(N)] for i in range(N)]
    vF = [(mp.mpf(i+1) - I1)/sigma for i in range(N)]
    return _rayleigh_quotient(S_tilde, vF)

def kappa_mode_hessian_fisher_schur(N: int, q: mp.mpf, tau=mp.mpf("1e-3")):
    I1, I2, x = I1I2_folded(N, q)
    sigma = mp.sqrt(I2 - I1*I1)
    l = _cot_first_row(N); H = _circulant_from_first_row(l)
    Ghs = [mp.sqrt(x[r]) for r in range(1, N+1)]
    Ht = [[Ghs[i]*H[i][j]*Ghs[j] for j in range(N)] for i in range(N)]
    lam1 = _eigenvalue_k_of_circulant(l, 1)
    lam1_abs2 = (mp.re(lam1)**2 + mp.im(lam1)**2)
    HtHtT = matmul(Ht, transpose(Ht))
    S_tilde = [[HtHtT[i][j]/lam1_abs2 for j in range(N)] for i in range(N)]
    vF = [(mp.mpf(i+1) - I1)/sigma for i in range(N)]
    return schur_curvature(S_tilde, vF, tau=tau)

def kappa_mode_hessian_fisher_schur_p1(N: int, q: mp.mpf, tau=mp.mpf("1e-3")):
    return kappa_mode_hessian_fisher_schur(N, q, tau=tau)

def kappa_mode_hessian_fisher_schur_pK(N: int, q: mp.mpf, K: int = 2, tau=mp.mpf("1e-3"), norm_space:str="PK"):
    Kmax = max(1, N//2)
    K = max(1, min(K, Kmax))
    I1, I2, x = I1I2_folded(N, q)
    sigma = mp.sqrt(I2 - I1*I1)
    l = _cot_first_row(N); H = _circulant_from_first_row(l)
    Ghs = [mp.sqrt(x[r]) for r in range(1, N+1)]
    Ht = [[Ghs[i]*H[i][j]*Ghs[j] for j in range(N)] for i in range(N)]
    lam1 = _eigenvalue_k_of_circulant(l, 1)
    lam1_abs2 = (mp.re(lam1)**2 + mp.im(lam1)**2)
    HtHtT = matmul(Ht, transpose(Ht))
    S_tilde = [[HtHtT[i][j]/lam1_abs2 for j in range(N)] for i in range(N)]
    vF = [(mp.mpf(i+1) - I1)/sigma for i in range(N)]
    Pk, _ = _proj_k_le_K_real(N, K)
    Ak = matmul(Pk, matmul(S_tilde, Pk))
    vk = _apply(Pk, vF)
    return schur_curvature(Ak, vk, tau=tau)

# =======================
# Schur machinery
# =======================
def _pseudo_inverse_block(Q, A, tau=mp.mpf("1e-3")):
    m = len(Q)
    if m == 0: return []
    B = [[mp.mpf("0") for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            val = mp.mpf("0")
            for k in range(len(Q[i])):
                aik = mp.fsum(A[k][t]*Q[j][t] for t in range(len(Q[i])))
                val += Q[i][k] * aik
            if i == j: val += tau
            B[i][j] = val
    Id = [[mp.mpf("0") if i!=j else mp.mpf("1") for j in range(m)] for i in range(m)]
    M = [B[i] + Id[i] for i in range(m)]
    for col in range(m):
        pivot = M[col][col]
        if mp.fabs(pivot) < mp.mpf("1e-30"): continue
        invp = 1/pivot
        M[col] = [invp*t for t in M[col]]
        for r in range(m):
            if r==col: continue
            fac = M[r][col]
            if fac==0: continue
            M[r] = [M[r][c] - fac*M[col][c] for c in range(2*m)]
    return [row[m:] for row in M]

def schur_curvature(A, v, tau=mp.mpf("1e-3")):
    u, Q = orthonormal_complement(v)
    Au = _apply(A, u)
    k0 = mp.fsum(u[i]*Au[i] for i in range(len(u)))
    if not Q: return k0
    m = len(Q)
    b = [mp.fsum(Q[i][k]*Au[k] for k in range(len(u))) for i in range(m)]
    C_inv = _pseudo_inverse_block(Q, A, tau=tau)
    y = [mp.fsum(C_inv[i][j]*b[j] for j in range(m)) for i in range(m)]
    return k0 - mp.fsum(b[i]*y[i] for i in range(m))

# =======================
# α map, slope, sequences
# =======================
def scale_kappa_with_mrho2(kappa_base_at_2: mp.mpf, m_rho2: mp.mpf) -> mp.mpf:
    mref = mp.mpf("2.0")
    return kappa_base_at_2 if m_rho2 == mref else kappa_base_at_2 * (mref / m_rho2)

def alpha_inv_from(I1: mp.mpf, f2: mp.mpf) -> mp.mpf:
    return 16 * mp.pi**3 * (1 - I1 / mp.sqrt(f2))**2

def f2_pre(N: int, I1: mp.mpf, m_rho2: mp.mpf) -> mp.mpf:
    return N - (4 * I1**2) / (N * m_rho2)

def f2_eff(N: int, I1: mp.mpf, m_rho2: mp.mpf, kappa_schur: mp.mpf) -> mp.mpf:
    return f2_pre(N, I1, m_rho2) + kappa_schur / N

def one_loop_slope_lnq(N: int, m_rho2: mp.mpf, q: mp.mpf, kappa_schur: mp.mpf) -> mp.mpf:
    I1, I2 = I1I2_closed(N, q)
    a = N + kappa_schur / N
    b = 4 / (N * m_rho2)
    f2 = a - b * I1 * I1
    uprime = I2 - I1 * I1
    A = mp.sqrt(f2)
    return -32 * mp.pi**3 * (1 - I1 / A) * uprime * a / (f2 ** (mp.mpf("1.5")))

def qed_norm_lnq_to_lnmu(N: int, q: mp.mpf, kappa_base_at_2: mp.mpf) -> mp.mpf:
    raw = one_loop_slope_lnq(N, mp.mpf("2.0"), q, kappa_base_at_2)
    return (mp.mpf("2") / (3 * mp.pi)) / mp.fabs(raw)

def kappa_pk_sequence(N, q, K_list, tau, norm_space):
    vals = []
    Kmax = max(1, N//2)
    for K in K_list:
        Kc = max(1, min(K, Kmax))
        kappa = kappa_mode_hessian_fisher_schur_pK(N, q, K=Kc, tau=mp.mpf(tau), norm_space=norm_space)
        vals.append((Kc, kappa))
    seen = {}
    for K,k in vals: seen[K] = k
    return sorted(seen.items(), key=lambda t: t[0])

def extrapolate_kappa_infty_linear(kappa_vs_K):
    xs = [mp.mpf(1)/mp.mpf(K) for K, _ in kappa_vs_K]
    ys = [mp.mpf(k) for _, k in kappa_vs_K]
    Sx = mp.fsum(xs); Sy = mp.fsum(ys)
    Sxx = mp.fsum(x*x for x in xs)
    Sxy = mp.fsum(xs[i]*ys[i] for i in range(len(xs)))
    n = mp.mpf(len(xs))
    denom = n*Sxx - Sx*Sx
    if mp.fabs(denom) < mp.mpf("1e-30"): return ys[-1]
    a = (Sy*Sxx - Sx*Sxy) / denom
    return a

def extrapolate_kappa_infty_quad(kappa_vs_K):
    xs = [mp.mpf(1)/mp.mpf(K) for K,_ in kappa_vs_K]
    ys = [mp.mpf(k) for _,k in kappa_vs_K]
    Sx=mp.fsum(xs); Sy=mp.fsum(ys)
    Sxx=mp.fsum(x*x for x in xs)
    Sxxx=mp.fsum(x*x*x for x in xs)
    Sxxxx=mp.fsum(x*x*x*x for x in xs)
    Sxy=mp.fsum(xs[i]*ys[i] for i in range(len(xs)))
    Sxxy=mp.fsum((xs[i]**2)*ys[i] for i in range(len(xs)))
    n=mp.mpf(len(xs))
    M = [[n, Sx, Sxx],[Sx, Sxx, Sxxx],[Sxx, Sxxx, Sxxxx]]
    R = [Sy, Sxy, Sxxy]
    for i in range(3):
        piv = M[i][i]
        if mp.fabs(piv) < mp.mpf("1e-30"): return ys[-1]
        invp = 1/piv
        M[i] = [invp*t for t in M[i]]; R[i] *= invp
        for j in range(3):
            if j==i: continue
            f = M[j][i]
            M[j] = [M[j][k]-f*M[i][k] for k in range(3)]
            R[j] -= f*R[i]
    a = R[0]
    return a

# =======================
# Helpers for α(q), derivatives, and scale mapping
# =======================
def alpha_inv_of_q(N:int, q:mp.mpf, m_rho2:mp.mpf, kappa_cal:mp.mpf) -> mp.mpf:
    I1, _ = I1I2_closed(N, q)
    fpre = f2_pre(N, I1, m_rho2)
    feff = fpre + kappa_cal / N
    return alpha_inv_from(I1, feff)  # y(q) = α^{-1}(q)

def d_dlnq(f, q0):
    # d/d(ln q) = q * d/dq
    return q0 * mp.diff(lambda qq: f(qq), q0)

def d2_dlnq2(f, q0):
    # d^2/d(ln q)^2 = q * d/dq ( q * d/dq f )
    return q0 * mp.diff(lambda qq: qq * mp.diff(lambda z: f(z), qq), q0)

def map_derivatives_to_mu(S1_q, S2_q, s, c=mp.mpf("0")):
    """
    ln q = a + s t + 0.5 c t^2,  t = ln μ - ln μ0
    ⇒  S1_μ = s S1_q
        S2_μ = s^2 S2_q + c S1_q
    """
    S1_mu = s * S1_q
    S2_mu = (s * s) * S2_q + c * S1_q
    return S1_mu, S2_mu

# =======================
# Optional helpers: weights/truncation
# =======================
def apply_weight_family(x_raw: List[mp.mpf], family: str = "uniform"):
    if family.lower() == "uniform":
        return x_raw
    N = len(x_raw)
    if family.lower() == "cosine":
        w = [mp.mpf("0.5")*(1+mp.cos(mp.pi*(i)/(N))) for i in range(N)]
    elif family.lower() == "gaussian":
        mu = mp.mpf(N)/2; sig = mp.mpf(N)/6
        w = [mp.e**(- (i-mu)**2 /(2*sig**2)) for i in range(N)]
    else:
        return x_raw
    s = mp.fsum(w[i]*x_raw[i] for i in range(N))
    return [ (w[i]*x_raw[i])/s for i in range(N) ]

def truncate_M(x_raw: List[mp.mpf], M: Optional[int] = None, taper: str = "hard"):
    if not M or M >= len(x_raw): return x_raw
    N = len(x_raw)
    if taper == "hard":
        xr = [x_raw[i] if i < M else mp.mpf("0") for i in range(N)]
    elif taper == "linear":
        xr = [x_raw[i]*(mp.mpf(M-i)/M) if i < M else mp.mpf("0") for i in range(N)]
    else:
        xr = x_raw[:]
    s = mp.fsum(xr)
    return [t/s for t in xr] if s != 0 else xr

# =======================
# Unit calibration
# =======================
def compute_unit_factor_from_measured(measured_kappa: mp.mpf, scheme: str = "geom") -> mp.mpf:
    """
    Compute a single multiplicative factor c so that κ_cal = c * κ_meas before mapping to α.
    - 'none' → c = 1
    - 'geom' → c = κ_geom / κ_meas   (if κ_meas ≠ 0)
    """
    if scheme == "none":
        return mp.mpf("1")
    if measured_kappa == 0:
        return mp.mpf("1")
    if scheme == "geom":
        return KAPPA_SCHUR_GEOM_DEFAULT / measured_kappa
    return mp.mpf("1")

# =======================
# Wrappers for runs and exports
# =======================
def ainv_and_delta_at_mrho2(
    N:int, q:mp.mpf, mrho2:mp.mpf, kappa_base:mp.mpf,
    scale_with_mrho2_flag:bool, unit_factor: mp.mpf = mp.mpf("1")
) -> Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    I1, _ = I1I2_closed(N, q)
    kappa_here = (scale_kappa_with_mrho2(kappa_base, mrho2) if scale_with_mrho2_flag else kappa_base)
    kappa_here *= unit_factor
    fpre = f2_pre(N, I1, mrho2)
    feff = fpre + kappa_here / N
    ainv = alpha_inv_from(I1, feff)
    delta = ainv - CODATA_ALPHA_INV
    return kappa_here, fpre, feff, delta

def alpha_from_kappa(
    N, q, mrho2, kappa, unit_factor: mp.mpf = mp.mpf("1"),
    scale_with_mrho2_flag: bool = True
):
    I1, _ = I1I2_closed(N, q)
    k_here = scale_kappa_with_mrho2(kappa, mrho2) if scale_with_mrho2_flag else kappa
    k_here *= unit_factor
    fpre = f2_pre(N, I1, mrho2)
    feff = fpre + k_here / N
    return alpha_inv_from(I1, feff)

def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))

def tighten_report(
    N, q, mrho2, scale_with_mrho2_flag, tau_grid, norms,
    unit_factor: mp.mpf, out_prefix="outputs"
):
    print_header("Tightening toward CODATA (first-principles, SAFE fits)")
    Kmax = max(1, N//2)
    K_grid   = list(range(1, Kmax+1))
    rows = []
    for norm in norms:
        for tau in tau_grid:
            for K in K_grid:
                kappa = kappa_mode_hessian_fisher_schur_pK(N, q, K=K, tau=mp.mpf(tau), norm_space=norm)
                kappa_cal = unit_factor * kappa
                ainv  = alpha_from_kappa(N, q, mrho2, kappa_cal, unit_factor=mp.mpf("1"), scale_with_mrho2_flag=scale_with_mrho2_flag)
                delta = ainv - CODATA_ALPHA_INV
                rows.append((K, norm, mp.nstr(tau,6), kappa_cal, ainv, delta))
    rows.sort(key=lambda r: mp.fabs(r[5]))
    print(f"{'rank':>4} {'K':>2} {'norm':>4} {'τ':>8} {'κ_base':>14} {'α^-1@2.0':>14} {'Δ vs CODATA':>14}")
    for i, (K,norm,tau,kappa,ainv,delta) in enumerate(rows[:8], start=1):
        print(f"{i:>4} {K:>2} {norm:>4} {tau:>8} {mp.nstr(kappa,12):>14} "
              f"{mp.nstr(ainv,12):>14} {mp.nstr(delta,12):>14}")
    save_csv(f"{out_prefix}/ridge_sweep.csv", 
             [[K,norm,tau, mp.nstr(k,18), mp.nstr(a,18), mp.nstr(d,18)] for (K,norm,tau,k,a,d) in rows],
             ["K","norm","tau","kappa_base_cal","alpha_inv","delta_vs_CODATA"])
    best = rows[0]
    print(f"\nBest (lock m_rho^2=2.0) from grid: K={best[0]}, norm={best[1]}, τ={best[2]}, "
          f"α^-1={mp.nstr(best[4],12)}, Δ={mp.nstr(best[5],12)}")
    # Extrapolation on best settings
    K_list_fit = [K for K in range(2, Kmax+1)]
    tau_ex = mp.mpf(best[2]); norm_ex = best[1]
    seq_raw = kappa_pk_sequence(N, q, K_list_fit, tau=tau_ex, norm_space=norm_ex)
    seq = [(K, unit_factor*k) for (K,k) in seq_raw]
    print_header(f"K→∞ extrapolation on {norm_ex} (τ={best[2]})")
    print("K   κ(K) [calibrated]")
    for K,k in seq:
        print(f"{K:>2} {mp.nstr(k,12)}")
    save_csv(f"{out_prefix}/kappa_vs_K.csv", [(int(K), mp.nstr(k,18)) for K,k in seq], ["K","kappa(K)_cal"])
    kappa_inf_lin  = extrapolate_kappa_infty_linear(seq)
    kappa_inf_quad = extrapolate_kappa_infty_quad(seq)
    kappa_inf = kappa_inf_quad if kappa_inf_lin < mp.mpf("0") else kappa_inf_lin
    if kappa_inf < mp.mpf("0"): kappa_inf = mp.mpf("0")
    ainv_inf  = alpha_from_kappa(N, q, mrho2, kappa_inf, unit_factor=mp.mpf("1"), scale_with_mrho2_flag=scale_with_mrho2_flag)
    delta_inf = ainv_inf - CODATA_ALPHA_INV
    txt = []
    txt.append(f"κ∞ (linear 1/K) = {mp.nstr(kappa_inf_lin,18)}")
    txt.append(f"κ∞ (quad 1/K^2) = {mp.nstr(kappa_inf_quad,18)}")
    txt.append(f"κ∞ (chosen ≥0)  = {mp.nstr(kappa_inf,18)}")
    txt.append(f"α^-1@2.0 (κ∞)  = {mp.nstr(ainv_inf,18)}")
    txt.append(f"Δ vs CODATA    = {mp.nstr(delta_inf,18)}")
    save_text(f"{out_prefix}/kappa_extrapolation.txt", "\n".join(txt))
    

# =======================
# Main driver
# =======================
def run(
    N: int,
    m_rho2_list: List[mp.mpf],
    dps: int,
    do_N_sweep: bool,
    kappa_mode: str,
    kappa_geom_base_if_geom: mp.mpf,
    scale_with_mrho2_flag: bool,
    pK: int,
    schur_tau: mp.mpf,
    norm_space: str,
    unit_calib_scheme: str = "geom",
    do_tighten_report: bool = True,
    show_codata_presets: bool = True,
    out_prefix: str = "outputs",
    run_toy_check: bool = False,
    dps_list: Optional[List[int]] = None,
    map_c: mp.mpf = mp.mpf("0"),
    apply_map_c: bool = False,
    auto_c: bool = False,
):
    mp.dps = dps
    q = PHI ** (-2)

    if run_toy_check:
        print_header("S0: Gaussian toy validation")
        report = toy_gaussian_dn(N=12, m2=mp.mpf("0.5"))
        print(report)
        save_text(f"{out_prefix}/toy_validation.txt", report)

    print("=== Folded D_N → stiffness → α^{-1} (band-normalized pK + unit calibration) ===")
    print(f"mp.dps={mp.dps}")
    print(f"N={N},  q=φ^(-2)={n(q, 20)}")

    # I1, I2 certifications
    I1_c, I2_c = I1I2_closed(N, q)
    I1_f, I2_f, _ = I1I2_folded(N, q)
    assert mp.almosteq(I1_c, I1_f, rel_eps=mp.mpf("1e-14"))
    assert mp.almosteq(I2_c, I2_f, rel_eps=mp.mpf("1e-12"))

    print_header("Folded-zone moments (certified)")
    print(f"I1 (closed)  = {n(I1_c, 12)}")
    print(f"I2 (closed)  = {n(I2_c, 12)}")
    print(f"I1 (folded)  = {n(I1_f, 12)}")
    print(f"I2 (folded)  = {n(I2_f, 12)}")

    # Γ diagnostic (optional)
    print_header("Γ via Fourier/Schur closure (diagnostic)")
    Gamma = gamma_hilbert_from_cot_symbol(N_spec=240)
    print(f"Γ ≈ {n(Gamma, 12)}  (diagnostic only)")

    # κ baseline for chosen mode
    mode = kappa_mode.lower()
    header = f"κ_Schur baseline at m_rho^2=2.0  [mode = {kappa_mode}]"
    if mode == "hessian-fisher-schur-pk":
        header += f", K={pK}, norm={norm_space}, τ={n(schur_tau, 6)}"
    print_header(header)

    if mode == "geom":
        kappa_base = kappa_geom_base_if_geom
    elif mode == "hessian-rayleigh":
        kappa_base = kappa_mode_hessian_rayleigh(N, q)
    elif mode == "hessian-fisher":
        kappa_base = kappa_mode_hessian_fisher(N, q)
    elif mode == "hessian-fisher-schur":
        kappa_base = kappa_mode_hessian_fisher_schur(N, q, tau=schur_tau)
    elif mode == "hessian-fisher-schur-p1":
        kappa_base = kappa_mode_hessian_fisher_schur_p1(N, q, tau=schur_tau)
    elif mode == "hessian-fisher-schur-pk":
        kappa_base = kappa_mode_hessian_fisher_schur_pK(N, q, K=pK, tau=schur_tau, norm_space=norm_space)
    else:
        raise ValueError("Unknown kappa_mode")

    # Unit calibration factor (single constant)
    unit_factor = compute_unit_factor_from_measured(kappa_base, scheme=unit_calib_scheme)
    print(f"Unit calibration factor c = {n(unit_factor, 12)}  (scheme={unit_calib_scheme})")
    print(f"κ_Schur(base, calibrated) = {n(unit_factor * kappa_base, 15)}")

    # α^{-1} & detuning sweep
    print_header("Lock-point α^{-1} and detuning sweep")
    rows = []
    print(f"{'m_rho^2':>9} {'κ_Schur':>18} {'f_pre^2':>14} {'f_eff^2':>14} {'α^{-1}':>14} {'Δ vs CODATA':>16}")
    sweep_deltas = []
    for mr in m_rho2_list:
        mr = mp.mpf(mr)
        kappa_here, fpre, feff, delta = ainv_and_delta_at_mrho2(
            N, q, mr, kappa_base, scale_with_mrho2_flag, unit_factor=unit_factor
        )
        ainv = alpha_inv_from(I1_c, feff)
        sweep_deltas.append(delta)
        rows.append([n(mr,8), n(kappa_here,16), n(fpre,16), n(feff,16), n(ainv,16), n(delta,16)])
        print(f"{rows[-1][0]:>9} {rows[-1][1]:>18} {rows[-1][2]:>14} {rows[-1][3]:>14} {rows[-1][4]:>14} {rows[-1][5]:>16}")
    save_csv(f"{out_prefix}/detuning_sweep.csv",
             rows, ["m_rho^2","kappa_cal","f_pre2","f_eff2","alpha_inv","delta_vs_CODATA"])

    zc = zero_crossing(m_rho2_list, sweep_deltas)
    if zc is not None:
        print(f"\nApprox. zero-crossing (Δ≈0) between points → m_rho^2 ≈ {n(zc, 8)}")
        save_text(f"{out_prefix}/detuning_zero_crossing.txt", f"{n(zc,8)}")

    # === One-loop & two-loop with ln q → ln μ mapping ===
    print_header("One-loop & two-loop at the lock (with ln q → ln μ mapping)")
    q_star = PHI**(-2)
    m_rho2_lock = mp.mpf("2.0")
    kappa_cal = unit_factor * kappa_base

    # y(q) = α^{-1}(q) and value at the lock
    y_q = lambda qq: alpha_inv_of_q(N, qq, m_rho2_lock, kappa_cal)
    y0  = y_q(q_star)  # <-- define y0 BEFORE using it to set S1 target

    # Raw geometric derivatives in ln q
    S1_q = d_dlnq(y_q, q_star)        # dy/d(ln q)
    S2_q = d2_dlnq2(y_q, q_star)      # d²y/d(ln q)²

    # Exact one-loop target including b1/y0 term; fixes s via S1_mu_univ = s*S1_q
    S1_mu_univ = - ( 2/(3*mp.pi) + (1/(4*mp.pi**2))/y0 )
    s = S1_mu_univ / S1_q

    # Map derivatives with *linear* mapping (c=0)
    S1_mu_lin, S2_mu_lin = map_derivatives_to_mu(S1_q, S2_q, s, c=mp.mpf("0"))

    print(f"S1_q  = d(α⁻¹)/d ln q @ lock : {n(S1_q, 15)}")
    print(f"s     = d ln q / d ln μ     : {n(s, 18)}")
    print(f"S1_μ  = d(α⁻¹)/d ln μ        : {n(S1_mu_lin, 15)}   (expect {n(S1_mu_univ, 15)})")

    # Two-loop extraction under linear mapping
    b1_target = 1/(4*mp.pi**2)  # nf = 1
    b0_target = 2/(3*mp.pi)     # nf = 1

    print("\nTwo-loop from local curvature (linear mapping c=0):")
    print(f"y0    = α⁻¹(lock)                  : {n(y0, 15)}")
    print(f"S2_μ  = d²(α⁻¹)/d(ln μ)² (linear)   : {n(S2_mu_lin, 15)}")
    b1_est_lin = (S2_mu_lin * y0**2) / S1_mu_lin
    b0_est_lin = -S1_mu_lin - b1_est_lin / y0
    print("β(α)  = b0 α² + b1 α³ (extracted vs universal):")
    print(f"b0_est(lin) = {n(b0_est_lin, 15)}    b0_target = {n(b0_target, 15)}    Δ = {n(b0_est_lin - b0_target, 10)}")
    print(f"b1_est(lin) = {n(b1_est_lin, 15)}    b1_target = {n(b1_target, 15)}    Δ = {n(b1_est_lin - b1_target, 10)}")

    save_text(
        f"{out_prefix}/two_loop_linear.txt",
        "Two-loop (linear scale map c=0)\n"
        f"y0={n(y0,30)}\n"
        f"S1_mu={n(S1_mu_lin,30)}  expected={n(S1_mu_univ,30)}\n"
        f"S2_mu={n(S2_mu_lin,30)}\n"
        f"b0_est={n(b0_est_lin,30)}  b0_target={n(b0_target,30)}\n"
        f"b1_est={n(b1_est_lin,30)}  b1_target={n(b1_target,30)}\n"
    )

    # ---- Diagnose second-order scale mapping ln q -> ln μ ----
    print_header("Scale-map curvature needed to hit universal two-loop")
    S2_mu_req = (b1_target * S1_mu_lin) / (y0**2)
    c_required = (S2_mu_req - (s*s)*S2_q) / S1_q
    S1_mu_req, S2_mu_with_c_req = map_derivatives_to_mu(S1_q, S2_q, s, c=c_required)
    b1_with_c_req = (S2_mu_with_c_req * y0**2) / S1_mu_req
    b0_with_c_req = -S1_mu_req - b1_with_c_req / y0

    print(f"c_required (2nd-order in ln q(t)) : {n(c_required, 12)}")
    print(f"c/s (dimensionless)               : {n(c_required/s, 12)}")
    print(f"S2_μ required by universal b1     : {n(S2_mu_req, 15)}")
    print(f"b1 with c_required                : {n(b1_with_c_req, 15)}   (target {n(b1_target, 15)})")
    print(f"b0 with c_required                : {n(b0_with_c_req, 15)}   (target {n(b0_target, 15)})")

    save_text(f"{out_prefix}/scale_curvature_diagnosis.txt",
              f"S1_q={n(S1_q,30)}\nS2_q={n(S2_q,30)}\n"
              f"s={n(s,30)}\n"
              f"c_required={n(c_required,30)}\n"
              f"c_over_s={n(c_required/s,30)}\n"
              f"S2_mu_required={n(S2_mu_req,30)}\n"
              f"b1_with_c_required={n(b1_with_c_req,30)}  target={n(b1_target,30)}\n"
              f"b0_with_c_required={n(b0_with_c_req,30)}  target={n(b0_target,30)}\n")

    # ---- Optional: apply user-provided c or auto-apply c_required
    applied = None
    if auto_c:
        applied = ("auto(c_required)", c_required)
    elif apply_map_c:
        applied = ("user(c)", map_c)

    if applied is not None:
        label, c_use = applied
        print_header(f"Applied scale curvature: {label} = {n(c_use, 12)}")
        S1_mu_app, S2_mu_app = map_derivatives_to_mu(S1_q, S2_q, s, c=c_use)
        b1_app = (S2_mu_app * y0**2) / S1_mu_app
        b0_app = -S1_mu_app - b1_app / y0
        print(f"S1_μ (applied)  : {n(S1_mu_app, 15)}")
        print(f"S2_μ (applied)  : {n(S2_mu_app, 15)}")
        print(f"b0 (applied)    : {n(b0_app, 15)}   target {n(b0_target, 15)}   Δ={n(b0_app-b0_target,10)}")
        print(f"b1 (applied)    : {n(b1_app, 15)}   target {n(b1_target, 15)}   Δ={n(b1_app-b1_target,10)}")
        save_text(f"{out_prefix}/two_loop_applied_c.txt",
                  f"c_used={n(c_use,30)}\n"
                  f"S1_mu={n(S1_mu_app,30)}\nS2_mu={n(S2_mu_app,30)}\n"
                  f"b0={n(b0_app,30)}  target={n(b0_target,30)}\n"
                  f"b1={n(b1_app,30)}  target={n(b1_target,30)}\n")

    # Optional N sweep
    if do_N_sweep:
        print_header("Small N-sweep around 12 (lock robustness)")
        ns_rows = []
        for Ntest in [N-2, N-1, N, N+1, N+2]:
            if Ntest <= 2: continue
            I1t, I2t = I1I2_closed(Ntest, q)
            kappa_base_t = compute_kappa_base(Ntest, q, kappa_mode, pK, schur_tau, norm_space)
            fpre_t = f2_pre(Ntest, I1t, mp.mpf("2.0"))
            feff_t = fpre_t + (unit_factor * kappa_base_t) / Ntest
            ainv_t = alpha_inv_from(I1t, feff_t)
            y_q_t = lambda qq: alpha_inv_of_q(Ntest, qq, m_rho2_lock, unit_factor*kappa_base_t)
            S1_q_t = d_dlnq(y_q_t, q)
            s_t = S1_mu_univ / S1_q_t
            S1_mu_t, _ = map_derivatives_to_mu(S1_q_t, mp.mpf("0"), s_t, c=mp.mpf("0"))
            ns_rows.append([Ntest, n(unit_factor * kappa_base_t,10), n(I1t,10), n(ainv_t,10), n(S1_mu_t,10)])
            print(f"N={Ntest:2d} | κ_base(cal)={ns_rows[-1][1]}  I1={ns_rows[-1][2]}  α⁻¹={ns_rows[-1][3]}  dα⁻¹/dlnμ={ns_rows[-1][4]}")
        save_csv(f"{out_prefix}/alpha_by_N.csv", ns_rows, ["N","kappa_base_cal","I1","alpha_inv","dα^-1/dlnμ"])

    # CODATA presets comparison (at m_rho^2=2.0)
    if show_codata_presets:
        print_header("CODATA boosters @ m_rho^2 = 2.0 (band-normalized + unit-calibrated)")
        presets = [
            ("pK K=2, norm=PK, τ=1e-3", lambda: kappa_mode_hessian_fisher_schur_pK(N, q, K=2, tau=mp.mpf("1e-3"), norm_space="PK"), True),
            ("pK K=2, norm=P1, τ=1e-3", lambda: kappa_mode_hessian_fisher_schur_pK(N, q, K=2, tau=mp.mpf("1e-3"), norm_space="P1"), True),
            ("pK K=2, norm=PK, τ=1e-2", lambda: kappa_mode_hessian_fisher_schur_pK(N, q, K=2, tau=mp.mpf("1e-2"), norm_space="PK"), True),
            ("p1 (|k|=1), τ=1e-3",     lambda: kappa_mode_hessian_fisher_schur_p1(N, q, tau=mp.mpf("1e-3")), True),
            ("geom (paper κ)",         lambda: KAPPA_SCHUR_GEOM_DEFAULT, False),
        ]
        table_rows = []
        print(f"{'Preset':<28} {'κ_base':>16} {'α^-1 @2.0':>16} {'Δ vs CODATA':>16}")
        for name, fn, apply_cal in presets:
            kappa_b_raw = fn()
            kappa_b = (unit_factor * kappa_b_raw) if apply_cal else kappa_b_raw
            ainv = alpha_from_kappa(N, q, mp.mpf("2.0"), kappa_b, unit_factor=mp.mpf("1"))
            delta = ainv - CODATA_ALPHA_INV
            table_rows.append([name, n(kappa_b,12), n(ainv,12), n(delta,12)])
            print(f"{name:<28} {table_rows[-1][1]:>16} {table_rows[-1][2]:>16} {table_rows[-1][3]:>16}")
        save_csv(f"{out_prefix}/codata_presets.csv", table_rows, ["preset","kappa_base_cal","alpha_inv@2.0","delta_vs_CODATA"])

    # Tightening / K→∞ extrapolation
    if do_tighten_report:
        tighten_report(
            N=N, q=q, mrho2=mp.mpf("2.0"),
            scale_with_mrho2_flag=scale_with_mrho2_flag,
            tau_grid=[mp.mpf("1e-4"), mp.mpf("1e-3"), mp.mpf("1e-2"), mp.mpf("5e-2")],
            norms=["P1","PK"],
            unit_factor=unit_factor,
            out_prefix=out_prefix
        )

    print_header("Targets / Notes")
    print(f"CODATA α⁻¹ (2022): {CODATA_ALPHA_INV}")
    print("• I1, I2 certified (closed = folded).")
    print("• κ modes: geom | hessian-rayleigh | hessian-fisher | hessian-fisher-schur | hessian-fisher-schur-p1 | hessian-fisher-schur-pk")
    print(f"• pK/P1 use band λ₁² normalization; optional unit calibration: {unit_calib_scheme}.")
    print("• One-loop fixed by universal coefficient (exact form used); two-loop from local curvature.")
    print("• 'c_required' is reported; pass --auto-c to apply it when printing applied values.")
    print("• Γ is diagnostic only (not used to tune κ).")

def compute_kappa_base(N, q, kappa_mode, pK, schur_tau, norm_space):
    m = kappa_mode.lower()
    if m == "geom": return KAPPA_SCHUR_GEOM_DEFAULT
    if m == "hessian-rayleigh": return kappa_mode_hessian_rayleigh(N, q)
    if m == "hessian-fisher": return kappa_mode_hessian_fisher(N, q)
    if m == "hessian-fisher-schur": return kappa_mode_hessian_fisher_schur(N, q, tau=schur_tau)
    if m == "hessian-fisher-schur-p1": return kappa_mode_hessian_fisher_schur_p1(N, q, tau=schur_tau)
    if m == "hessian-fisher-schur-pk": return kappa_mode_hessian_fisher_schur_pK(N, q, K=pK, tau=schur_tau, norm_space=norm_space)
    raise ValueError("Unknown mode")

def zero_crossing(mrho2_list: List[mp.mpf], deltas: List[mp.mpf]) -> Optional[mp.mpf]:
    for i in range(len(mrho2_list)-1):
        a, b = mp.mpf(mrho2_list[i]), mp.mpf(mrho2_list[i+1])
        fa, fb = deltas[i], deltas[i+1]
        if fa == 0: return a
        if fa*fb < 0: return a - fa*(b-a)/(fb-fa)
    return None

def gamma_hilbert_from_cot_symbol(N_spec: int = 240) -> mp.mpf:
    l = [mp.mpf("0")] * N_spec
    for j in range(1, N_spec):
        ang = mp.pi * j / N_spec
        l[j] = mp.cos(ang) / mp.sin(ang)
    for j in range(1, N_spec):
        l[N_spec - j] = -l[j]
    def eig_at_k(k: int):
        tw = mp.e ** (-2j * mp.pi * k / N_spec)
        s = mp.mpf("0")
        for j in range(N_spec):
            s += l[j] * (tw ** j)
        return s
    lam1 = eig_at_k(1)
    return 1 / mp.fabs(lam1)

# =======================
# CLI
# =======================
def parse_args():
    p = argparse.ArgumentParser(description="All-in-one FSC supplement runner (band-normalized pK + unit calibration)")
    p.add_argument("--N", type=int, default=12, help="Folded zone size (default: 12)")
    p.add_argument("--mrho2", type=str, default="2.0,2.2,2.4,2.6", help="Comma-separated list for detuning sweep")
    p.add_argument("--dps", type=int, default=DEFAULT_DPS, help=f"mpmath precision (default: {DEFAULT_DPS})")
    p.add_argument("--N-sweep", action="store_true", help="Run a small N sweep around N")
    p.add_argument("--kappa-mode", type=str, default="hessian-fisher-schur-pk",
                   choices=["geom","hessian-rayleigh","hessian-fisher","hessian-fisher-schur","hessian-fisher-schur-p1","hessian-fisher-schur-pk"],
                   help="How to obtain κ_Schur baseline at m_rho^2=2.0")
    p.add_argument("--kappa-geom", type=str, default=str(KAPPA_SCHUR_GEOM_DEFAULT),
                   help="If --kappa-mode geom, use this κ at m_rho^2=2.0")
    p.add_argument("--no-scale-with-mrho2", action="store_true",
                   help="If set, κ is held constant across m_rho^2 (no 2.0/m_rho^2 scaling).")
    p.add_argument("--pK", type=int, default=2, help="K for hessian-fisher-schur-pK (harmonic window |k|≤K)")
    p.add_argument("--schur-tau", type=str, default="1e-3", help="Ridge τ for Schur complement (e.g., 1e-3 or 1e-2)")
    p.add_argument("--norm", type=str, default="PK", choices=["P1","PK"],
                   help="(kept for compatibility; pK uses band normalization regardless)")
    p.add_argument("--unit-calib", type=str, default="geom",
                   choices=["none","geom"],
                   help="Multiply κ by a fixed unit factor before mapping to α. 'geom' uses κ_geom/κ_meas at the lock point.")
    p.add_argument("--no-tighten", action="store_true", help="Disable the tightening suite (K-sweep & extrapolation)")
    p.add_argument("--no-codata-presets", action="store_true", help="Disable the CODATA booster presets table")
    p.add_argument("--out", type=str, default="outputs", help="Output folder prefix")
    p.add_argument("--run-toy-check", action="store_true", help="Run the Gaussian toy (S0) validation and save report")
    p.add_argument("--dps-list", type=str, default="", help="Comma-separated precision list for stability sweep (e.g. 64,100,200)")
    # Scale mapping options
    p.add_argument("--map-c", type=str, default="0", help="Second-order curvature c in ln q(ln μ). Default 0.")
    p.add_argument("--apply-map-c", action="store_true", help="Apply the provided --map-c when reporting two-loop.")
    p.add_argument("--auto-c", action="store_true", help="Apply c_required that enforces universal two-loop.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mr_list = [mp.mpf(s.strip()) for s in args.mrho2.split(",") if s.strip()]
    dps_list = [int(s.strip()) for s in args.dps_list.split(",") if s.strip()] if args.dps_list else None
    run(
        N=args.N,
        m_rho2_list=mr_list,
        dps=args.dps,
        do_N_sweep=args.N_sweep,
        kappa_mode=args.kappa_mode,
        kappa_geom_base_if_geom=mp.mpf(args.kappa_geom),
        scale_with_mrho2_flag=not args.no_scale_with_mrho2,
        pK=args.pK,
        schur_tau=mp.mpf(args.schur_tau),
        norm_space=args.norm,
        unit_calib_scheme=args.unit_calib,
        do_tighten_report=not args.no_tighten,
        show_codata_presets=not args.no_codata_presets,
        out_prefix=args.out,
        run_toy_check=args.run_toy_check,   # ← fixed
        dps_list=dps_list,
        map_c=mp.mpf(args.map_c),
        apply_map_c=args.apply_map_c,
        auto_c=args.auto_c,
    )

