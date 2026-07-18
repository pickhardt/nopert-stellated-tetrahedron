"""Tight RIGOROUS interval n×n determinant (n=4,5), replacing naive-Laplace det whose dependency
blowup (measured up to ~2500x on narrow boxes; 96x |det| in the SB-box normals) craters the SB-box
facet-margin lower bound AND spuriously trips the facet-stability guard at higher delta.

Method (multilinear ROW expansion). M = M0 + E, rows m_i = m0_i + e_i, |e_ij| <= r_ij.
  det(M) = det(M0)                                   [S=empty, enclosed by point-interval arithmetic]
         + sum_i sum_j e_ij * Cof_ij(M0)             [|S|=1, EXACT first order]
         + R                                          [|S|>=2, Hadamard-bounded remainder]
First-order interval = +/- sum_ij |Cof_ij(M0)| r_ij.  Remainder |R| <=
  prod_i(a_i+b_i) - prod_i a_i - sum_i b_i prod_{j!=i} a_j,   a_i=||m0_i||, b_i=||r_i|| (Euclidean),
which Hadamard-bounds every |S|>=2 term (|det of rows| <= prod ||row||). Radius rounded OUTWARD;
det(M0) and cofactors enclosed by point-interval Laplace (only rounding width). Sound: encloses
det(M) for every M in the box.

Uses:
- det4_tight: SB-box facet normal a_k (4x4 minors of the 4 facet difference vectors).
- det5_tight: facet-STABILITY test s = a·(r_P - r0) = det[diff_1..4, r_P-r0] (avoids a-then-dot blowup)."""
import math
import fast_interval as F
FI = F.FI

def _det_pt(mp):
    """Rigorous interval determinant of a POINT float matrix (n<=4) via Laplace. Point inputs =>
    no dependency blowup, only rounding width."""
    n = len(mp)
    if n == 1:
        return FI(mp[0][0])
    if n == 2:
        return FI(mp[0][0]) * FI(mp[1][1]) - FI(mp[0][1]) * FI(mp[1][0])
    s = FI(0.0)
    for j in range(n):
        sub = [[mp[a][b] for b in range(n) if b != j] for a in range(1, n)]
        t = FI(mp[0][j]) * _det_pt(sub)
        s = s + (t if j % 2 == 0 else FI(0.0) - t)
    return s

def _cof(mp, i, j):
    n = len(mp)
    sub = [[mp[a][b] for b in range(n) if b != j] for a in range(n) if a != i]
    d = _det_pt(sub)
    return d if (i + j) % 2 == 0 else (FI(0.0) - d)

def det_tight(M):
    """M: n×n list-of-lists of FI (n in {2,3,4,5}). Tight rigorous FI enclosure of det over the box."""
    n = len(M)
    mp = [[(M[i][j].lo + M[i][j].hi) * 0.5 for j in range(n)] for i in range(n)]
    r  = [[max(0.0, (M[i][j].hi - M[i][j].lo) * 0.5) for j in range(n)] for i in range(n)]
    Cof = [[_cof(mp, i, j) for j in range(n)] for i in range(n)]
    det0 = FI(0.0)
    for j in range(n):
        det0 = det0 + FI(mp[0][j]) * Cof[0][j]
    # first-order radius
    fo = 0.0
    for i in range(n):
        for j in range(n):
            cmax = max(abs(Cof[i][j].lo), abs(Cof[i][j].hi))
            fo = F.ru(fo + F.ru(cmax * r[i][j]))
    # Hadamard remainder for |S|>=2
    a = [F.ru(math.sqrt(sum(mp[i][j] * mp[i][j] for j in range(n)))) for i in range(n)]
    b = [F.ru(math.sqrt(sum(r[i][j] * r[i][j] for j in range(n)))) for i in range(n)]
    prodab = 1.0
    for i in range(n):
        prodab = F.ru(prodab * F.ru(a[i] + b[i]))
    proda = 1.0
    for i in range(n):
        proda = F.rd(proda * a[i])
    slin = 0.0
    for i in range(n):
        p = 1.0
        for j in range(n):
            if j != i:
                p = F.rd(p * a[j])
        slin = F.rd(slin + F.rd(b[i] * p))
    rem = prodab - proda - slin
    if rem < 0.0:
        rem = 0.0
    rad = F.ru(fo + rem)
    return FI(F.rd(det0.lo - rad), F.ru(det0.hi + rad))

def det4_tight(M):
    return det_tight(M)

def det5_tight(M):
    return det_tight(M)

# ---- self-test: enclosure soundness + tightness vs naive Laplace, narrow boxes, n=4 and n=5 ----
def _naive_det(M):
    n = len(M)
    if n == 1: return M[0][0]
    s = FI(0.0)
    for j in range(n):
        sub = [[M[a][b] for b in range(n) if b != j] for a in range(1, n)]
        t = M[0][j] * _naive_det(sub); s = s + (t if j % 2 == 0 else FI(0.0) - t)
    return s

if __name__ == '__main__':
    import numpy as np
    def mk(n, seed, wid):
        vals = [((seed * 2654435761 + k * 40503) % 1000) / 500.0 - 1.0 for k in range(n * n)]
        M = [[FI(F.rd(vals[n*i+j] - wid), F.ru(vals[n*i+j] + wid)) for j in range(n)] for i in range(n)]
        return M, np.array([[vals[n*i+j] for j in range(n)] for i in range(n)])
    for n in (4, 5):
        worst_ratio = 0.0; nbad = 0; nsamp = 0
        for seed in range(1, 50):
            for wid in (1e-4, 5e-4, 2e-3):
                M, M0 = mk(n, seed, wid)
                dt = det_tight(M); dn = _naive_det(M)
                for s2 in range(30):
                    E = np.array([[((seed*13+s2*7+n*i+j) % 100)/100.0*2-1 for j in range(n)] for i in range(n)]) * wid
                    dv = float(np.linalg.det(M0 + E)); nsamp += 1
                    if not (dt.lo - 1e-12 <= dv <= dt.hi + 1e-12): nbad += 1
                wt = dt.hi - dt.lo; wn = dn.hi - dn.lo
                if wt > 0: worst_ratio = max(worst_ratio, wn / wt)
        print(f"n={n}: {nsamp} samples, enclosure violations={nbad} (must be 0); naive up to {worst_ratio:.0f}x wider than tight")
