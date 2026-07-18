"""Rigorous per-row order-4 remainder constant C4 for the (SF) certificate.

C4 := sup_{delta in (0,rho0], xhat in Vbox, phi in chart, rows} |h(delta)-P3(delta)|/delta^4,
where h is the TRUE normalized SF constraint (full rotation exponentials, true unit
normal) and P3 its exact degree-3 delta-Taylor part.  Since h(delta)=sum_{k>=1} c_k delta^k
with the exact c_1,c_2,c_3 folded into P3,
    |h-P3|/delta^4 = |sum_{k>=4} c_k delta^{k-4}| <= |c_4| + rho0|c_5| + ... .
We enclose c_4..c_8 in outward-rounded interval arithmetic (matrix-power delta-series,
so NO sin/cos/theta/division of the axis is needed -- exp(delta B)=sum delta^k B^k/k!),
and bound the k>=9 tail by the crude-but-rigorous majorant of c4_lemma (times rho0^5,
utterly negligible).  Sweeps phi and the xhat-box with subdivision; certifies C4 < TARGET.

The delta-coefficients c_0..c_8 of every intermediate are EXACT interval enclosures
(the matrix series is truncated at order 8, but products only read coeffs <= 8, so
c_0..c_8 of h are exact); only the >=9 tail is majorized.
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_interval_pure import FI, rd, ru
from fractions import Fraction as Fr

K = 9                      # Taylor order carried (exact for c_0..c_9, or c_0..c_8 after one delta-shift)
RHO0 = 1e-3
a_st = Fr(11, 20)

# ---- exact rational frame vertices wh = (Wst @ V^T)^T, Wst has 1/sqrt2 rows ----
_vs = [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]
_V = [tuple(Fr(x) for x in v) for v in _vs] + [tuple(-a_st*Fr(x) for x in v) for v in _vs]
names = ['v1','v2','v3','v4','p1','p2','p3','p4']
CONFIGS = {'pentA':['v1','v4','p1','v2','p4'],
           'hex1' :['v1','v4','p1','v3','v2','p4'],
           'pentB':['v1','v4','p1','v3','p4']}
_INV_S2 = FI(rd(1.0/math.sqrt(2.0)), ru(1.0/math.sqrt(2.0)))   # 1/sqrt2 enclosure
def _wh(j):
    x,y,z = _V[j]
    c0 = FI(float(z))
    c1 = _INV_S2 * FI(float(-x-y))
    c2 = _INV_S2 * FI(float(x-y))
    return [c0, c1, c2]
WH = [_wh(j) for j in range(8)]

FI0 = FI(0.0); FI1 = FI(1.0)
def iabs(a): return max(abs(a.lo), abs(a.hi))

# ---------------- Taylor series (list of K+1 FI, delta-coeffs) ----------------
def ts_zero(): return [FI0]*(K+1)
def ts_const(fi):
    r = [FI0]*(K+1); r[0] = fi; return r
def ts_add(a,b): return [a[i]+b[i] for i in range(K+1)]
def ts_sub(a,b): return [a[i]-b[i] for i in range(K+1)]
def ts_scale(a, fi): return [a[i]*fi for i in range(K+1)]
def ts_mul(a,b):
    c = [FI0]*(K+1)
    for i in range(K+1):
        ai = a[i]
        if ai.lo == 0.0 and ai.hi == 0.0: continue
        for j in range(K+1-i):
            c[i+j] = c[i+j] + ai*b[j]
    return c

# ---------------- matrices (3x3 of FI) and their power-series ----------------
def matmul(A,B):
    return [[A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j] for j in range(3)] for i in range(3)]
def matvec(A,v):
    return [A[i][0]*v[0]+A[i][1]*v[1]+A[i][2]*v[2] for i in range(3)]

_FACT = [FI(float(math.factorial(k))) for k in range(K+1)]
def exp_series(B):
    """returns 3x3 matrix, each entry a TS: coeff_k = (B^k/k!)[i][j]."""
    P = [[FI1 if i==j else FI0 for j in range(3)] for i in range(3)]   # B^0
    coeffs = [P]
    for k in range(1, K+1):
        P = matmul(P, B); coeffs.append(P)
    M = [[[FI0]*(K+1) for _ in range(3)] for _ in range(3)]
    for k in range(K+1):
        invk = FI1/_FACT[k]
        for i in range(3):
            for j in range(3):
                M[i][j][k] = coeffs[k][i][j]*invk
    return M
def mat_ts_mul(A,B):
    return [[ ts_add(ts_add(ts_mul(A[i][0],B[0][j]), ts_mul(A[i][1],B[1][j])), ts_mul(A[i][2],B[2][j]))
              for j in range(3)] for i in range(3)]
def mat_ts_vec(A, wh):
    return [ ts_add(ts_add(ts_scale(A[i][0],wh[0]), ts_scale(A[i][1],wh[1])), ts_scale(A[i][2],wh[2]))
             for i in range(3)]

# ---------------- 1/sqrt(G) as TS via binomial in w=(G-G0)/G0 ----------------
_BIN = [Fr(1)]                            # binom(-1/2, m), m=0,1,2,...
def _binom_half(m):
    while len(_BIN) <= m:
        n = len(_BIN)
        _BIN.append(_BIN[-1]*(Fr(-1,2)-(n-1))/n)
    return _BIN[m]
def _fi_of(cf):
    v = cf.numerator/cf.denominator       # nearest float; +-1 ulp encloses cf
    return FI(rd(v), ru(v))
def ts_invsqrt(G):
    G0 = G[0]
    invG0 = FI1/G0
    w = [FI0]*(K+1)
    for k in range(1,K+1): w[k] = G[k]*invG0
    res = ts_const(FI1); wp = ts_const(FI1)
    for m in range(1,K+1):
        wp = ts_mul(wp, w)
        res = ts_add(res, ts_scale(wp, _fi_of(_binom_half(m))))
    inv_sqrt_G0 = (FI1/G0).sqrt()          # G0^{-1/2}
    return ts_scale(res, inv_sqrt_G0)

# ---------------- build h Taylor coeffs for a cell ----------------
def _delta_shift(a):
    """ts for a(delta)/delta given a[0]==0 (enclosing): shift coeffs down one."""
    return a[1:] + [FI0]

def cell_c4(cfg, cbox, xbox, T_tail, labels):
    """cbox=(clo,chi); xbox=[(lo,hi)]*5 for xhat; labels=set of (edge,vertex) tight rows.
    Returns max over the tight rows of |c4|+rho0|c5|+...+rho0^4|c8|+T_tail."""
    clo, chi = cbox
    c = FI(clo, chi)
    s = (FI1 - c*c).sqrt()                       # s = sqrt(1-c^2) >= 0 on all charts
    x = [FI(lo,hi) for (lo,hi) in xbox]
    # B_M = -[ah]_x, ah=(-s,c,0): [ah]_x=[[0,0,c],[0,0,s],[-c,-s,0]]
    negc, negs = FI0-c, FI0-s
    BM = [[FI0,FI0,negc],[FI0,FI0,negs],[c,s,FI0]]
    # B_R = [v]_x, v=(-x2,x1,x3): [[0,-x3,x1],[x3,0,x2],[-x1,-x2,0]]
    BR = [[FI0, FI0-x[2], x[0]],[x[2], FI0, x[1]],[FI0-x[0], FI0-x[1], FI0]]
    M = exp_series(BM); R = exp_series(BR)
    RM = mat_ts_mul(R, M)
    dts = [FI0]*(K+1); dts[1] = FI1         # delta
    worst = 0.0
    hull = CONFIGS[cfg]; idx = [names.index(nm) for nm in hull]
    outer = [mat_ts_vec(M, WH[j]) for j in range(8)]
    inner = [mat_ts_vec(RM, WH[j]) for j in range(8)]
    for j in range(8):
        inner[j][0] = ts_add(inner[j][0], ts_scale(dts, x[3]))
        inner[j][1] = ts_add(inner[j][1], ts_scale(dts, x[4]))
    for k in range(len(idx)):
        a_i, b_i = idx[k], idx[(k+1)%len(idx)]
        elabel = f"{hull[k]}-{hull[(k+1)%len(idx)]}"
        if not any((elabel, names[j]) in labels for j in range(8)):
            continue                                    # edge carries no tight row
        qa0, qa1 = outer[a_i][0], outer[a_i][1]
        qb0, qb1 = outer[b_i][0], outer[b_i][1]
        dq0 = ts_sub(qb0, qa0); dq1 = ts_sub(qb1, qa1)
        nt0 = ts_scale(dq1, FI(-1.0)); nt1 = dq0        # J2(qb-qa)=(-dq1,dq0)
        G = ts_add(ts_mul(nt0,nt0), ts_mul(nt1,nt1))
        if G[0].lo <= 0.0:
            # doubled-vertex short edge: nt(0)=0, factor nt=delta*ntilde (nh=ntilde/|ntilde|
            # is still analytic).  Shift both components down one order.
            nt0, nt1 = _delta_shift(nt0), _delta_shift(nt1)
            G = ts_add(ts_mul(nt0,nt0), ts_mul(nt1,nt1))
            if G[0].lo <= 0.0: return None              # vanishes to higher order; subdivide
        isq = ts_invsqrt(G)
        nh0 = ts_mul(nt0, isq); nh1 = ts_mul(nt1, isq)
        for j in range(8):
            if (elabel, names[j]) not in labels: continue
            dx0 = ts_sub(inner[j][0], qa0); dx1 = ts_sub(inner[j][1], qa1)
            h = ts_add(ts_mul(nh0,dx0), ts_mul(nh1,dx1))
            b = (iabs(h[4]) + RHO0*iabs(h[5]) + RHO0**2*iabs(h[6])
                 + RHO0**3*iabs(h[7]) + RHO0**4*iabs(h[8]) + T_tail)
            if b > worst: worst = b
    return worst

# ---------------- adaptive sweep ----------------
def _cos_box(a, b):
    """outward cos([a,b]) for [a,b] in [0,pi] (cos decreasing)."""
    import math as _m
    return (rd(_m.cos(b)) - 1e-12, ru(_m.cos(a)) + 1e-12)

def sweep(cfg, phi_lo, phi_hi, labels, T_tail=1e-5, accept=55.0, maxdepth=48):
    """adaptive: subdivide widest of (phi,x1,x2,x3) until cell bound<accept or maxdepth.
    Returns (certified_max, n_cells, n_maxdepth)."""
    X4 = (-0.0285, 0.0285)
    full = (phi_lo, phi_hi, -2.39, 2.39, -2.39, 2.39, -0.0285, 0.0285)
    w0 = [phi_hi-phi_lo, 4.78, 4.78, 0.057]
    stack = [(full, 0)]
    gmax = 0.0; ncell = 0; nmax = 0
    while stack:
        (pl,ph,a1,b1,a2,b2,a3,b3), d = stack.pop()
        ncell += 1
        cbox = _cos_box(pl, ph)
        xbox = [(a1,b1),(a2,b2),(a3,b3),X4,X4]
        b = cell_c4(cfg, cbox, xbox, T_tail, labels)
        if b is None:
            # degenerate enclosure: force subdivide (unless too deep)
            b = accept + 1.0
        if b < accept:
            if b > gmax: gmax = b
            continue
        if d >= maxdepth:
            if b > gmax: gmax = b
            nmax += 1
            continue
        # subdivide widest (normalized) dimension
        wid = [(ph-pl)/w0[0], (b1-a1)/w0[1], (b2-a2)/w0[2], (b3-a3)/w0[3]]
        dim = wid.index(max(wid))
        if dim == 0:
            m=(pl+ph)/2; stack += [((pl,m,a1,b1,a2,b2,a3,b3),d+1),((m,ph,a1,b1,a2,b2,a3,b3),d+1)]
        elif dim == 1:
            m=(a1+b1)/2; stack += [((pl,ph,a1,m,a2,b2,a3,b3),d+1),((pl,ph,m,b1,a2,b2,a3,b3),d+1)]
        elif dim == 2:
            m=(a2+b2)/2; stack += [((pl,ph,a1,b1,a2,m,a3,b3),d+1),((pl,ph,a1,b1,m,b2,a3,b3),d+1)]
        else:
            m=(a3+b3)/2; stack += [((pl,ph,a1,b1,a2,b2,a3,m),d+1),((pl,ph,a1,b1,a2,b2,m,b3),d+1)]
    return gmax, ncell, nmax

if __name__ == '__main__':
    import sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sf_preflight as SF
    PHI_W = math.atan(11*math.sqrt(2)/9)
    ranges = {'pentA': (1e-4, PHI_W-1e-4), 'hex1': (PHI_W+1e-4, math.pi-PHI_W-1e-4),
              'pentB': (math.pi-PHI_W+1e-4, math.pi-1e-4)}
    LAB = {cfg: set(tuple(l.split('|')) for l in SF.EV[cfg].labels) for cfg in ranges}
    which = sys.argv[1:] or list(ranges)
    acc = 55.0
    gmax = 0.0
    for cfg in which:
        lo, hi = ranges[cfg]
        t0 = time.time()
        m, nc, nmx = sweep(cfg, lo, hi, LAB[cfg], accept=acc)
        gmax = max(gmax, m)
        print(f"{cfg}: C4 <= {m:.3f}   cells={nc} maxdepth-hit={nmx}  {time.time()-t0:.1f}s", flush=True)
    print(f"\nOVERALL per-row C4 <= {gmax:.3f}  (accept={acc}); k>=9 tail folded in (1e-5)")
