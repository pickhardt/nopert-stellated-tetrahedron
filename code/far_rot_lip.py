"""Tight dir-Lipschitz of G_lambda via affine arithmetic  (fixes the ~14x-loose R3*s bound).

certify() bounds the rotation-cell variation of G_lambda by  rot = R3*s*(dtw+dpw) + R3*sw
with R3=sqrt3 -- a WORST-CASE-over-all-geometry bound that ignores the actual edge normals and
lam weights, measured ~10-17x loose at large s (where 96% of the far-manifest leaves live).

Here we compute the ACTUAL  Lambda_dt, Lambda_dp = sup_box |dG_lambda/d(dt)|,|dG_lambda/d(dp)|
by the same affine-arithmetic forward-mode AD that g_deriv_aa uses for (theta,phi): the rotation
axis  dir = (sin dt cos dp, sin dt sin dp, cos dt)  has the SAME form as u2, so we seed the two AA
noise symbols on (dt,dp), build  R(dt,dp) = I + sin(s0)[dir]x + (1-cos s0)[dir]x^2  as a DA-matrix
(s at center; the s-variation stays in certify's tight R3*sw term), and read dG/d(dt),dG/d(dp).

SOUNDNESS.  u2 (theta,phi) is carried as an OUTWARD INTERVAL over its whole cell (DA.const(FI)),
so the derivatives dG/d(dt),dG/d(dp) are enclosed for EVERY u2 in the cell -- exactly the telescoping
requirement (g_deriv_aa module doc): the u2-variation is bounded separately by glam_lip, and this
bounds the rotation-variation for u2 ranging.  Every op rounds outward; Lambda = max(|.lo|,|.hi|) of
the derivative's interval concretization is a proven upper bound on sup_box|dG/dparam|.
"""
import math
import fast_interval as F
from fast_interval import FI
import g_deriv_aa as GA
from g_deriv_aa import AA, DA

def _dir_aa(dt, dtw, dp, dpw):
    """dir(dt,dp) and its (dt,dp) partials as DA -- identical shape to g_deriv_aa.u2_aa."""
    DT = AA(dt, dtw, 0.0, 0.0); DP = AA(dp, 0.0, dpw, 0.0)
    ct = DT.cos(); st = DT.sin(); cp = DP.cos(); sp = DP.sin(); Z = AA.const(0.0)
    return [DA(st * cp, ct * cp, Z - st * sp),
            DA(st * sp, ct * sp, st * cp),
            DA(ct, Z - st, Z)]

def _skew(d):
    Z = DA.const(0.0)
    return [[Z, -d[2], d[1]], [d[2], Z, -d[0]], [-d[1], d[0], Z]]

def _matmul(A, B):
    return [[GA.dvdot(A[i], [B[0][j], B[1][j], B[2][j]]) for j in range(3)] for i in range(3)]

def _rot_da(dt, dtw, dp, dpw, s0):
    """R(dt,dp) = I + sin(s0)[dir]x + (1-cos s0)[dir]x^2 as a 3x3 DA-matrix (s at center)."""
    d = _dir_aa(dt, dtw, dp, dpw)
    K = _skew(d); K2 = _matmul(K, K)
    cs, sn = F.trig_fi(s0, s0)                     # trig_fi returns (cos, sin) as point-intervals
    a = DA.const(sn); b = DA.const(FI(1.0) - cs)   # sin(s0),  1 - cos(s0)
    I = [[DA.const(1.0), DA.const(0.0), DA.const(0.0)],
         [DA.const(0.0), DA.const(1.0), DA.const(0.0)],
         [DA.const(0.0), DA.const(0.0), DA.const(1.0)]]
    return [[I[i][j] + a * K[i][j] + b * K2[i][j] for j in range(3)] for i in range(3)]

def _u2_cell_edges(th, thw, ph, phw, eo, aa):
    """edge normals n_k and offsets ci_k over the WHOLE u2-cell, as FI (outward interval)."""
    u2 = F.iu2(th - thw, th + thw, ph - phw, ph + phw)
    W = F.frame(u2); V = F.verts(aa)
    wv = [F.applyW(W, v) for v in V]               # 3D framed vertices (interval)
    qo = [F.proj(w) for w in wv]                   # projected hull (interval)
    edges = []
    for (p_, r_, sgn) in eo:
        d = [qo[r_][0] - qo[p_][0], qo[r_][1] - qo[p_][1]]; n = [FI(0.0) - d[1], d[0]]
        if sgn < 0: n = [-n[0], -n[1]]
        nn = F.vnorm(n); n = [x / nn for x in n]
        ci = None
        for w in qo:
            val = F.dot(n, w)
            ci = val if ci is None else FI(min(ci.lo, val.lo), min(ci.hi, val.hi))
        edges.append((n, ci))
    return edges, wv

def _skew_fi(d):
    Z = FI(0.0); return [[Z, -d[2], d[1]], [d[2], Z, -d[0]], [-d[1], d[0], Z]]
def _matmul_fi(A, B):
    return [[F.dot(A[i], [B[0][j], B[1][j], B[2][j]]) for j in range(3)] for i in range(3)]

def glam_lip_s(box, eo, lam, aa):
    """Lambda_s = proven sup over the box of |dG_lambda/ds|.  Same rotation R=I+sin s[dir]x+(1-cos s)[dir]x^2
    as glam_lip_dir, differentiated w.r.t. s (dir CONSTANT); dir carried as an INTERVAL over its (dt,dp)-cell
    and u2 as an interval, so the bound holds for all (u2,dir) in the box (telescoping soundness)."""
    th, thw, ph, phw, dt, dtw, dp, dpw, s0, sw = box
    edges, wv = _u2_cell_edges(th, thw, ph, phw, eo, aa)
    dirI = F.idir(dt - dtw, dt + dtw, dp - dpw, dp + dpw)   # dir over its whole cell (interval)
    K = _skew_fi(dirI); K2 = _matmul_fi(K, K)
    S = AA(s0, sw, 0.0, 0.0)
    a = DA(S.sin(), S.cos(), AA.const(0.0))                 # sin(s),  d/ds = cos(s)
    b = DA(AA.const(1.0) - S.cos(), S.sin(), AA.const(0.0)) # 1-cos(s), d/ds = sin(s)
    Z = DA.const(0.0)
    I = [[DA.const(1.0), Z, Z], [Z, DA.const(1.0), Z], [Z, Z, DA.const(1.0)]]
    R = [[I[i][j] + a * DA.const(K[i][j]) + b * DA.const(K2[i][j]) for j in range(3)] for i in range(3)]
    G = DA.const(0.0)
    for k, w in lam:
        ei, vj = k // 8, k % 8; n, ci = edges[ei]
        wvj = [DA.const(wv[vj][t]) for t in range(3)]
        rw = [GA.dvdot(R[t], wvj) for t in range(3)]
        pw = [rw[1], DA.const(0.0) - rw[0]]
        nD = [DA.const(n[0]), DA.const(n[1])]
        gc = DA.const(ci) - GA.dvdot(nD, pw)
        G = G + DA.const(float(w)) * gc
    ds = G.dth.to_FI()                                      # slot 1 carries the s-derivative
    return max(abs(ds.lo), abs(ds.hi))

def glam_lip_dir(box, eo, lam, aa):
    """Lambda_dt, Lambda_dp = proven sup over the box of |dG_lambda/d(dt)|, |dG_lambda/d(dp)|."""
    th, thw, ph, phw, dt, dtw, dp, dpw, s0, sw = box
    edges, wv = _u2_cell_edges(th, thw, ph, phw, eo, aa)
    R = _rot_da(dt, dtw, dp, dpw, s0)
    # rotate each framed vertex: wc[k] = R * wv[k]  (R is DA in dt,dp; wv[k] is interval -> DA.const)
    G = DA.const(0.0)
    for k, w in lam:
        ei, vj = k // 8, k % 8; n, ci = edges[ei]
        wvj = [DA.const(wv[vj][t]) for t in range(3)]
        rw = [GA.dvdot(R[t], wvj) for t in range(3)]      # R * (W v)  -- DA vector in (dt,dp)
        pw = [rw[1], DA.const(0.0) - rw[0]]               # proj
        nD = [DA.const(n[0]), DA.const(n[1])]
        gc = DA.const(ci) - GA.dvdot(nD, pw)
        G = G + DA.const(float(w)) * gc
    ddt = G.dth.to_FI(); ddp = G.dph.to_FI()
    return max(abs(ddt.lo), abs(ddt.hi)), max(abs(ddp.lo), abs(ddp.hi))

if __name__ == '__main__':
    import ball_cover as BC, ball_cover5 as B5
    import numpy as np
    print("=== validate dir-Lipschitz vs finite differences + vs R3*s bound ===")
    th, ph = 1.60, 0.35; u2c = BC.u2center(th, ph); eo = BC.edge_order(u2c)
    aa = B5.aa
    for (dt, dp, s) in [(math.pi*0.5, 2*math.pi*0.4, 2.8), (math.pi*0.4, 1.0, 2.0),
                        (math.pi*0.6, 3.0, 1.0), (math.pi*0.5, 0.5, 0.3)]:
        mu, lam = BC.lam_lp(u2c, s*np.array(BC.dircenter(dt, dp)))
        if lam is None: print(f'  (dt={dt:.2f},s={s}) no lam'); continue
        dtw = dpw = 1e-4
        box = (th, 1e-9, ph, 1e-9, dt, dtw, dp, dpw, s, 1e-9)
        Ldt, Ldp = glam_lip_dir(box, eo, lam, aa)
        # finite-diff the true |dG/d(dt)| at center
        def Gpt(dt_, dp_): return B5.Gpoint(th, ph, dt_, dp_, s, eo, lam).lo
        e = 1e-6
        fd_dt = abs(Gpt(dt+e, dp) - Gpt(dt-e, dp)) / (2*e)
        fd_dp = abs(Gpt(dt, dp+e) - Gpt(dt, dp-e)) / (2*e)
        crude = B5.R3 * s
        print(f'  s={s}: Ldt(AA)={Ldt:.4f} fd={fd_dt:.4f} | Ldp(AA)={Ldp:.4f} fd={fd_dp:.4f} '
              f'| R3*s={crude:.3f}  tighten={crude/max(Ldt,Ldp,1e-9):.1f}x  sound={Ldt>=fd_dt-1e-6 and Ldp>=fd_dp-1e-6}')
