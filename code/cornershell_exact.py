"""cornershell_exact.py -- RIGOROUS (SF)(a) corner-shell slope law.

Proves  min over (w in corner cone, phi) of slope(w,phi) >= CPSI = 0.023  (float min 0.0239),
which is the load-bearing constant currently HARDCODED as slope_floor=0.023917 in
sf_rowbounds_bounds.py:149 (feeds the C3<=17 relative bound and the mode-2 remainder).

slope(w,phi) = -min_lam Sum_i lam_i (gl_i . w)  over the stress polytope
                 { lam >= 0 on endpoint rows E, Sum lam = 1, Sum lam G_i = 0 (5 cols) },
where G = drift (fd, CONSTANT in phi per K1) and gl = the degree-1 in-plane part of the
constraint value field 'g' (symbolic in c,s; from SYM).  The polytope is fixed (G constant),
so its VERTICES are a fixed finite set; slope(w,phi) = max over vertices v of (-B^v(phi) . w),
B^v = Sum lam^v gl.  A rigorous lower bound is a DUAL CERTIFICATE: per (phi-cell, cone sub-arc)
exhibit a vertex v with -B^v(phi).w >= CPSI for all w in the sub-arc (linear in w -> the two
sub-arc rays) and all phi in the cell (outward-rounded interval on B^v via qinterval).

Validated to reproduce the float slope_lp (cornershell_preflight) exactly at all 20 cached configs.
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from fractions import Fraction as Fr
import sf_exact2 as SF
import qinterval as QI

c = SF._S['c']; s = SF._S['s']; xp1 = SF._S['xp1']; xp2 = SF._S['xp2']
gS = SF._S['g']; t1 = SF._S['t1']; t2 = SF._S['t2']
Z0 = {xp1: 0, xp2: 0, gS: 0, t1: 0, t2: 0}
PHI_W = math.atan(11 * math.sqrt(2) / 9)

def ep_rows(cfg):
    return [i for i, r in enumerate(SF.SYM[cfg]['tight']) if SF.is_ep(r)]

def glPi(cfg, i):
    """in-plane linear part (B_x, B_y) of the 'g' field of row i, symbolic in (c,s)."""
    g = SF.SYM[cfg]['tight'][i]['g']
    return sp.expand(sp.diff(g, xp1).subs(Z0)), sp.expand(sp.diff(g, xp2).subs(Z0))

_VCACHE = {}
_VPKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cornershell_verts.pkl')
def vertices(cfg):
    """exact stress-polytope vertices: (support, lam, (B1,B2)) with B=Sum lam glPi, symbolic in c,s.
    Enumerate <=6-row supports of E; solve_stress (Sum lam=1, Sum lam G=0 all 5); keep lam>=0.
    Loads from cornershell_verts.pkl (precomputed) if present -- the sympy solve is a one-time ~60s/cfg."""
    if cfg in _VCACHE: return _VCACHE[cfg]
    if os.path.exists(_VPKL):
        import pickle
        raw = pickle.load(open(_VPKL, 'rb'))
        if cfg in raw:
            V = []
            for (sup, lam_sr, _B1, _B2) in raw[cfg]:
                sup = tuple(sup)
                lam = [eval(x, {'__builtins__': {}}, sp.__dict__) for x in lam_sr]   # radical numbers, no symbols
                B1 = sp.expand(sum(lam[a] * glPi(cfg, sup[a])[0] for a in range(len(sup))))   # canonical (c,s)
                B2 = sp.expand(sum(lam[a] * glPi(cfg, sup[a])[1] for a in range(len(sup))))
                V.append((sup, lam, (B1, B2)))
            _VCACHE[cfg] = V
            return V
    E = ep_rows(cfg); pool, _ = SF.Gexact(cfg)
    E = [i for i in E if i in pool]           # constant-G endpoint rows only
    out = []; seen = set()
    for sz in (6, 5, 4):
        for sup in itertools.combinations(E, sz):
            st = SF.solve_stress(cfg, sup, (), 0, 'B')
            if st is None: continue
            lam = st[0]
            key = tuple(sp.srepr(l) for l in lam) + sup
            if key in seen: continue
            seen.add(key)
            B1 = sp.expand(sum(lam[a] * glPi(cfg, sup[a])[0] for a in range(len(sup))))
            B2 = sp.expand(sum(lam[a] * glPi(cfg, sup[a])[1] for a in range(len(sup))))
            out.append((sup, lam, (B1, B2)))
    _VCACHE[cfg] = out
    return out

def cone_rows(cfg):
    """(idx_in_pool, (Gx,Gy)) for constant-G endpoint rows with nonzero in-plane drift."""
    E = ep_rows(cfg); pool, GM = SF.Gexact(cfg)
    E = [i for i in E if i in pool]
    out = []
    for i in E:
        gx, gy = float(GM[i][0]), float(GM[i][1])
        if gx * gx + gy * gy > 1e-18: out.append((i, (gx, gy)))
    return out

def cone_center_hw(cfg, phi_mid):
    """corner cone {w: G_i^Pi . w >= 0} as (center_angle, half_width), wraparound-safe via the
    bisector = normalize(sum of cone-row normals).  Single ray e(phi) (hw=0) if no cone rows."""
    cr = cone_rows(cfg)
    if not cr:
        return (phi_mid, 0.0)
    sx = sum(gx for _, (gx, gy) in cr); sy = sum(gy for _, (gx, gy) in cr)
    bc = math.atan2(sy, sx)                                   # bisector direction (interior of cone)
    def angdist(a, b):
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)
    hw = math.pi / 2 - max(angdist(math.atan2(gy, gx), bc) for _, (gx, gy) in cr)
    return (bc, max(hw, 0.0))

def cone_interval(cfg, phi_mid):
    bc, hw = cone_center_hw(cfg, phi_mid)
    return (bc - hw, bc + hw)

def slope_float(cfg, phi_val, th):
    """float slope(w=e(th), phi) via max over exact vertices of -B.w (for validation only)."""
    cc, ss = math.cos(phi_val), math.sin(phi_val)
    w1, w2 = math.cos(th), math.sin(th)
    best = -1e9
    for _, _, (B1, B2) in vertices(cfg):
        v = -(float(B1.subs({c: cc, s: ss})) * w1 + float(B2.subs({c: cc, s: ss})) * w2)
        best = max(best, v)
    return best

if __name__ == '__main__':
    import pickle
    from scipy.optimize import linprog
    d = pickle.load(open('sf_cache.pkl', 'rb')) if os.path.exists('sf_cache.pkl') else pickle.load(open('../sf_cache.pkl', 'rb'))
    def lp(dat, w):
        import numpy as np
        E = [k for k in range(len(dat['drift'])) if SF.is_ep(SF.SYM[dat_cfg]['tight'][k]) and k in SF.Gexact(dat_cfg)[0]]
        G = dat['G'][E]; gl = dat['gl'][E]; n = G.shape[0]
        A = np.vstack([np.ones((1, n)), G.T]); beq = np.zeros(6); beq[0] = 1
        r = linprog(gl[:, :2] @ w, A_eq=A, b_eq=beq, bounds=[(0, None)] * n, method='highs')
        return -r.fun
    import numpy as np
    worst = 1e9; maxerr = 0.0
    for key in sorted(d.keys()):
        phi_val, dat_cfg = key; dat = d[key]
        lo, hi = cone_interval(dat_cfg, phi_val)
        ths = [lo + (hi - lo) * k / 40 for k in range(41)] if hi > lo else [lo]
        cmin = min(slope_float(dat_cfg, phi_val, th) for th in ths)
        fmin = min(lp(dat, np.array([math.cos(th), math.sin(th)])) for th in ths)
        maxerr = max(maxerr, abs(cmin - fmin)); worst = min(worst, cmin)
        print(f'  {str(key):22s} cone=[{lo:+.3f},{hi:+.3f}] exact_min={cmin:.5f} float_min={fmin:.5f} d={abs(cmin-fmin):.1e}')
    print(f'\nTRUST GATE: max|exact-float|={maxerr:.2e}  global exact min slope over cones={worst:.5f}  (target CPSI=0.023)')

# ================= RIGOROUS CONTINUUM SWEEP =================
import math as _m
CPSI = Fr(2385, 10**5)  # proven slope floor target (paper's c_psi)
PAD = Fr(1, 10**9)      # outward pad dominating libm error (~1e-16)

def cossin_iv(a, b):
    """outward-rounded (cw,sw) Fraction-intervals enclosing (cos th, sin th) for th in [a,b]."""
    cand = [a, b] + [k * _m.pi / 2 for k in range(int(_m.floor(a / (_m.pi / 2))), int(_m.ceil(b / (_m.pi / 2))) + 1) if a <= k * _m.pi / 2 <= b]
    cs = [_m.cos(x) for x in cand]; ss = [_m.sin(x) for x in cand]
    def iv(lo, hi):
        return (Fr(lo).limit_denominator(10**12) - PAD, Fr(hi).limit_denominator(10**12) + PAD)
    return iv(min(cs), max(cs)), iv(min(ss), max(ss))

def witness_bound(cfg, csbox, Bev, lo, hi, min_arc):
    """min certified lower bound of slope over cone sub-arc [lo,hi] x csbox, covering with vertices."""
    cw, sw = cossin_iv(lo, hi)
    best = None
    for (b1, b2) in Bev:                       # slope >= -(B.w) for this vertex; interval lower bd
        lb = QI.neg(QI.add(QI.mul(b1, cw), QI.mul(b2, sw)))[0]
        best = lb if best is None else max(best, lb)
    if best >= CPSI: return best
    if hi - lo <= min_arc: return best         # cannot certify this sub-arc at floor
    m = (lo + hi) / 2
    return min(witness_bound(cfg, csbox, Bev, lo, m, min_arc),
               witness_bound(cfg, csbox, Bev, m, hi, min_arc))

def cell_bound(cfg, csbox, verts, center, hw, min_arc):
    Bev = [(QI.eval_iv(B1, csbox), QI.eval_iv(B2, csbox)) for (_, _, (B1, B2)) in verts]
    if hw == 0.0:
        cw, sw = cossin_iv(center, center)
        return max(QI.neg(QI.add(QI.mul(b1, cw), QI.mul(b2, sw)))[0] for b1, b2 in Bev)
    return witness_bound(cfg, csbox, Bev, center - hw, center + hw, min_arc)

def sweep_pent(cfg, verts, center, hw):
    """sweep s=sin phi over [0, S_W); adaptive, fine near the wall S_W."""
    SW = Fr(_m.sin(PHI_W)).limit_denominator(10**12) - Fr(1, 10**7)   # stay < sin(phi_w), <1
    worst = None; ncell = 0
    def rec(s0, s1, depth):
        nonlocal worst, ncell
        ncell += 1
        cb = SF.csbox_pent(cfg, s0, s1)
        bnd = cell_bound(cfg, cb, verts, center, hw, min_arc=2e-3)
        if bnd >= CPSI:
            worst = bnd if worst is None else min(worst, bnd); return True
        if depth >= 40:
            worst = bnd if worst is None else min(worst, bnd); return False
        sm = (s0 + s1) / 2
        return rec(s0, sm, depth + 1) & rec(sm, s1, depth + 1)
    ok = rec(Fr(0), SW, 0)
    return ok, worst, ncell

def sweep_hex(cfg, verts, center, hw):
    CW = Fr(_m.cos(PHI_W)).limit_denominator(10**12) - Fr(1, 10**7)
    worst = None; ncell = 0
    def rec(c0, c1, depth):
        nonlocal worst, ncell
        ncell += 1
        cb = SF.csbox_hex(c0, c1)
        bnd = cell_bound(cfg, cb, verts, center, hw, min_arc=2e-3)
        if bnd >= CPSI:
            worst = bnd if worst is None else min(worst, bnd); return True
        if depth >= 40:
            worst = bnd if worst is None else min(worst, bnd); return False
        cm = (c0 + c1) / 2
        return rec(c0, cm, depth + 1) & rec(cm, c1, depth + 1)
    ok = rec(-CW, CW, 0)
    return ok, worst, ncell

def run_sweep():
    res = {}
    for cfg in ('pentA', 'hex1', 'pentB'):
        verts = vertices(cfg)
        center, hw = cone_center_hw(cfg, PHI_W if cfg != 'hex1' else _m.pi / 2)
        if cfg == 'hex1':
            ok, worst, n = sweep_hex(cfg, verts, center, hw)
        else:
            ok, worst, n = sweep_pent(cfg, verts, center, hw)
        res[cfg] = (ok, worst, n)
        print(f'{cfg}: {"PROVED" if ok else "FAILED"}  min certified slope={float(worst):.6f}  cells={n}  vertices={len(verts)}')
    allok = all(r[0] for r in res.values())
    gmin = min(float(r[1]) for r in res.values())
    print(f'\nCORNER-SHELL LAW: {"PROVED slope>=%.4f over all (w in cone, phi)"%float(CPSI) if allok else "INCOMPLETE"}  |  global min certified={gmin:.6f}')
    return res
