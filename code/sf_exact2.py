"""sf_exact2 -- exact-verify layer for the (SF)(c) witness sweep.

Builds on the validated sf_exact.py core (round 65) but generalizes it in three ways
required for a SOUND full sweep (all discovered/verified this round):

 (K1) CONSTANT STRESSES.  For every chart cfg, every tight row EXCEPT hex1 rows {7,8}
      (the short-hex-edge flags v3-v2|v2, v3-v2|v3) has a CONSTANT f-gradient G:
      the five fd[unit] coefficients are numbers in Q(sqrt2, sqrt323, sqrt1123),
      independent of (c,s).  Machine-verified in selfcheck().  Hence a stress
      lam >= 0 with Sum lam = 1, Sum lam G_Pi = 0 (+ optional transverse equalities)
      can be solved ONCE per support, exactly over the radical field, and the
      f-side identity  Sum lam f = -v . tau  (v = -Sum lam G_T, constant) holds
      identically at every (c,s) in the chart -- there is NO f-side residual.
      (A fixed rational lam with only a center-phi float LP residual would be
      UNSOUND: the requirement is |Sum lam G_Pi| <~ 1.2e-7 over the cell.)

 (K2) PER-CELL (c,s) BOXES.  All brackets take an explicit csbox enclosing the
      phi-cell arc (outward-rounded via qinterval.sqrt_iv), not the whole chart.

 (K3) W3 RADIAL SECTORS at the two marginal corners P0 in {(0,0), (2c,2s)}.
      For a stress lam as in (K1) restricted to
        corner 0  : ep rows          (CS-1:    f,g,h3 constant terms vanish identically)
        corner 2e : Z rows           (CS-1-far: f,g,h3(tau=0) vanish at (2c,2s) identically;
                                      Z = ep rows minus the two transitioning-vertex flags)
      the tau=0 radial section is EXACTLY  Sum lam g(P0 + t e) = t*A(e) + t^2*B(e),
      certified per sector (rational tan-half-angle parameter w) and radial band [t0,t1].

Soundness of every accepted witness reduces to: outward-rounded exact interval
arithmetic (qinterval, all-Fraction), the proven constants LHESS=(1+sqrt3)/4 (Lemma RB-1),
C2<=15.57, the proven kill calculus (W-A / W-B / W-W2 / W-W3 as documented per function),
and the STILL-ASSUMED order-4 constant C4<=60 (exact proof pending; every manifest row
records this assumption).
"""
import os, sys, pickle, itertools
from fractions import Fraction as Fr
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qinterval as QI

_D = os.path.dirname(os.path.abspath(__file__))
SYM = pickle.load(open(os.path.join(_D, 'sf_rowbounds_sym.pkl'), 'rb'))

# canonical symbols (must be the pickled ones -- they carry assumptions)
_all = set()
for _cfg in SYM:
    for _r in SYM[_cfg]['tight']:
        _all |= _r['g'].free_symbols | _r['f'].free_symbols | _r['h3'].free_symbols
_S = {str(x): x for x in _all}
c = _S['c']; s = _S['s']; xp1 = _S['xp1']; xp2 = _S['xp2']
g = _S['g']; t1 = _S['t1']; t2 = _S['t2']
XS = (xp1, xp2, g, t1, t2)
TAUS = (g, t1, t2)
w = sp.Symbol('w', real=True)          # tan-half-angle sector parameter
t = sp.Symbol('t', positive=True)      # radial parameter

RHO0 = Fr(1, 1000)      # delta cap of Lemma 6'
C4   = Fr(60)           # order-4 remainder constant (PROVED <= 58.36 r69; exact proof pending)
BT   = Fr(285, 10000)   # transverse half-box (Thm 14.3)
RB   = Fr(1, 50)        # inner corner cutoff (r71: SB-box BP=9/500, reach BP/M(0.1) >= 0.0201141 > RB=0.02)
TLAST = Fr(1, 10)       # outer radius of the W3 annulus
LHI = QI.sqrt_iv(Fr(3), Fr(3))[1]
LHESS = (Fr(1) + LHI) / 4              # exact upper bound of (1+sqrt3)/4  (Lemma RB-1)

Z5 = (0, 0, 0, 0, 0)
UNITS = [tuple(1 if q == i else 0 for q in range(5)) for i in range(5)]
_TP = [(Fr(0), Fr(0), Fr(0))] + [tuple(Fr(z) * BT for z in p)
                                 for p in itertools.product([-1, 1], repeat=3)]

def tau_patterns(xbox):
    """corner patterns of the tau sub-box in xbox (g,t1,t2 intervals), plus 0 if inside."""
    ivs = [xbox[g], xbox[t1], xbox[t2]]
    pats = list(itertools.product(*[(iv[0], iv[1]) for iv in ivs]))
    if all(iv[0] <= 0 <= iv[1] for iv in ivs):
        pats.append((Fr(0), Fr(0), Fr(0)))
    return pats

# ------------------------------------------------------------------ row structure
def is_ep(row):
    a, b = row['edge'].split('-')
    return row['j'] in (a, b)

def ep_rows(cfg):
    return [i for i, r in enumerate(SYM[cfg]['tight']) if is_ep(r)]

# stress pool: ep rows with CONSTANT G (excludes hex1 short-edge flags 7,8)
def stress_pool(cfg):
    out = []
    for i in ep_rows(cfg):
        r = SYM[cfg]['tight'][i]
        if all(not sp.sympify(r['fd'].get(m, 0)).free_symbols for m in UNITS):
            out.append(i)
    return out

_G_CACHE = {}
def Gexact(cfg):
    """exact constant G matrix (rows = stress_pool order), sympy numbers."""
    if cfg in _G_CACHE: return _G_CACHE[cfg]
    pool = stress_pool(cfg)
    rows = SYM[cfg]['tight']
    M = {i: [sp.nsimplify(sp.sympify(rows[i]['fd'].get(m, 0))) for m in UNITS] for i in pool}
    _G_CACHE[cfg] = (pool, M)
    return _G_CACHE[cfg]

def z_rows(cfg):
    """Z = stress-pool rows with f,g,h3(tau=0) vanishing identically at (2c,2s) (CS-1-far).
    Verified symbolically here (cached)."""
    key = ('Z', cfg)
    if key in _G_CACHE: return _G_CACHE[key]
    pool, _ = Gexact(cfg)
    rows = SYM[cfg]['tight']
    out = []
    for i in pool:
        r = rows[i]
        sub = {xp1: 2*c, xp2: 2*s, g: 0, t1: 0, t2: 0}
        ok = True
        for fld in ('f', 'g', 'h3'):
            e = sp.expand(r[fld].subs(sub))
            if e != 0:
                e = sp.simplify(sp.expand(e).subs(s**2, 1 - c**2))
            if e != 0:
                ok = False; break
        if ok: out.append(i)
    _G_CACHE[key] = out
    return out

# ------------------------------------------------------------------ (c,s) boxes
def csbox_pent(cfg, s0, s1):
    """pentA/pentB phi-cell given by exact s-interval [s0,s1] (Fractions, 0<=s0<s1<1).
    Returns outward csbox dict."""
    assert 0 <= s0 < s1 < 1
    cl = QI.sqrt_iv(1 - s1*s1, 1 - s1*s1)[0]
    ch = QI.sqrt_iv(1 - s0*s0, 1 - s0*s0)[1]
    if cfg == 'pentA':
        return {c: (cl, ch), s: (s0, s1)}
    if cfg == 'pentB':
        return {c: (-ch, -cl), s: (s0, s1)}
    raise ValueError(cfg)

def csbox_hex(c0, c1):
    """hex1 phi-cell by exact c-interval [c0,c1] (may straddle 0).
    r85: TIGHT s-interval.  On hex1 (phi in (phi_w, pi-phi_w)) we have
    s = sin phi = +sqrt(1-c^2) > 0, monotone decreasing in |c|; hence over
    c in [c0,c1] the exact range is s in [sqrt(1-m^2), sqrt(1-mn^2)] with
    m = max|c|, mn = min|c| on the interval (mn = 0 iff the cell straddles 0).
    Both ends outward-rounded.  The old bound (s_hi = 1) was sound but loose;
    the new box is a subset containing the true arc, so every certificate
    verified over it remains a certificate for the arc.  (The loose bound was
    the sole cause of the hex1 near-wall W3 'marginal residue': |dA/ds| ~ 70
    made A_hi jump from -0.93 (true sup) to +6.05, unrepairable by any
    w x tau subdivision because the (c,s) box never shrank.)"""
    m = max(abs(c0), abs(c1))
    assert m < 1
    mn = min(abs(c0), abs(c1)) if c0*c1 > 0 else Fr(0)
    sl = QI.sqrt_iv(1 - m*m, 1 - m*m)[0]
    sh = QI.sqrt_iv(1 - mn*mn, 1 - mn*mn)[1]
    if sh > 1: sh = Fr(1)
    return {c: (c0, c1), s: (sl, sh)}

def csbox_of(cfg, kind, lo, hi):
    if kind == 's': return csbox_pent(cfg, Fr(lo), Fr(hi))
    if kind == 'c': return csbox_hex(Fr(lo), Fr(hi))
    raise ValueError(kind)

def env_of(csbox, xbox):
    e = dict(csbox)
    for k in XS: e[k] = xbox[k]
    return e

# ------------------------------------------------------------------ exact stress solve
_STRESS_CACHE = {}
def solve_stress(cfg, sup, acts, c0req, mode):
    """Solve for a CONSTANT exact stress lam on support sup (tuple of row indices,
    subset of stress_pool):
        Sum lam = 1
        Sum lam G_Pi = 0                       (2 eqs; mode 'B': all 5 G columns = 0)
        for k in acts:  sigma-signed transverse equality  (Sum lam G_T)_k * sgn = -c0req
                        i.e.  v_k * sigma_k = c0req  with v = -Sum lam G_T
    acts: tuple of (k, sigma_k) pairs (mode 'W2'/'W3'); ignored for mode 'B'.
    Returns (lam list of sympy numbers, v list of 3 sympy numbers) or None.
    Verifies: system residual == 0 symbolically, lam_k >= 0 by exact interval."""
    key = (cfg, tuple(sup), tuple(acts), Fr(c0req), mode)
    if key in _STRESS_CACHE: return _STRESS_CACHE[key]
    pool, GM = Gexact(cfg)
    if any(i not in GM for i in sup):
        _STRESS_CACHE[key] = None; return None
    n = len(sup)
    lamv = sp.symbols(f'l0:{n}', real=True)
    eqs = [sp.Add(*lamv) - 1]
    cols = range(5) if mode == 'B' else range(2)
    for col in cols:
        eqs.append(sp.expand(sum(lamv[a] * GM[sup[a]][col] for a in range(n))))
    if mode != 'B':
        for (k, sg) in acts:
            vk = -sum(lamv[a] * GM[sup[a]][2 + k] for a in range(n))
            eqs.append(sp.expand(vk * sg - sp.Rational(Fr(c0req).numerator, Fr(c0req).denominator)))
    try:
        sol = sp.solve(eqs, lamv, dict=True)
    except Exception:
        sol = []
    if not sol:
        _STRESS_CACHE[key] = None; return None
    sol = sol[0]
    if any(v not in sol and str(v) not in [str(kk) for kk in sol] for v in lamv):
        # underdetermined -> reject (driver should shrink support)
        if len(sol) < n:
            _STRESS_CACHE[key] = None; return None
    lam = [sp.radsimp(sp.nsimplify(sol.get(v, v))) for v in lamv]
    if any(l.free_symbols for l in lam):
        _STRESS_CACHE[key] = None; return None
    # verify residual identically zero (SOUND: symbolic)
    for e in eqs:
        r = sp.simplify(sp.expand(e.subs({lamv[a]: lam[a] for a in range(n)})))
        if r != 0:
            _STRESS_CACHE[key] = None; return None
    # verify lam >= 0 exactly
    for l in lam:
        lo, hi = QI.eval_iv(sp.expand(l), {})
        if lo < 0:
            if hi < 0:
                _STRESS_CACHE[key] = None; return None
            # interval straddles 0: accept only if exactly zero symbolically
            if sp.simplify(l) != 0:
                _STRESS_CACHE[key] = None; return None
    vvec = [sp.radsimp(sp.expand(-sum(lam[a] * GM[sup[a]][2 + k] for a in range(n))))
            for k in range(3)]
    _STRESS_CACHE[key] = (lam, vvec)
    return _STRESS_CACHE[key]

_WX_CACHE = {}
def wexpr2(cfg, lam, sup, field):
    key = (cfg, field, tuple(sup), tuple(sp.srepr(l) for l in lam))
    if key in _WX_CACHE: return _WX_CACHE[key]
    rows = SYM[cfg]['tight']
    e = sp.expand(sum(lam[a] * rows[sup[a]][field] for a in range(len(sup))))
    _WX_CACHE[key] = e
    return e

# ------------------------------------------------------------------ shared brackets
def _sup_corners(gexpr, csboxes, xbox, taus):
    """exact sup of gexpr over in-plane box corners x given tau patterns,
    (c,s) enclosed over the UNION of csboxes (arc-consistent subdivision);
    + LHESS*2*hw^2 in-plane curvature charge (valid when gexpr is a lam-convex
    combination of rows, Sum lam = 1, lam >= 0 -- Lemma RB-1)."""
    if isinstance(csboxes, dict): csboxes = [csboxes]
    x0, x1 = xbox[xp1]; y0, y1 = xbox[xp2]
    hw = max(x1 - x0, y1 - y0) / 2
    mx = None
    for cb in csboxes:
        for cx in (x0, x1):
            for cy in (y0, y1):
                for (tg, tt1, tt2) in taus:
                    env = dict(cb)
                    env[xp1] = (cx, cx); env[xp2] = (cy, cy)
                    env[g] = (tg, tg); env[t1] = (tt1, tt1); env[t2] = (tt2, tt2)
                    v = QI.eval_iv(gexpr, env)[1]
                    mx = v if mx is None else max(mx, v)
    # gamma-concavity charge: g may be QUADRATIC in the spin symbol (t1,t2 are exactly
    # linear -- Lemma 5.1(a) -- but gamma is a rotation coordinate).  Corner enumeration
    # misses an interior max by at most |a_-| * h_gamma^2 where a_- is the negative part
    # of the gamma^2-coefficient over the cell.  (Gap found in the r65 bracket_hi.)
    d2 = sp.expand(sp.diff(gexpr, g, 2) / 2)
    if d2 != 0:
        hg = (xbox[g][1] - xbox[g][0]) / 2
        if hg > 0:
            d2lo = min(QI.eval_iv(d2, env_of(cb2, xbox))[0] for cb2 in csboxes)
            if d2lo < 0:
                mx += (-d2lo) * hg * hg
    return mx + LHESS * 2 * hw * hw

def _sup_abs(expr, csboxes, xbox):
    if isinstance(csboxes, dict): csboxes = [csboxes]
    e = sp.expand(expr)
    return max(QI.iabs(QI.eval_iv(e, env_of(cb, xbox)))[1] for cb in csboxes)

def _b1_tau(gexpr, h3expr, csboxes, xbox):
    """exact l1 tau-gradient bound of (gexpr + RHO0*h3expr) over the box."""
    if isinstance(csboxes, dict): csboxes = [csboxes]
    R = sp.Rational(1, 1000)
    best = None
    for cb in csboxes:
        env = env_of(cb, xbox)
        tot = (Fr(0), Fr(0))
        for tau in TAUS:
            d = sp.diff(gexpr, tau) + R * sp.diff(h3expr, tau)
            tot = QI.add(tot, QI.iabs(QI.eval_iv(sp.expand(d), env)))
        best = tot[1] if best is None else max(best, tot[1])
    return best


def split_csbox(csbox, ns):
    """subdivide a (c,s) box arc-consistently: split the wider of the two coordinate
    intervals; re-tighten the other from the circle relation (outward rounding).
    Returns list of csbox dicts whose union contains the original arc."""
    c0, c1 = csbox[c]; s0, s1 = csbox[s]
    out = []
    if (s1 - s0) >= (c1 - c0):
        for i in range(ns):
            sl = s0 + (s1 - s0) * i / ns; sh = s0 + (s1 - s0) * (i + 1) / ns
            m = max(abs(sl), abs(sh)); mn = min(abs(sl), abs(sh))
            chh = QI.sqrt_iv(1 - mn*mn, 1 - mn*mn)[1]
            cll = QI.sqrt_iv(1 - m*m, 1 - m*m)[0]
            if c0 >= 0:   cc = (max(c0, cll), min(c1, chh))
            elif c1 <= 0: cc = (max(c0, -chh), min(c1, -cll))
            else:         cc = (c0, c1)
            out.append({c: cc, s: (sl, sh)})
    else:
        for i in range(ns):
            cl = c0 + (c1 - c0) * i / ns; ch = c0 + (c1 - c0) * (i + 1) / ns
            m = max(abs(cl), abs(ch)); mn = min(abs(cl), abs(ch))
            shh = QI.sqrt_iv(1 - mn*mn, 1 - mn*mn)[1]
            sll = QI.sqrt_iv(1 - m*m, 1 - m*m)[0]
            out.append({c: (cl, ch), s: (max(s0, sll), min(s1, shh))})
    return out

def _sup_sub(expr, csbox, wint, ns, nw):
    """exact sup of expr(c,s,w) via arc-consistent (c,s) subdivision x w subdivision."""
    w0, w1 = wint
    mx = None
    for cb in split_csbox(csbox, ns):
        for j in range(nw):
            wl = w0 + (w1 - w0) * j / nw; wh = w0 + (w1 - w0) * (j + 1) / nw
            env = dict(cb); env[w] = (wl, wh)
            v = QI.eval_iv(expr, env)[1]
            mx = v if mx is None else max(mx, v)
    return mx

def _c0_lo(vvec, acts, c0req):
    """exact lower bound of min_k v_k*sigma_k given transverse sign pattern in acts;
    for k in acts the equality v_k*sigma_k = c0req holds by construction."""
    sgn = dict(acts)
    lo = Fr(c0req)
    for k in range(3):
        if k in sgn:  # equality row
            continue
        # inactive: need its sign too -- caller passes full sigma via acts_full
        return None
    return lo

# ------------------------------------------------------------------ witnesses
def wa2(cfg, i, csbox, xbox):
    """W-A single-row kill on the cell, all delta in (0,RHO0].
    sup f_i <= -eps and (bracket_i <= 0 or eps >= RHO0*bracket_i)."""
    rows = SYM[cfg]['tight']
    x0, x1 = xbox[xp1]; y0, y1 = xbox[xp2]
    pats = tau_patterns(xbox)
    mxf = None
    for cx in (x0, x1):
        for cy in (y0, y1):
            for (tg, tt1, tt2) in pats:
                env = dict(csbox)
                env[xp1] = (cx, cx); env[xp2] = (cy, cy)
                env[g] = (tg, tg); env[t1] = (tt1, tt1); env[t2] = (tt2, tt2)
                v = QI.eval_iv(rows[i]['f'], env)[1]
                mxf = v if mxf is None else max(mxf, v)
    if mxf >= 0: return False, None
    eps = -mxf
    br = None
    for ns in (1, 4, 8):
        cbs = split_csbox(csbox, ns)
        br = (_sup_corners(rows[i]['g'], cbs, xbox, pats)
              + RHO0 * _sup_abs(rows[i]['h3'], cbs, xbox) + RHO0 * RHO0 * C4)
        if (br <= 0) or (eps >= RHO0 * br):
            return True, dict(eps=str(eps), br=str(br))
    return False, dict(eps=str(eps), br=str(br))

def wb2(cfg, sup, csbox, xbox):
    """W-B full-stress kill: lam solves Sum lam=1, Sum lam G=0 (all 5) exactly =>
    Sum lam f == 0 identically (CS-1 zero drift on the pool); kill for all
    delta in (0,RHO0] iff bracket < 0."""
    st = solve_stress(cfg, sup, (), 0, 'B')
    if st is None: return False, None
    lam, _ = st
    ge = wexpr2(cfg, lam, sup, 'g'); he = wexpr2(cfg, lam, sup, 'h3')
    br = None
    for ns in (1, 4, 8, 16):
        cbs = split_csbox(csbox, ns)
        br = (_sup_corners(ge, cbs, xbox, _TP)
              + RHO0 * _sup_abs(he, cbs, xbox) + RHO0 * RHO0 * C4)
        if br < 0:
            return True, dict(br=str(br), lam=[sp.srepr(l) for l in lam])
    return False, dict(br=str(br), lam=[sp.srepr(l) for l in lam])

def ww22(cfg, sup, sigma, acts, c0req, csbox, xbox):
    """W-W2 orthant-stress kill on transverse orthant sigma (tuple of +-1):
    lam >= 0, Sum lam = 1, Sum lam G_Pi = 0 exactly, v = -Sum lam G_T with
    v_k sigma_k = c0req for k in acts and v_k sigma_k >= c0used for all k.
    Kill for all delta in (0,RHO0] iff  B0 < 0  and  c0used >= RHO0*B1,  where
    B0 bounds sup over the tau=0 face + RHO0-charged h3 + order-4, and B1 bounds
    the l1 tau-gradient of (Sum lam g + RHO0 Sum lam h3)."""
    st = solve_stress(cfg, sup, tuple((k, sigma[k]) for k in acts), c0req, 'W2')
    if st is None: return False, None
    lam, vvec = st
    # exact lower bound of v_k*sigma_k over all k
    c0u = Fr(c0req)
    for k in range(3):
        if k in acts: continue
        lo = QI.eval_iv(sp.expand(vvec[k] * sigma[k]), {})[0]
        c0u = min(c0u, lo)
    if c0u <= 0: return False, None
    ge = wexpr2(cfg, lam, sup, 'g'); he = wexpr2(cfg, lam, sup, 'h3')
    B0 = B1 = None
    for ns in (1, 4, 8, 16):
        cbs = split_csbox(csbox, ns)
        B0 = (_sup_corners(ge, cbs, xbox, [(Fr(0), Fr(0), Fr(0))])
              + RHO0 * _sup_abs(he, cbs, xbox) + RHO0 * RHO0 * C4)
        B1 = _b1_tau(ge, he, cbs, xbox)
        if (B0 < 0) and (c0u >= RHO0 * B1):
            return True, dict(B0=str(B0), B1=str(B1), c0=str(c0u),
                              lam=[sp.srepr(l) for l in lam])
    return False, dict(B0=str(B0), B1=str(B1), c0=str(c0u), lam=[sp.srepr(l) for l in lam])

def wc2(cfg, lam_fr, rowidx, sigma, csbox, box):
    """W-C cart cone-relaxed stress kill (hex1 near-corner cart AABB cells).
    Like W-W2 but the stress EQUALITY Sum lam G_Pi = 0 is RELAXED to the exact
    one-signed AABB cone   sup_{xhat_Pi in box} (Sum lam G_Pi) . xhat_Pi <= 0 :
    since Sum lam G_Pi . xhat is LINEAR in (xp1,xp2), the sup is at a box corner,
    so checking <=0 at the 4 corners certifies the in-plane first-order part of
    Sum lam f is a signed bonus <=0 over the whole box and is NEVER charged.  The
    transverse (v_k sigma_k >= c0u > 0) and 2nd-order (B0<0, tau=0 face) parts are
    the W-W2 computation verbatim.  All rows in stress_pool (constant exact G,
    zero drift); lam RATIONAL >=0, Sum lam =1.  Kill for all delta in (0,RHO0]:
      Sum lam h / delta = [V_Pi.xhat_Pi - v.tau] + delta*Sum lam g + delta^2*Sum lam h3 + O4
        <= 0 - c0u*S|tau| + delta*(sup_face + B1*S|tau|) + delta*(RHO0 sup|h3| + RHO0^2 C4)
        <= delta*B0 + S|tau|*(delta*B1 - c0u)  < 0   iff  B0<0 and c0u >= RHO0*B1."""
    x0, x1, y0, y1 = (Fr(b) for b in box)
    pool, GM = Gexact(cfg)
    if any(i not in pool for i in rowidx): return False, None
    assert all(l >= 0 for l in lam_fr) and sum(lam_fr) == 1
    lam = [sp.Rational(l.numerator, l.denominator) for l in lam_fr]
    V = [sp.expand(sum(lam[a] * GM[rowidx[a]][m] for a in range(len(rowidx))))
         for m in range(5)]
    # [cone] in-plane one-signed over the AABB: V0*xc + V1*yc <= 0 at the 4 corners
    for xc, yc in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
        if QI.eval_iv(sp.expand(V[0] * xc + V[1] * yc), {})[1] > 0:
            return False, dict(why='cone', corner=(str(xc), str(yc)))
    # [transverse] c0u = min_k lower(-V_k sigma_k) > 0
    c0u = None
    for k in range(3):
        lo = QI.eval_iv(sp.expand(-V[2 + k] * sigma[k]), {})[0]
        c0u = lo if c0u is None else min(c0u, lo)
    if c0u <= 0: return False, dict(why='c0', c0=str(c0u))
    ge = wexpr2(cfg, lam, rowidx, 'g'); he = wexpr2(cfg, lam, rowidx, 'h3')
    xbox = {xp1: (x0, x1), xp2: (y0, y1), g: (-BT, BT), t1: (-BT, BT), t2: (-BT, BT)}
    B0 = B1 = None
    for ns in (1, 4, 8, 16):
        cbs = split_csbox(csbox, ns)
        B0 = (_sup_corners(ge, cbs, xbox, [(Fr(0), Fr(0), Fr(0))])
              + RHO0 * _sup_abs(he, cbs, xbox) + RHO0 * RHO0 * C4)
        B1 = _b1_tau(ge, he, cbs, xbox)
        if (B0 < 0) and (c0u >= RHO0 * B1):
            return True, dict(B0=str(B0), B1=str(B1), c0=str(c0u))
    return False, dict(B0=str(B0), B1=str(B1), c0=str(c0u))

def _ecomp(half):
    den = 1 + w*w
    return (sp.Integer(half) * (1 - w*w) / den, sp.Integer(half) * 2 * w / den)

_G0_CACHE = {}
def _g0_zero(cfg, lam, sup, corner):
    """verify Sum lam g == 0 identically at the corner (tau=0): CS-1 / CS-1-far."""
    key = (cfg, tuple(sup), tuple(sp.srepr(l) for l in lam), corner)
    if key in _G0_CACHE: return _G0_CACHE[key]
    ge = wexpr2(cfg, lam, sup, 'g')
    P = {xp1: 0, xp2: 0} if corner == 'near' else {xp1: 2*c, xp2: 2*s}
    e = sp.expand(ge.subs({**P, g: 0, t1: 0, t2: 0}))
    if e != 0:
        e = sp.simplify(sp.expand(e).subs(s**2, 1 - c**2))
    _G0_CACHE[key] = (e == 0)
    return _G0_CACHE[key]

def w32(cfg, sup, sigma, acts, c0req, csbox, corner, half, wint, tband):
    """W-W3 radial-sector orthant-stress kill.
    Region: { P0(c,s) + t*e : t in [t0,t1], e = half*((1-w^2)/(1+w^2), 2w/(1+w^2)),
              w in [w0,w1] } x transverse box, over the (c,s) cell;
    P0 = (0,0) ('near') or (2c,2s) ('far', support must be within Z rows).
    Stress as in W-W2.  On the tau=0 face, radial homogeneity is EXACT:
       Sum lam g(P0 + t e, 0) = t*A(w;c,s) + t^2*B(w;c,s)
    (the t^0 term vanishes identically by CS-1 / CS-1-far, verified symbolically).
    B0 = sup_t,w,cs [tA + t^2 B] + RHO0*sup|Sum lam h3| + RHO0^2 C4 < 0
    and c0used >= RHO0*B1 as in W-W2 (h3/B1 over the sector AABB, tau in full box)."""
    t0, t1_ = Fr(tband[0]), Fr(tband[1])
    w0, w1 = Fr(wint[0]), Fr(wint[1])
    if corner == 'far':
        Z = z_rows(cfg)
        if any(i not in Z for i in sup): return False, None
    st = solve_stress(cfg, sup, tuple((k, sigma[k]) for k in acts), c0req, 'W2')
    if st is None: return False, None
    lam, vvec = st
    if not _g0_zero(cfg, lam, sup, corner): return False, None
    c0u = Fr(c0req)
    for k in range(3):
        if k in acts: continue
        lo = QI.eval_iv(sp.expand(vvec[k] * sigma[k]), {})[0]
        c0u = min(c0u, lo)
    if c0u <= 0: return False, None
    ge = wexpr2(cfg, lam, sup, 'g'); he = wexpr2(cfg, lam, sup, 'h3')
    ex, ey = _ecomp(half)
    P0 = (sp.Integer(0), sp.Integer(0)) if corner == 'near' else (2*c, 2*s)
    tau0 = {g: 0, t1: 0, t2: 0}
    gx = sp.diff(ge, xp1).subs({xp1: P0[0], xp2: P0[1], **tau0})
    gy = sp.diff(ge, xp2).subs({xp1: P0[0], xp2: P0[1], **tau0})
    gxx = sp.diff(ge, xp1, 2); gxy = sp.diff(ge, xp1, xp2); gyy = sp.diff(ge, xp2, 2)
    A = sp.expand(gx * ex + gy * ey)
    B = sp.expand((gxx * ex * ex + 2 * gxy * ex * ey + gyy * ey * ey) / 2)
    A_hi = _sup_sub(A, csbox, (w0, w1), 8, 2)
    B_hi = _sup_sub(B, csbox, (w0, w1), 4, 1)
    B_iv = (None, B_hi)
    # sup over t in [t0,t1] of t*A_hi + t^2*B_hi  (exact; endpoints + interior vertex)
    def psi(tv): return tv * A_hi + tv * tv * B_iv[1]
    cands = [psi(t0), psi(t1_)]
    if B_hi < 0 and A_hi > 0:
        tstar = -A_hi / (2 * B_hi)
        if t0 < tstar < t1_: cands.append(psi(tstar))
    sup_ray = max(cands)
    # sector AABB for h3 / B1 (xp intervals = P0 + [t0,t1]*e-interval)
    exi = QI.eval_iv(ex, {w: (w0, w1)}); eyi = QI.eval_iv(ey, {w: (w0, w1)})
    p0x = (Fr(0), Fr(0)) if corner == 'near' else QI.mul((Fr(2), Fr(2)), csbox[c])
    p0y = (Fr(0), Fr(0)) if corner == 'near' else QI.mul((Fr(2), Fr(2)), csbox[s])
    xb = {xp1: QI.add(p0x, QI.mul((t0, t1_), exi)),
          xp2: QI.add(p0y, QI.mul((t0, t1_), eyi)),
          g: (-BT, BT), t1: (-BT, BT), t2: (-BT, BT)}
    B0 = sup_ray + RHO0 * _sup_abs(he, csbox, xb) + RHO0 * RHO0 * C4
    B1 = _b1_tau(ge, he, csbox, xb)
    ok = (B0 < 0) and (c0u >= RHO0 * B1)
    return ok, dict(B0=str(B0), B1=str(B1), c0=str(c0u), A_hi=str(A_hi),
                    lam=[sp.srepr(l) for l in lam])


def w12(cfg, lam_fr, rowidx, csbox, xbox):
    """W-W1 combo first-order kill on cell x tau-sub-box, RATIONAL weights.
    lam_fr: list of Fractions >= 0 summing to 1 (exactness of the sum is checked),
    over arbitrary tight rows (no stress identity needed -- pure f-combination):
      sup_cell Sum lam f  <= -eps   (corner enumeration + (c,s) splitting; f affine in x)
      kill for all delta <= RHO0 iff  eps >= RHO0 * bracket,
      bracket = sup Sum lam g + RHO0 sup|Sum lam h3| + RHO0^2 C4  (gamma-charged)."""
    rows = SYM[cfg]['tight']
    assert all(l >= 0 for l in lam_fr) and sum(lam_fr) == 1
    lam = [sp.Rational(l.numerator, l.denominator) for l in lam_fr]
    fe = sp.expand(sum(lam[a] * rows[rowidx[a]]['f'] for a in range(len(rowidx))))
    ge = sp.expand(sum(lam[a] * rows[rowidx[a]]['g'] for a in range(len(rowidx))))
    he = sp.expand(sum(lam[a] * rows[rowidx[a]]['h3'] for a in range(len(rowidx))))
    x0, x1 = xbox[xp1]; y0, y1 = xbox[xp2]
    pats = tau_patterns(xbox)
    mxf = None
    for ns in (1, 4):
        cbs = split_csbox(csbox, ns)
        mxf = None
        for cb in cbs:
            for cx in (x0, x1):
                for cy in (y0, y1):
                    for (tg, tt1, tt2) in pats:
                        env = dict(cb)
                        env[xp1] = (cx, cx); env[xp2] = (cy, cy)
                        env[g] = (tg, tg); env[t1] = (tt1, tt1); env[t2] = (tt2, tt2)
                        v = QI.eval_iv(fe, env)[1]
                        mxf = v if mxf is None else max(mxf, v)
        if mxf < 0: break
    if mxf >= 0: return False, None
    eps = -mxf
    br = None
    for ns in (1, 4):
        cbs = split_csbox(csbox, ns)
        br = (_sup_corners(ge, cbs, xbox, pats)
              + RHO0 * _sup_abs(he, cbs, xbox) + RHO0 * RHO0 * C4)
        if (br <= 0) or (eps >= RHO0 * br):
            return True, dict(eps=str(eps), br=str(br))
    return False, dict(eps=str(eps), br=str(br))

def orth_subbox(xbox, sigma):
    """tau sub-box of xbox for orthant sigma."""
    out = dict(xbox)
    for k, tau in enumerate(TAUS):
        out[tau] = (Fr(0), BT) if sigma[k] > 0 else (-BT, Fr(0))
    return out

# ------------------------------------------------------------------ selfcheck
def selfcheck(verbose=True):
    """machine-verify K1 (constant G off the excluded rows) and CS-1 / CS-1-far
    identities for the stress pools; report pool/Z sizes."""
    rep = {}
    for cfg in SYM:
        pool, GM = Gexact(cfg)
        ep = ep_rows(cfg)
        excl = [i for i in ep if i not in pool]
        # CS-1: pool rows have zero constant terms of f,g,h3
        cs1 = True
        for i in pool:
            r = SYM[cfg]['tight'][i]
            sub = {xp1: 0, xp2: 0, g: 0, t1: 0, t2: 0}
            for fld in ('f', 'g', 'h3'):
                e = sp.expand(r[fld].subs(sub))
                if e != 0:
                    e = sp.simplify(sp.expand(e).subs(s**2, 1 - c**2))
                if e != 0: cs1 = False
        Z = z_rows(cfg)
        rep[cfg] = dict(n_tight=len(SYM[cfg]['tight']), n_ep=len(ep),
                        pool=pool, excluded=excl, cs1=cs1, Z=Z)
        if verbose:
            print(f"{cfg}: tight={len(SYM[cfg]['tight'])} ep={len(ep)} pool={len(pool)} "
                  f"excl={excl} CS-1={cs1} |Z|={len(Z)} Z={Z}")
    return rep

if __name__ == '__main__':
    selfcheck()


def _sup_quad_w(Qw, w0, w1):
    """exact-outward sup over w in [w0,w1] of a quadratic (in w) with radical-field
    constant coefficients: candidates = endpoints + interior stationary point."""
    P = sp.Poly(sp.expand(Qw), w)
    cs = P.all_coeffs()  # degree<=2
    while len(cs) < 3: cs = [sp.Integer(0)] + cs
    q2, q1, q0 = cs
    cands = [Qw.subs(w, sp.Rational(w0.numerator, w0.denominator)),
             Qw.subs(w, sp.Rational(w1.numerator, w1.denominator))]
    if q2 != 0:
        # stationary point w* = -q1/(2 q2); include iff bracketable inside [w0,w1]
        ws = sp.expand(-q1 / (2 * q2))
        lo_w, hi_w = QI.eval_iv(ws, {})
        if hi_w > w0 and lo_w < w1:
            cands.append(sp.expand(q0 - q1 * q1 / (4 * q2)))
    return max(QI.eval_iv(sp.expand(cnd), {})[1] for cnd in cands)


def w42(cfg, lam_fr, rowidx, sigma, csbox, half, wint, tband):
    """W-W4 sector-cone MIXED kill at the NEAR corner (wall-edge sectors).

    Like W-W3 but the stress equality Sum lam G_Pi = 0 is RELAXED to the exact
    one-signed cone condition   sup_{w in [w0,w1]} (Sum lam G_Pi) . e(w) <= 0 :
    the in-plane first-order part  t * V_Pi.e(w)  of  Sum lam f  is then a signed
    bonus <= 0 on the whole sector (t >= 0) and is NEVER charged.  This unlocks
    the transitioning-vertex flags (constant-G pool rows whose in-plane gradient
    vanishes exactly on the corner-cone edge theta = -+phi_w), which carry the
    transverse kill needed in the stuck orthants while their in-plane action is
    one-signed on wall-edge sectors.  All rows must lie in stress_pool (constant
    exact G, zero drift), lam RATIONAL >= 0 with Sum lam = 1 (only inequalities
    are verified, so no radical-field solve is needed).

    Kill inequality (all delta in (0,RHO0], tau in the sigma-orthant box, phi in
    the cell):  Sum lam h / delta =
         [t*V_Pi.e(w) - v.tau] + delta*[Sum lam g] + delta^2*[Sum lam h3] + O4
      <= 0 - c0u*S|tau| + delta*(sup_ray + B1*S|tau|) + delta*(RHO0*sup|h3| + RHO0^2*C4)
      <= delta*B0 + S|tau|*(delta*B1 - c0u)  <  0
    iff  B0 < 0  and  c0u >= RHO0*B1   -- the W-W3 computation verbatim plus the
    signed bonus term (cone condition).  CS-1 (Sum lam g == 0 at the corner,
    tau=0) holds per-row on the pool and is re-verified symbolically."""
    t0, t1_ = Fr(tband[0]), Fr(tband[1])
    w0, w1 = Fr(wint[0]), Fr(wint[1])
    assert t0 >= 0
    pool, GM = Gexact(cfg)
    if any(i not in pool for i in rowidx): return False, None
    assert all(l >= 0 for l in lam_fr) and sum(lam_fr) == 1
    lam = [sp.Rational(l.numerator, l.denominator) for l in lam_fr]
    # V = Sum lam G  (exact radical-field constants)
    V = [sp.expand(sum(lam[a] * GM[rowidx[a]][m] for a in range(len(rowidx))))
         for m in range(5)]
    # [cone] Q(w) = half * (V1*(1-w^2) + V2*2w)  <= 0  on [w0,w1]
    #        (equals (1+w^2) * V_Pi.e(w), same sign)
    Qw = sp.expand(sp.Integer(half) * (V[0] * (1 - w * w) + V[1] * 2 * w))
    q_hi = _sup_quad_w(Qw, w0, w1)
    if q_hi > 0: return False, dict(why='cone', q_hi=str(q_hi))
    # [transverse] v = -V_T ; c0u = min_k lower(v_k sigma_k) > 0
    c0u = None
    for k in range(3):
        lo = QI.eval_iv(sp.expand(-V[2 + k] * sigma[k]), {})[0]
        c0u = lo if c0u is None else min(c0u, lo)
    if c0u <= 0: return False, dict(why='c0', c0=str(c0u))
    # [CS-1] corner identity
    if not _g0_zero(cfg, lam, rowidx, 'near'): return False, dict(why='g0')
    ge = wexpr2(cfg, lam, rowidx, 'g'); he = wexpr2(cfg, lam, rowidx, 'h3')
    ex, ey = _ecomp(half)
    tau0 = {g: 0, t1: 0, t2: 0}
    gx = sp.diff(ge, xp1).subs({xp1: 0, xp2: 0, **tau0})
    gy = sp.diff(ge, xp2).subs({xp1: 0, xp2: 0, **tau0})
    gxx = sp.diff(ge, xp1, 2); gxy = sp.diff(ge, xp1, xp2); gyy = sp.diff(ge, xp2, 2)
    A = sp.expand(gx * ex + gy * ey)
    B = sp.expand((gxx * ex * ex + 2 * gxy * ex * ey + gyy * ey * ey) / 2)
    A_hi = _sup_sub(A, csbox, (w0, w1), 8, 2)
    B_hi = _sup_sub(B, csbox, (w0, w1), 4, 1)
    def psi(tv): return tv * A_hi + tv * tv * B_hi
    cands = [psi(t0), psi(t1_)]
    if B_hi < 0 and A_hi > 0:
        tstar = -A_hi / (2 * B_hi)
        if t0 < tstar < t1_: cands.append(psi(tstar))
    sup_ray = max(cands)
    exi = QI.eval_iv(ex, {w: (w0, w1)}); eyi = QI.eval_iv(ey, {w: (w0, w1)})
    xb = {xp1: QI.mul((t0, t1_), exi), xp2: QI.mul((t0, t1_), eyi),
          g: (-BT, BT), t1: (-BT, BT), t2: (-BT, BT)}
    B0 = sup_ray + RHO0 * _sup_abs(he, csbox, xb) + RHO0 * RHO0 * C4
    B1 = _b1_tau(ge, he, csbox, xb)
    ok = (B0 < 0) and (c0u >= RHO0 * B1)
    return ok, dict(B0=str(B0), B1=str(B1), c0=str(c0u), q_hi=str(q_hi),
                    A_hi=str(A_hi))


# ------------------------------------------------------------------ Lemma H (r76/77)
_RH_CACHE = {}
def h_rows(cfg):
    """R_H: tight rows that are identically PURE-TRANSVERSE (Lemma H, r76):
    f has zero constant term and zero in-plane gradient IDENTICALLY in (c,s)
    (checked symbolically, s^2 -> 1-c^2), and g vanishes at x=0.  For such rows
    f_i = T_i(c,s) . tau EXACTLY (f is linear in x by construction)."""
    if cfg in _RH_CACHE: return _RH_CACHE[cfg]
    rows = SYM[cfg]['tight']
    out = []
    Z0 = {xp1: 0, xp2: 0, g: 0, t1: 0, t2: 0}
    for i, r in enumerate(rows):
        ok = True
        for e in (r['f'].subs(Z0), sp.diff(r['f'], xp1), sp.diff(r['f'], xp2),
                  r['g'].subs(Z0)):
            e = sp.expand(e)
            if e != 0:
                e = sp.simplify(sp.expand(e).subs(s**2, 1 - c**2))
            if e != 0:
                ok = False; break
        if ok: out.append(i)
    _RH_CACHE[cfg] = out
    return out


def h2(cfg, lam_fr, rowidx, sigma, csbox, half, wint, tband):
    """W-H homogeneous transverse Farkas kill (Lemma H, r76) at the NEAR corner.

    rowidx must lie in h_rows(cfg) (pure-transverse rows of THIS chart: f_i =
    T_i(c,s).tau exactly, g_i(0)=0).  lam_fr rational >= 0, Sum = 1.  NO stress
    equality, NO cone condition: the kill is uniform across the D_Pi boundary.

    Region: x_Pi = t e(w), t in tband (t0>=0), w in wint, e = half*((1-w^2),2w)/(1+w^2);
    tau in the sigma-orthant box [0,BT]^sgn; (c,s) in csbox; all delta in (0,RHO0].

    Chain (h_i/delta = f_i + delta g_i + delta^2 h3_i + r_i, |Sum lam r_i| <= delta^3 C4):
      Sum lam h_i / delta
        =  Sum_k T^l_k(c,s) tau_k  +  delta g^l(x)  +  delta^2 h3^l(x)  +  r
       <=  Sum_k |tau_k| [ hi_cs(T^l_k sigma_k) + RHO0 L_k ]                    (H1)
         + delta [ sup_ray + RHO0 sup|h3^l| + RHO0^2 C4 ]                       (H2)
    where L_k = sup |d g^l/d tau_k| over csbox x sector-AABB x orthant tau-box
    (mean value along tau coordinate segments, which stay in the box), and
    sup_ray = sup_{t,w,cs} [t A + t^2 B]  with  g^l(t e(w), 0) = t A(w) + t^2 B(w)
    EXACTLY (g^l has zero constant term at x=0 -- per-row R_H property -- and
    degree <= 2 in (xp1,xp2) at tau=0, verified symbolically below).
    Kill iff (H1) each bracket <= 0 and (H2) the delta bracket <= -m < 0: then
    Sum lam h_i <= -delta^2 m < 0, so some REAL flag h_i < 0 on the cell.  SOUND:
    convex combination of the chart's own tight rows (subset-safe, Remark 4.2)."""
    t0, t1_ = Fr(tband[0]), Fr(tband[1])
    w0, w1 = Fr(wint[0]), Fr(wint[1])
    assert t0 >= 0
    RH = h_rows(cfg)
    pool, _GM = Gexact(cfg)
    # C4<=60 is proved for convex combinations of stress_pool rows (r69); require
    # rowidx within R_H AND the pool so the order-4 charge is in proved scope.
    if any((i not in RH) or (i not in pool) for i in rowidx): return False, dict(why='rows')
    assert all(l >= 0 for l in lam_fr) and sum(lam_fr) == 1
    lam = [sp.Rational(l.numerator, l.denominator) for l in lam_fr]
    rows = SYM[cfg]['tight']
    fe = sp.expand(sum(lam[a] * rows[rowidx[a]]['f'] for a in range(len(rowidx))))
    ge = sp.expand(sum(lam[a] * rows[rowidx[a]]['g'] for a in range(len(rowidx))))
    he = sp.expand(sum(lam[a] * rows[rowidx[a]]['h3'] for a in range(len(rowidx))))
    Z0 = {xp1: 0, xp2: 0, g: 0, t1: 0, t2: 0}
    # combo-level re-verification of pure-transversality (defense in depth)
    for e0 in (fe.subs(Z0), sp.diff(fe, xp1), sp.diff(fe, xp2), ge.subs(Z0)):
        e0 = sp.expand(e0)
        if e0 != 0:
            e0 = sp.simplify(sp.expand(e0).subs(s**2, 1 - c**2))
        if e0 != 0: return False, dict(why='not-transverse')
    # sector AABB x orthant tau box
    ex, ey = _ecomp(half)
    exi = QI.eval_iv(ex, {w: (w0, w1)}); eyi = QI.eval_iv(ey, {w: (w0, w1)})
    xb = {xp1: QI.mul((t0, t1_), exi), xp2: QI.mul((t0, t1_), eyi)}
    for k, tau in enumerate(TAUS):
        xb[tau] = (Fr(0), BT) if sigma[k] > 0 else (-BT, Fr(0))
    # [H1] per transverse coordinate
    H1 = []
    for k, tau in enumerate(TAUS):
        Tk = sp.expand(sp.diff(fe, tau))
        if Tk.has(xp1) or Tk.has(xp2) or Tk.has(g) or Tk.has(t1) or Tk.has(t2):
            return False, dict(why='f-nonlinear')
        thi = QI.eval_iv(sp.expand(Tk * sigma[k]), {c: csbox[c], s: csbox[s]})[1]
        Lk = QI.iabs(QI.eval_iv(sp.expand(sp.diff(ge, tau)), env_of(csbox, xb)))[1]
        H1.append(thi + RHO0 * Lk)
    if any(v > 0 for v in H1):
        return False, dict(why='H1', H1=[str(v) for v in H1])
    # [H2] exact ray identity  g^l(t e(w), 0) = t A(w) + t^2 B(w)
    geP = sp.expand(ge.subs({g: 0, t1: 0, t2: 0}))
    if sp.Poly(geP, xp1, xp2).total_degree() > 2:
        return False, dict(why='g-cubic')
    gx = sp.diff(geP, xp1).subs({xp1: 0, xp2: 0})
    gy = sp.diff(geP, xp2).subs({xp1: 0, xp2: 0})
    gxx = sp.diff(geP, xp1, 2); gxy = sp.diff(sp.diff(geP, xp1), xp2); gyy = sp.diff(geP, xp2, 2)
    A = sp.expand(gx * ex + gy * ey)
    B = sp.expand((gxx * ex * ex + 2 * gxy * ex * ey + gyy * ey * ey) / 2)
    A_hi = _sup_sub(A, csbox, (w0, w1), 8, 2)
    B_hi = _sup_sub(B, csbox, (w0, w1), 4, 1)
    def psi(tv): return tv * A_hi + tv * tv * B_hi
    cands = [psi(t0), psi(t1_)]
    if B_hi < 0 and A_hi > 0:
        tstar = -A_hi / (2 * B_hi)
        if t0 < tstar < t1_: cands.append(psi(tstar))
    sup_ray = max(cands)
    B0 = sup_ray + RHO0 * _sup_abs(he, csbox, xb) + RHO0 * RHO0 * C4
    ok = (B0 < 0)
    return ok, dict(B0=str(B0), H1=[str(v) for v in H1], A_hi=str(A_hi), m=str(-B0))
