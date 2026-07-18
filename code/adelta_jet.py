"""adelta_jet.py -- CERTIFIED bound on A_δ = sup |d²r_P/dδ²| (flag δ-acceleration).

This eliminates the paper's deferred residual (b): the §15.6 between-rung continuum argument
uses L_δ ≤ ½A_δ, where A_δ = sup over the localization box of the second δ-derivative of the
critical-disk flags r_P.  We compute a RIGOROUS upper bound by propagating a degree-2 Taylor
jet in δ (value, d/dδ, d²/dδ²), each coefficient a fast_interval FI over the δ×φ cell, through
the exact flag construction u₂→frame→applyW→proj→edge-normal→flag (mirroring flags_np/flags_FI).

Soundness: interval arithmetic on the 2-jet returns an enclosure of d²r_P/dδ² over the WHOLE cell
(δ via the trig seeds carrying the δ-interval; φ via cos/sin-φ interval constants).  The hull
combinatorial structure (active edges/vertices, normal orientation signs) is fixed from the cell
midpoint and re-validated as one-sided per cell (endpoints share the structure), so the smooth flag
map is what we differentiate -- exactly the regime the paper's A_δ refers to.  Transition windows
(±TRW) where the structure changes are EXCLUDED here; the paper charges those to margin-continuity,
not to A_δ (same handling as the rung sweep).  Chain-rule division/sqrt use the standard 2-jet
recursions; every op is outward-rounded via FI.

A_δ = max over active flags, components, and cells of max(|dd.lo|,|dd.hi|).
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_interval as F
from fast_interval import FI
import numpy as np
from scipy.spatial import ConvexHull

R2 = FI(F.rd(math.sqrt(0.5)), F.ru(math.sqrt(0.5)))
AA = 11.0 / 20.0                      # stellation scale a
V4 = [[1.,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
VERTS = [[FI(x) for x in r] for r in V4] + [[FI(-AA*x) for x in r] for r in V4]
Z3 = FI(0.0)

# --- float geometry (structure detection only; matches sbbox_prove) ---
_V4 = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
VNP = np.vstack([_V4, -AA*_V4])
ZNP = np.array([[0., 1, 0], [-1, 0, 0]])
_ustar = np.array([1, -1, 0.])/math.sqrt(2); _B1 = np.array([0, 0, 1.]); _B2 = -np.array([1, 1, 0.])/math.sqrt(2)
def frame_np(delta, phi):
    v = math.sin(phi)*_B1 - math.cos(phi)*_B2
    u = math.cos(delta)*_ustar + math.sin(delta)*v; u = u/np.linalg.norm(u)
    h = np.array([0, 0, 1.]) if abs(u[2]) < 0.9 else np.array([1, 0, 0.])
    f1 = np.cross(h, u); f1 = f1/np.linalg.norm(f1); f2 = np.cross(u, f1)
    return np.array([f1, f2, u])

class J2:
    """degree-2 Taylor jet in δ: value v, first deriv d, second deriv dd (each an FI)."""
    __slots__ = ('v','d','dd')
    def __init__(s, v, d=None, dd=None):
        s.v = v; s.d = Z3 if d is None else d; s.dd = Z3 if dd is None else dd
    def __add__(s, o):
        if not isinstance(o, J2): o = J2(o)
        return J2(s.v+o.v, s.d+o.d, s.dd+o.dd)
    def __sub__(s, o):
        if not isinstance(o, J2): o = J2(o)
        return J2(s.v-o.v, s.d-o.d, s.dd-o.dd)
    def __neg__(s): return J2(Z3-s.v, Z3-s.d, Z3-s.dd)
    def __mul__(s, o):
        if not isinstance(o, J2): o = J2(o)
        v = s.v*o.v
        d = s.d*o.v + s.v*o.d
        dd = s.dd*o.v + (s.d*o.d)*FI(2.0) + s.v*o.dd
        return J2(v, d, dd)
    def __truediv__(s, o):
        if not isinstance(o, J2): o = J2(o)
        # q=s/o : s=q*o -> s'=q'o+qo', s''=q''o+2q'o'+qo''
        v = s.v/o.v
        d = (s.d - v*o.d)/o.v
        dd = (s.dd - d*o.d*FI(2.0) - v*o.dd)/o.v
        return J2(v, d, dd)
    def sqrt(s):
        # r=sqrt(s): r'=s'/(2r), r''=(s''-2 r'^2)/(2r)
        r = s.v.sqrt()
        two_r = r*FI(2.0)
        d = s.d/two_r
        dd = (s.dd - d.sqr()*FI(2.0))/two_r
        return J2(r, d, dd)

# ---- jet vector ops (lists of 3 J2) ----
def jdot(u, v):
    r = u[0]*v[0]
    for i in (1,2): r = r + u[i]*v[i]
    return r
def jcross(u, v):
    return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
def jvnorm(u):
    s = u[0].__mul__(u[0])
    for i in (1,2): s = s + u[i]*u[i]
    return s.sqrt()
def jnormalize(u):
    n = jvnorm(u); return [c/n for c in u]

def det2_tight(a, b, c, d):
    """rigorous TIGHT enclosure of a*b - c*d (a 2x2 determinant), via a centered form around each
    FI's midpoint. The naive a*b - c*d loses the cancellation when a*b and c*d are large and nearly
    equal (short-edge cross products); centering keeps the exact point-difference and charges only the
    small radius-coupling terms. Sound: center is an FI point-product difference, err is rounded up."""
    ac = 0.5*(a.lo+a.hi); ar = 0.5*(a.hi-a.lo)
    bc = 0.5*(b.lo+b.hi); br = 0.5*(b.hi-b.lo)
    cc = 0.5*(c.lo+c.hi); cr = 0.5*(c.hi-c.lo)
    dc = 0.5*(d.lo+d.hi); dr = 0.5*(d.hi-d.lo)
    center = FI(ac)*FI(bc) - FI(cc)*FI(dc)               # exact-ish point difference (rigorous FI)
    e = (abs(bc)*ar + abs(ac)*br + ar*br) + (abs(dc)*cr + abs(cc)*dr + cr*dr)
    err = F.ru(e*(1.0+1e-12) + 1e-300)
    return FI(F.rd(center.lo-err), F.ru(center.hi+err))

def junit2d(mx, my):
    """tight unit-vector 2-jet for a 2D normal m=(mx,my) given as J2. Computes the unit normal's
    derivatives through its ANGLE ψ=atan2(my,mx): ĥ=(cosψ,sinψ), ĥ'=ψ'ĥ^⊥, ĥ''=-ψ'²ĥ+ψ''ĥ^⊥,
    with ψ'=(m×m')/L², ψ''=(N'D−N D')/D² (N=m×m'=mx·my'−my·mx', D=L²=mx²+my²). This is TIGHT even
    for a short edge (small L): N and D are both small but exact-tight, so their ratio does not
    inflate, unlike a generic m/L vector-division jet whose 1/L³ terms lose the cancellation.
    Returns (ĥx, ĥy) as J2. Rigorous: every step is FI-outward-rounded; D=L²>0 off the edge collapse."""
    two = FI(2.0)
    L = (mx.v.sqr() + my.v.sqr()).sqrt()
    hx = mx.v/L; hy = my.v/L                              # unit normal value ĥ
    D = mx.v.sqr() + my.v.sqr()                           # L²
    N = det2_tight(mx.v, my.d, my.v, mx.d)               # m×m' = mx·my'−my·mx' (centered det)
    Np = det2_tight(mx.v, my.dd, my.v, mx.dd)            # N' = mx·my''−my·mx'' (centered det)
    Dp = two*(mx.v*mx.d + my.v*my.d)                      # (L²)'  (sum: no cancellation)
    psip = N/D                                            # ψ'
    psipp = det2_tight(Np, D, N, Dp)/(D.sqr())           # ψ'' = (N'D−N D')/D²  (centered num)
    hxp = Z3 - psip*hy;  hyp = psip*hx                    # ĥ' = ψ' ĥ^⊥ = ψ'(−ĥy, ĥx)
    hxpp = (Z3 - psip.sqr()*hx) - psipp*hy               # ĥ''_x = −ψ'²ĥx − ψ''ĥy
    hypp = (Z3 - psip.sqr()*hy) + psipp*hx               # ĥ''_y = −ψ'²ĥy + ψ''ĥx
    return J2(hx, hxp, hxpp), J2(hy, hyp, hypp)

E3J = [J2(FI(0.0)), J2(FI(0.0)), J2(FI(1.0))]

def frame_jet(u):
    u = jnormalize(u)
    f1 = jnormalize(jcross(E3J, u))
    f2 = jcross(u, f1)
    return [f1, f2, u]

def u2_jet(dlo, dhi, plo, phi):
    """u₂(δ,φ)=cosδ·(r2,-r2,0)+sinδ·(cp·r2,cp·r2,sp) as a 3-vector of J2 in δ, over δ∈[dlo,dhi],
    φ∈[plo,phi]. δ enters via the cos/sin jets (value,-sin,-cos)/(sin,cos,-sin); φ via cp,sp const."""
    cd, sd = F.trig_fi(dlo, dhi)
    cp, sp = F.trig_fi(plo, phi)
    cosd = J2(cd, Z3-sd, Z3-cd)       # d/dδ cos=-sin ; d²/dδ² cos=-cos
    sind = J2(sd, cd, Z3-sd)          # d/dδ sin= cos ; d²/dδ² sin=-sin
    cpr = J2(cp*R2); spj = J2(sp)     # φ-constants (w.r.t. δ)
    ux = cosd*R2 + sind*cpr
    uy = cosd*(Z3-R2) + sind*cpr
    uz = sind*spj
    return [ux, uy, uz]

def _hull2d(pts):
    """CCW convex-hull vertex indices of 8 planar points (monotone chain). ~15µs, no scipy."""
    idx = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    def cross(o, a, b):
        return (pts[a][0]-pts[o][0])*(pts[b][1]-pts[o][1]) - (pts[a][1]-pts[o][1])*(pts[b][0]-pts[o][0])
    lower = []
    for i in idx:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0: lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(idx):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0: upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]

def flag_struct(delta, phi):
    """hull EDGE structure (p_, r_, sign) at the (δ,φ) POINT, via a fast 2D hull (no scipy). We key
    one-sidedness on the combinatorial EDGE set (tolerance-free, stable off transitions) rather than
    the on-edge vertex tolerance that flips marginally near a transition. Returns (edges, n) or None."""
    W = frame_np(delta, phi); qf = (ZNP @ (W @ VNP.T)).T     # 8×2 projected vertices
    try: idx = _hull2d(qf)
    except Exception: return None
    if len(idx) < 3: return None
    ctr = qf[idx].mean(0)
    edges = []
    for k in range(len(idx)):
        p_, r_ = idx[k], idx[(k+1)%len(idx)]
        dvec = qf[r_]-qf[p_]; nf = np.array([-dvec[1], dvec[0]]); nn = np.linalg.norm(nf)
        if nn == 0: return None
        nf = nf/nn
        sign = 1.0 if nf@(ctr-qf[p_]) >= 0 else -1.0
        edges.append((p_, r_, sign))
    return edges, len(edges)

def flags_jet(dlo, dhi, plo, phi, struct=None):
    """flags as 5-lists of J2 (δ-jets) over the cell, for a FIXED hull structure. If struct is None
    it is detected at the cell midpoint; the caller must keep the cell inside a smooth region (the
    sweep validates one-sidedness by matching struct at the cell corners). None if degenerate."""
    dmid = 0.5*(dlo+dhi); pmid = 0.5*(plo+phi)
    if struct is None:
        s = flag_struct(dmid, pmid)
        if s is None: return None
        struct, nfl = s
    else:
        nfl = len(struct)
    # --- jet construction over the cell ---
    u2 = u2_jet(dlo, dhi, plo, phi)
    W = frame_jet(u2)
    Wv = [[jdot(W[r], [J2(x) for x in V]) for r in range(3)] for V in VERTS]
    q = [[wv[1], J2(FI(0.0))-wv[0]] for wv in Wv]     # proj = (Wv[1], -Wv[0])
    zc = [wv[2] for wv in Wv]
    # for each hull edge, the unit-normal jet; flags for ALL 8 vertices (a tolerance-free superset of
    # the certificate's on-edge flags -> its sup is a sound upper bound on the paper's A_δ).
    flags = []
    for (p_, r_, sign) in struct:
        dq = [q[r_][0]-q[p_][0], q[r_][1]-q[p_][1]]
        rawx = J2(FI(0.0))-dq[1]; rawy = dq[0]             # raw n=(-dy,dx)
        nx, ny = junit2d(rawx, rawy)                       # tight unit-normal 2-jet
        if sign < 0: nx = J2(FI(0.0))-nx; ny = J2(FI(0.0))-ny
        for j in range(8):
            Jq = nx*(J2(FI(0.0))-q[j][1]) + ny*q[j][0]      # n·Jq_j, Jq=(-q1,q0)
            flags.append([zc[j]*nx, zc[j]*ny, Jq, nx, ny])
    return flags, nfl

def cell_Adelta(dlo, dhi, plo, phi, struct=None):
    """rigorous sup|d²r_P/dδ²| over the δ×φ cell (max |dd| across flags & 5 components). None if
    structure degenerate / mismatched (caller subdivides or skips as a transition window)."""
    res = flags_jet(dlo, dhi, plo, phi, struct)
    if res is None: return None
    flags, _ = res
    m = 0.0
    for fl in flags:
        for comp in fl:
            m = max(m, abs(comp.dd.lo), abs(comp.dd.hi))
    return m

if __name__ == '__main__':
    # smoke test at one benign cell
    a = cell_Adelta(5.5e-3, 5.6e-3, 1.0, 1.01)
    print('cell A_δ enclosure (δ~5.5e-3, φ~1.0):', a)
