"""Affine-arithmetic forward-mode AD for the u2-Lipschitz constant of G_lambda.

Motivation (see rupert-stellated-tetrahedron.md "Λ-TIGHTENING MORNING"): g_deriv.py's DI
dual-INTERVAL AD is tight at a point but WRAPS over a (theta,phi)-cell (Lambda_th ~5 at hw=0.01
vs the true ~0.4, a ~13x over-estimate) because the normalize divisions and the ci-min sub-gradient
hull decorrelate the interval endpoints.  Here we replace the interval representation with AFFINE
ARITHMETIC: every scalar carries its FIRST-ORDER dependence on the two cell noise-symbols

    eps_th, eps_ph in [-1,1],    theta = th + thw*eps_th,   phi = ph + phw*eps_ph,

as an exact affine form  x = c + cth*eps_th + cph*eps_ph + r*eps_extra,  where c is the central
float value, cth,cph are the (float) coefficients of the SHARED cell symbols, and r>=0 is a lumped
non-negative remainder (rounding + linearization error) carried on a fresh independent symbol.
Linear/correlated sub-expressions then CANCEL in cth,cph instead of wrapping (e.g. cos^2+sin^2 -> 1
to first order), while r only ever grows by genuine second-order/rounding terms.  Because we only
ever create the two primary symbols (constants become pure-remainder forms), the tracked part is
exactly (cth,cph): a 4-float "first-order Taylor model" in (theta,phi) -- cheap and non-wrapping.

Soundness.  Every op rounds OUTWARD (directed rounding on the center/coeff floats, upward on the
remainder) and every nonlinear op uses an interval mean-value form  f(u) in f(xc) + f'([a,b])*(u-xc)
(MVT, rigorous across extrema since f'([a,b]) is an outward interval enclosure).  Concretizing an
affine form to an interval, x.to_FI() = [c - R, c + R] with R = |cth|+|cph|+r (outward), is a proven
enclosure of the whole cell.  So Lambda_th = max(|G.dth.lo|,|G.dth.hi|) is a PROVEN upper bound on
sup_cell |dG_lambda/dtheta| -- exactly the requirement (a positive G.lo is a proof, so Lambda must
never under-estimate).  The one non-smooth node (ci = min_v <n,v>) keeps g_deriv's sub-gradient hull,
but tightened: if a single vertex is provably the unique minimizer over the cell (all others' value
lower bounds exceed the min of the upper bounds) the whole affine derivative is carried through with
NO hull loss; only genuinely-ambiguous cells fall back to the interval hull.  The rotation is
evaluated at the s/dt/dp CENTER (const in u2) -- the telescoping-mean-value argument of g_deriv is
preserved verbatim: certify's sqrt3 * s/dt/dp terms carry the rotation-cell variation, so Lth,Lph
need only bound |dG/dparam| over the (theta,phi)-cell at rotation-center.
"""
import math
import fast_interval as F
from fast_interval import FI, rd, ru
from math import sqrt as _sqrt

def _gap(z):
    """rigorous bound on the round-to-nearest error of the float result z of one real op:
    the true value lies in [rd(z),ru(z)] and so does z, hence |z-true| <= ru(z)-rd(z)."""
    return ru(ru(z) - rd(z))
def _usum(xs):
    """upward-rounded sum of non-negative floats (a rigorous upper bound on their real sum)."""
    s = 0.0
    for x in xs: s = ru(s + x)
    return s

class AA:
    """affine form c + cth*eps_th + cph*eps_ph + r*eps_extra, r>=0 lumped remainder (fresh symbol)."""
    __slots__ = ('c', 'cth', 'cph', 'r')
    def __init__(s, c, cth=0.0, cph=0.0, r=0.0):
        s.c = c; s.cth = cth; s.cph = cph; s.r = r
    @staticmethod
    def const(x): return AA(float(x), 0.0, 0.0, 0.0)
    @staticmethod
    def from_FI(fi):
        """enclose the interval fi by a symmetric pure-remainder form (independent uncertainty)."""
        c = 0.5 * (fi.lo + fi.hi)
        r = ru(max(ru(fi.hi - c), ru(c - fi.lo)))
        return AA(c, 0.0, 0.0, r)
    def rad(s):  # upward total half-width contribution of the deviation part
        return _usum((abs(s.cth), abs(s.cph), s.r))
    def to_FI(s):
        hw = s.rad()
        return FI(rd(s.c - hw), ru(s.c + hw))
    # ---- linear ops (exact in the shared symbols; rounding -> remainder) ----
    def __add__(a, b):
        if type(b) is not AA:
            c = a.c + float(b); return AA(c, a.cth, a.cph, _usum((a.r, _gap(c))))
        c = a.c + b.c; cth = a.cth + b.cth; cph = a.cph + b.cph
        r = _usum((a.r, b.r, _gap(c), _gap(cth), _gap(cph)))
        return AA(c, cth, cph, r)
    __radd__ = __add__
    def __neg__(a): return AA(-a.c, -a.cth, -a.cph, a.r)
    def __sub__(a, b):
        if type(b) is not AA:
            c = a.c - float(b); return AA(c, a.cth, a.cph, _usum((a.r, _gap(c))))
        c = a.c - b.c; cth = a.cth - b.cth; cph = a.cph - b.cph
        r = _usum((a.r, b.r, _gap(c), _gap(cth), _gap(cph)))
        return AA(c, cth, cph, r)
    def __rsub__(a, b): return (-a).__add__(b)
    def __mul__(a, b):
        if type(b) is not AA:                       # scalar float multiply
            k = float(b); c = a.c * k; cth = a.cth * k; cph = a.cph * k
            r = _usum((ru(abs(k) * a.r), _gap(c), _gap(cth), _gap(cph)))
            return AA(c, cth, cph, r)
        xc, yc = a.c, b.c
        c = xc * yc
        cth = xc * b.cth + yc * a.cth
        cph = xc * b.cph + yc * a.cph
        radx = a.rad(); rady = b.rad()
        # linear-in-remainder cross terms + the second-order deviation product + rounding
        r = _usum((ru(abs(xc) * b.r), ru(abs(yc) * a.r), ru(radx * rady),
                   _gap(c),
                   _gap(xc * b.cth), _gap(yc * a.cth), _gap(cth),
                   _gap(xc * b.cph), _gap(yc * a.cph), _gap(cph)))
        return AA(c, cth, cph, r)
    __rmul__ = __mul__
    def _cheb(s, fc, p0i, d2max):
        """tangent (center-slope) affine form of a smooth unary f: linearize at the CENTER slope
        f'(xc)=p0i (point interval) and bound only the true 2nd-order term.  By Lagrange,
        f(u) = f(xc) + f'(xc)(u-xc) + 1/2 f''(zeta)(u-xc)^2, so the remainder <= 1/2*sup|f''|*radx^2.
        This is ~2x tighter than dumping the whole slope-interval width into the remainder, and is
        rigorous provided d2max >= sup_[a,b]|f''| and p0i encloses f'(xc)."""
        p0 = 0.5 * (p0i.lo + p0i.hi); p0r = ru(max(ru(p0i.hi - p0), ru(p0 - p0i.lo)))
        cc = 0.5 * (fc.lo + fc.hi); cr = ru(max(ru(fc.hi - cc), ru(cc - fc.lo)))
        radx = s.rad(); cth = p0 * s.cth; cph = p0 * s.cph
        curv = ru(0.5 * ru(d2max * ru(radx * radx)))
        r = _usum((cr, ru(abs(p0) * s.r), ru(p0r * radx), curv, _gap(cth), _gap(cph)))
        return AA(cc, cth, cph, r)
    def sqr(s):
        xc = s.c; two = xc + xc
        c = xc * xc; cth = two * s.cth; cph = two * s.cph; radx = s.rad()
        r = _usum((ru(abs(two) * s.r), ru(radx * radx), _gap(c), _gap(cth), _gap(cph)))
        return AA(c, cth, cph, r)
    def sqrt(s):
        fi = s.to_FI(); a, b = fi.lo, fi.hi
        if a <= 0.0:                                # degenerate: fall back to interval sqrt (sound)
            return AA.from_FI(fi.sqrt())
        fc = FI(rd(_sqrt(s.c)) if s.c > 0 else 0.0, ru(_sqrt(s.c)))
        p0 = FI(0.5) / FI(rd(_sqrt(s.c)), ru(_sqrt(s.c)))   # f'(xc)=1/(2 sqrt xc)
        a15 = (FI(a, a).sqrt() * FI(a, a)).lo       # rigorous lower bound on a^1.5 (a>0)
        d2 = ru(0.25 / a15)                         # sup|f''|=1/(4 u^1.5) at u=a (smallest)
        return s._cheb(fc, p0, d2)
    def recip(s):
        fi = s.to_FI(); a, b = fi.lo, fi.hi
        if a <= 0.0 <= b: raise ZeroDivisionError('AA.recip over interval containing 0')
        fc = FI(1.0) / FI(s.c, s.c)
        p0 = FI(0.0) - (FI(1.0) / FI(s.c, s.c).sqr())       # f'(xc)=-1/xc^2
        m = min(abs(a), abs(b))
        m3 = (FI(m, m).sqr() * FI(m, m)).lo         # rigorous lower bound on |u|_min^3
        d2 = ru(2.0 / m3)                           # sup|f''|=2/|u|^3 at min|u|
        return s._cheb(fc, p0, d2)
    def __truediv__(a, b):
        if type(b) is not AA: return a.__mul__(1.0 / float(b))
        return a.__mul__(b.recip())
    def cos(s):
        fi = s.to_FI()
        fc = F.trig_fi(s.c, s.c)[0]
        p0 = FI(0.0) - F.trig_fi(s.c, s.c)[1]       # f'(xc)=-sin(xc)
        cc = F.trig_fi(fi.lo, fi.hi)[0]; d2 = max(abs(cc.lo), abs(cc.hi))  # sup|cos| over cell = |f''|
        return s._cheb(fc, p0, d2)
    def sin(s):
        fi = s.to_FI()
        fc = F.trig_fi(s.c, s.c)[1]
        p0 = F.trig_fi(s.c, s.c)[0]                 # f'(xc)=cos(xc)
        ss = F.trig_fi(fi.lo, fi.hi)[1]; d2 = max(abs(ss.lo), abs(ss.hi))  # sup|sin| over cell = |f''|
        return s._cheb(fc, p0, d2)

# ---------------- dual affine: value + partials d/dtheta, d/dphi, each an AA over the cell ----------
_AA0 = AA.const(0.0)
class DA:
    __slots__ = ('v', 'dth', 'dph')
    def __init__(s, v, dth=None, dph=None):
        s.v = v if type(v) is AA else (AA.from_FI(v) if type(v) is FI else AA.const(v))
        s.dth = dth if dth is not None else AA.const(0.0)
        s.dph = dph if dph is not None else AA.const(0.0)
    @staticmethod
    def const(x):
        v = AA.from_FI(x) if type(x) is FI else AA.const(x)
        return DA(v, AA.const(0.0), AA.const(0.0))
    def __add__(a, b):
        b = b if type(b) is DA else DA.const(b); return DA(a.v + b.v, a.dth + b.dth, a.dph + b.dph)
    __radd__ = __add__
    def __neg__(a): return DA(-a.v, -a.dth, -a.dph)
    def __sub__(a, b):
        b = b if type(b) is DA else DA.const(b); return DA(a.v - b.v, a.dth - b.dth, a.dph - b.dph)
    def __rsub__(a, b): return (b if type(b) is DA else DA.const(b)).__sub__(a)
    def __mul__(a, b):
        b = b if type(b) is DA else DA.const(b)
        return DA(a.v * b.v, a.dth * b.v + a.v * b.dth, a.dph * b.v + a.v * b.dph)  # product rule
    __rmul__ = __mul__
    def __truediv__(a, b):
        b = b if type(b) is DA else DA.const(b)
        g2 = b.v * b.v
        return DA(a.v / b.v, (a.dth * b.v - a.v * b.dth) / g2, (a.dph * b.v - a.v * b.dph) / g2)
    def sqr(a): return DA(a.v.sqr(), (a.v * a.dth) * 2.0, (a.v * a.dph) * 2.0)
    def sqrt(a):
        r = a.v.sqrt(); tworr = r * 2.0
        return DA(r, a.dth / tworr, a.dph / tworr)

# ---------------- DA vector ops (mirror g_deriv / fast_interval) ----------------
def dvdot(u, v):
    r = DA.const(0.0)
    for a, b in zip(u, v): r = r + a * b
    return r
def dvcross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
def dvnorm(u):
    s = DA.const(0.0)
    for c in u: s = s + c.sqr()
    return s.sqrt()
def dvnormalize(u):
    n = dvnorm(u); return [a / n for a in u]
def dvproj(w): return [w[1], DA.const(0.0) - w[0]]
_E3 = [DA.const(0.0), DA.const(0.0), DA.const(1.0)]
def dvframe(u):
    u = dvnormalize(u); f1 = dvnormalize(dvcross(_E3, u)); f2 = dvcross(u, f1); return [f1, f2, u]
def dvapplyW(W, v): return [dvdot(W[0], v), dvdot(W[1], v), dvdot(W[2], v)]
def dvmat(Rfi, dv):
    """apply constant FI matrix Rfi to a DA vector (rotation is const in u2 -> rotate each partial)."""
    out = []
    for i in range(3):
        acc = DA.const(0.0)
        for j in range(3): acc = acc + DA.const(Rfi[i][j]) * dv[j]
        out.append(acc)
    return out
def _rot_mat(xi):
    """materialize the constant rotation e^[xi] (xi FI list) as a 3x3 FI matrix R[i][j]."""
    R = F.rodrigues(xi)
    E = [[FI(1.0), FI(0.0), FI(0.0)], [FI(0.0), FI(1.0), FI(0.0)], [FI(0.0), FI(0.0), FI(1.0)]]
    cols = [R(E[j]) for j in range(3)]
    return [[cols[j][i] for j in range(3)] for i in range(3)]

def u2_aa(box):
    """u2 and its (theta,phi) partials as DA, with theta,phi carried as the primary noise symbols."""
    th, thw, ph, phw = box[0], box[1], box[2], box[3]
    TH = AA(th, thw, 0.0, 0.0); PH = AA(ph, 0.0, phw, 0.0)
    ct = TH.cos(); st = TH.sin(); cp = PH.cos(); sp = PH.sin()
    Z = AA.const(0.0)
    # u=(st cp, st sp, ct); du/dth=(ct cp, ct sp, -st); du/dph=(-st sp, st cp, 0)
    return [DA(st * cp, ct * cp, Z - st * sp),
            DA(st * sp, ct * sp, st * cp),
            DA(ct, Z - st, Z)]

def glam_G(box, eo, lam, aa):
    """the DA G_lambda over the cell (value + d/dtheta,d/dphi as affine forms).  Rotation at center."""
    th, thw, ph, phw, dt, dtw, dp, dpw, s0, sw = box
    u2 = u2_aa(box); W = dvframe(u2); V = F.verts(aa)
    qo = [dvproj(dvapplyW(W, [DA.const(x) for x in v])) for v in V]
    nv = len(V)
    # qo_k - qo_j, formed by subtracting the already-projected DA points (proj is linear).  Only the
    # tiny frame-normalization remainder adds; crucially the edge-normal n's LARGE normalization
    # remainder is applied ONCE downstream (n . (qo_k-qo_j)) instead of twice as in (n.qo_k)-(n.qo_j),
    # which is what keeps the active-set test tight -- see the hull-corruption diagnosis.
    def _dq(k, j):
        return [qo[k][0] - qo[j][0], qo[k][1] - qo[j][1]]
    edges = []
    for (p_, r_, sgn) in eo:
        d = [qo[r_][0] - qo[p_][0], qo[r_][1] - qo[p_][1]]; n = [DA.const(0.0) - d[1], d[0]]
        if sgn < 0: n = [-n[0], -n[1]]
        nn = (n[0].sqr() + n[1].sqr()).sqrt(); n = [x / nn for x in n]
        # ci = min_v <n,v>.  value = interval min; derivative = sub-gradient hull over active vertices,
        # but if a SINGLE vertex is provably the unique minimizer over the cell, carry it exactly (AA).
        # Active set via CORRELATED pairwise differences: vertex k is inactive iff some j has
        # (val_k - val_j).hi_bound... i.e. val_k > val_j over the WHOLE cell ((val_k-val_j).lo>0).
        # The affine difference cancels the shared theta,phi dependence -> a far tighter (fewer false-
        # active) set than comparing each interval to min-of-highs, which decorrelates the endpoints.
        vals = [dvdot(n, v) for v in qo]
        fis = [dv.v.to_FI() for dv in vals]
        civ_hi = min(f.hi for f in fis); civ_lo = min(f.lo for f in fis)
        active = []
        for k in range(nv):
            dominated = False
            for j in range(nv):
                # k inactive iff <n,qo_k - qo_j> > 0 over the WHOLE cell (k strictly above j everywhere).
                if j != k and dvdot(n, _dq(k, j)).v.to_FI().lo > 0.0: dominated = True; break
            if not dominated: active.append(k)
        if len(active) == 1:
            ci = vals[active[0]]                     # exact: no hull loss (full first-order correlation)
        else:
            dths = [vals[k].dth.to_FI() for k in active]; dphs = [vals[k].dph.to_FI() for k in active]
            tl = min(f.lo for f in dths); th_ = max(f.hi for f in dths)
            pl = min(f.lo for f in dphs); ph_ = max(f.hi for f in dphs)
            ci = DA(AA.from_FI(FI(civ_lo, civ_hi)), AA.from_FI(FI(tl, th_)), AA.from_FI(FI(pl, ph_)))
        edges.append((n, ci))
    # rotation at s/dt/dp CENTER (const in u2) -- telescoping soundness note preserved (see module doc)
    dirI = F.idir(dt, dt, dp, dp); Rc = _rot_mat([FI(s0) * x for x in dirI])
    wc = [dvmat(Rc, dvapplyW(W, [DA.const(x) for x in v])) for v in V]
    G = DA.const(0.0)
    for k, w in lam:
        ei, vj = k // 8, k % 8; n, ci = edges[ei]
        gc = ci - dvdot(n, dvproj(wc[vj]))
        G = G + DA.const(float(w)) * gc              # drop O(hs) dg term (handled by certify's s-terms)
    return G

def glam_lip(box, eo, lam, aa):
    """Lambda_th, Lambda_ph = proven sup over the (theta,phi)-cell of |dG_lambda/dtheta|,|dG_lambda/dphi|."""
    G = glam_G(box, eo, lam, aa)
    dt_fi = G.dth.to_FI(); dp_fi = G.dph.to_FI()
    return max(abs(dt_fi.lo), abs(dt_fi.hi)), max(abs(dp_fi.lo), abs(dp_fi.hi))

if __name__ == '__main__':
    # -------- Gate 1: AA derivative matches finite differences at a point (scalar sanity) --------
    def f_aa(th, ph):
        TH = AA(th, 0.0, 0.0, 0.0); PH = AA(ph, 0.0, 0.0, 0.0)  # a point (no cell width)
        x = DA(TH.sin(), TH.cos(), AA.const(0.0))               # x=sin th
        y = DA(PH.cos(), AA.const(0.0), AA.const(0.0) - PH.sin())  # y=cos ph
        return (x * x + x * y).sqrt() + x / (y + DA.const(2.0))
    def f_float(th, ph):
        import math as m; x = m.sin(th); y = m.cos(ph)
        return m.sqrt(x * x + x * y) + x / (y + 2.0)
    print("Gate 1 (scalar AA-AD vs finite diff at a point):")
    for th, ph in [(1.3, 0.6), (0.9, 2.1), (1.55, 0.75)]:
        r = f_aa(th, ph); eps = 1e-7
        dth_fd = (f_float(th + eps, ph) - f_float(th - eps, ph)) / (2 * eps)
        dph_fd = (f_float(th, ph + eps) - f_float(th, ph - eps)) / (2 * eps)
        dth = r.dth.to_FI(); dph = r.dph.to_FI()
        print('  th=%.2f ph=%.2f: dth AA=[%.6f,%.6f] fd=%.6f | dph AA=[%.6f,%.6f] fd=%.6f' % (
            th, ph, dth.lo, dth.hi, dth_fd, dph.lo, dph.hi, dph_fd))
try:
    from g_aa_cy import AA as _AA_cy, _gap as _gap_cy, _usum as _usum_cy
    AA=_AA_cy; _gap=_gap_cy; _usum=_usum_cy; _AA_CY=True
except Exception: _AA_CY=False
try:
    from g_da_cy import DA as _DA_cy
    DA=_DA_cy
    _AA0=AA.const(0.0); _E3=[DA.const(0.0),DA.const(0.0),DA.const(1.0)]   # rebuild with cython DA/AA
    _DA_CY=True
except Exception: _DA_CY=False
