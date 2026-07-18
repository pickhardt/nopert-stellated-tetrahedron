"""aan.py -- n-symbol affine arithmetic (outward-rounded), generalizing g_deriv_aa_pure.AA.

Form: x = c + sum_k coef[k]*eps_k + r*eps_fresh, eps in [-1,1], r>=0 lumped remainder.
Every op rounds OUTWARD: centers/coefs are float ops whose rounding gap is charged to r
(gap(z) = ru(z)-rd(z) bounds the round-to-nearest error); nonlinear ops use a
center-tangent form with a rigorous Lagrange second-order remainder (same scheme as
g_deriv_aa_pure._cheb, proven sound there).  to_FI() is a proven enclosure.

Soundness contract: if X = expr(AAN inputs) then the real value of expr over the whole
cell lies in X.to_FI().  In particular X.hi() < 0 proves sup_cell expr < 0.
"""
import math
import fast_interval as F
from fast_interval import FI, rd, ru

def _gap(z):
    return ru(ru(z) - rd(z))

def _usum(xs):
    s = 0.0
    for x in xs: s = ru(s + x)
    return s

class AAN:
    __slots__ = ('c', 'co', 'r')   # co: tuple of n floats
    N = 5
    def __init__(s, c, co=None, r=0.0):
        s.c = float(c)
        s.co = tuple(co) if co is not None else (0.0,)*AAN.N
        s.r = r
    @staticmethod
    def sym(c, k, hw):
        """the affine form c + hw*eps_k (exact if c,hw floats)."""
        co = [0.0]*AAN.N; co[k] = float(hw)
        return AAN(float(c), co, 0.0)
    @staticmethod
    def const(x): return AAN(float(x), None, 0.0)
    @staticmethod
    def from_FI(fi):
        c = 0.5*(fi.lo + fi.hi)
        r = ru(max(ru(fi.hi - c), ru(c - fi.lo)))
        return AAN(c, None, r)
    def rad(s):
        return _usum(tuple(abs(x) for x in s.co) + (s.r,))
    def to_FI(s):
        hw = s.rad()
        return FI(rd(s.c - hw), ru(s.c + hw))
    def hi(s): return s.to_FI().hi
    def lo(s): return s.to_FI().lo
    # ---- linear ops ----
    def __add__(a, b):
        if type(b) is not AAN:
            c = a.c + float(b); return AAN(c, a.co, _usum((a.r, _gap(c))))
        c = a.c + b.c
        co = tuple(x + y for x, y in zip(a.co, b.co))
        r = _usum((a.r, b.r, _gap(c)) + tuple(_gap(x + y) for x, y in zip(a.co, b.co)))
        return AAN(c, co, r)
    __radd__ = __add__
    def __neg__(a): return AAN(-a.c, tuple(-x for x in a.co), a.r)
    def __sub__(a, b):
        if type(b) is not AAN:
            c = a.c - float(b); return AAN(c, a.co, _usum((a.r, _gap(c))))
        return a.__add__(b.__neg__())
    def __rsub__(a, b): return (-a).__add__(b)
    def __mul__(a, b):
        if type(b) is not AAN:
            k = float(b); c = a.c*k
            co = tuple(x*k for x in a.co)
            r = _usum((ru(abs(k)*a.r), _gap(c)) + tuple(_gap(x*k) for x in a.co))
            return AAN(c, co, r)
        xc, yc = a.c, b.c
        c = xc*yc
        co = tuple(xc*y + yc*x for x, y in zip(a.co, b.co))
        radx = a.rad(); rady = b.rad()
        gaps = [_gap(c)]
        for x, y in zip(a.co, b.co):
            gaps += [_gap(xc*y), _gap(yc*x), _gap(xc*y + yc*x)]
        r = _usum((ru(abs(xc)*b.r), ru(abs(yc)*a.r), ru(radx*rady)) + tuple(gaps))
        return AAN(c, co, r)
    __rmul__ = __mul__
    def sqr(s):
        return s.__mul__(s)   # (mildly loose vs dedicated sqr; sound)
    def _cheb(s, fc, p0i, d2max):
        """tangent form: f(x) = f(xc) + f'(xc)(x-xc) + R, |R|<=d2max/2*rad^2 (Lagrange).
        fc: FI enclosing f(xc); p0i: FI enclosing f'(xc); d2max >= sup_cell |f''|."""
        p0 = 0.5*(p0i.lo + p0i.hi); p0r = ru(max(ru(p0i.hi - p0), ru(p0 - p0i.lo)))
        cc = 0.5*(fc.lo + fc.hi); cr = ru(max(ru(fc.hi - cc), ru(cc - fc.lo)))
        radx = s.rad()
        co = tuple(p0*x for x in s.co)
        curv = ru(0.5*ru(d2max*ru(radx*radx)))
        r = _usum((cr, ru(abs(p0)*s.r), ru(p0r*radx), curv) + tuple(_gap(p0*x) for x in s.co))
        return AAN(cc, co, r)
    def cos(s):
        fi = s.to_FI()
        fc = F.trig_fi(s.c, s.c)[0]
        p0 = FI(0.0) - F.trig_fi(s.c, s.c)[1]
        cc = F.trig_fi(fi.lo, fi.hi)[0]
        d2 = max(abs(cc.lo), abs(cc.hi))
        return s._cheb(fc, p0, d2)
    def sin(s):
        fi = s.to_FI()
        fc = F.trig_fi(s.c, s.c)[1]
        p0 = F.trig_fi(s.c, s.c)[0]
        ss = F.trig_fi(fi.lo, fi.hi)[1]
        d2 = max(abs(ss.lo), abs(ss.hi))
        return s._cheb(fc, p0, d2)

# ---------- series f(z) = sinc(sqrt z) = sum (-z)^k/(2k+1)!  and g(z) = (1-cos sqrt z)/z ----------
# For z in [0, Z] both are alternating series with terms decreasing once k! dominates;
# remainder after truncation at k=K is bounded by the first omitted term at z=Z (valid for
# Z <= 20, far above our use Z <= 0.1).  Implemented with Horner in AAN + interval tail.
_FCOEF = [1.0, -1/6, 1/120, -1/5040, 1/362880]          # z^0..z^4 of sinc(sqrt z)
_GCOEF = [0.5, -1/24, 1/720, -1/40320, 1/3628800]       # z^0..z^4 of (1-cos sqrt z)/z
_FTAIL = 1/39916800.0    # |z|^5 coefficient bound (1/11!)
_GTAIL = 1/479001600.0   # 1/12!

def _series(z, coef, tailc):
    zfi = z.to_FI()
    assert zfi.lo >= -1e-12 and zfi.hi <= 1.0, "series domain guard"
    acc = AAN.const(coef[-1])
    for c in reversed(coef[:-1]):
        acc = acc*z + c
    Z = max(abs(zfi.lo), abs(zfi.hi))
    tail = ru(1.1 * ru(tailc * ru(Z**5)))   # 1.1: geometric tail slack (Z<=1)
    return AAN(acc.c, acc.co, _usum((acc.r, tail)))

def sinc_sq(z):   return _series(z, _FCOEF, _FTAIL)
def oneminuscos_over(z): return _series(z, _GCOEF, _GTAIL)

def rot_from_vec(eta):
    """exp([eta]x) for a 3-vector of AAN (any norm): I + f(z)[eta] + g(z)(eta etaT - z I), z=|eta|^2."""
    z = eta[0]*eta[0] + eta[1]*eta[1] + eta[2]*eta[2]
    f = sinc_sq(z); g = oneminuscos_over(z)
    ex, ey, ez = eta
    # skew part f*[eta]
    R = [[None]*3 for _ in range(3)]
    outer = [[ex*ex, ex*ey, ex*ez], [ey*ex, ey*ey, ey*ez], [ez*ex, ez*ey, ez*ez]]
    sk = [[AAN.const(0), -f*ez, f*ey], [f*ez, AAN.const(0), -f*ex], [-f*ey, f*ex, AAN.const(0)]]
    for i in range(3):
        for j in range(3):
            t = sk[i][j] + g*outer[i][j]
            if i == j: t = t + 1.0 - g*z
            R[i][j] = t
    return R

def matvec(R, v):
    return [R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
            R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
            R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]]
