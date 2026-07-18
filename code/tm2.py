"""tm2.py -- degree-2 Taylor model in N=5 shared symbols, outward-rounded.

Form: x = c + sum_i L[i] eps_i + sum_{i<=j} Q[(i,j)] eps_i eps_j + r*[-1,1],
eps_i in [-1,1].  r >= 0 lumps all degree->=3 terms, rounding, and nonlinear-op
Lagrange remainders.  Soundness: every arithmetic op encloses the real result over
the whole cell; rounding gaps of every float op are charged upward to r
(gap(z)=ru(z)-rd(z) >= round-to-nearest error).  Range bound exploits eps_i^2 in [0,1]:
  sup <= c + sum|L| + sum_i max(Qii,0) + sum_{i<j}|Qij| + r   (and dually for inf).
This lets a concave (negative-curvature) hedged certificate function be bounded
above near its true sup -- the capability the degree-1 AA form lacks.
"""
import math
import fast_interval as F
from fast_interval import FI, rd, ru

N = 5
_IDX = [(i, j) for i in range(N) for j in range(i, N)]   # 15 pairs
_POS = {p: k for k, p in enumerate(_IDX)}

def _gap(z): return ru(ru(z) - rd(z))
def _usum(xs):
    s = 0.0
    for x in xs: s = ru(s + x)
    return s

Z5 = (0.0,)*N
Z15 = (0.0,)*15

class TM2:
    __slots__ = ('c', 'L', 'Q', 'r')
    def __init__(s, c, L=Z5, Q=Z15, r=0.0):
        s.c = float(c); s.L = tuple(L); s.Q = tuple(Q); s.r = r
    @staticmethod
    def sym(c, k, hw):
        L = [0.0]*N; L[k] = float(hw)
        return TM2(float(c), L, Z15, 0.0)
    @staticmethod
    def const(x): return TM2(float(x))
    @staticmethod
    def from_FI(fi):
        c = 0.5*(fi.lo + fi.hi)
        return TM2(c, Z5, Z15, ru(max(ru(fi.hi - c), ru(c - fi.lo))))
    # magnitudes
    def _bl(s): return _usum(tuple(abs(x) for x in s.L))
    def _bq(s): return _usum(tuple(abs(x) for x in s.Q))
    def rad(s): return _usum((s._bl(), s._bq(), s.r))
    def to_FI(s):
        # sharp on the quadratic diagonal: eps^2 in [0,1]
        bl = s._bl()
        qpos = 0.0; qneg = 0.0; qcross = 0.0
        for (i, j), v in zip(_IDX, s.Q):
            if i == j:
                if v > 0: qpos = ru(qpos + v)
                else:     qneg = ru(qneg - v)   # magnitude
            else:
                qcross = ru(qcross + abs(v))
        hi = ru(s.c + _usum((bl, qpos, qcross, s.r)))
        lo = rd(s.c - _usum((bl, qneg, qcross, s.r)))
        return FI(lo, hi)
    def hi(s): return s.to_FI().hi
    def lo(s): return s.to_FI().lo
    # ---- linear ----
    def __add__(a, b):
        if type(b) is not TM2:
            c = a.c + float(b); return TM2(c, a.L, a.Q, _usum((a.r, _gap(c))))
        c = a.c + b.c
        L = tuple(x + y for x, y in zip(a.L, b.L))
        Q = tuple(x + y for x, y in zip(a.Q, b.Q))
        r = _usum((a.r, b.r, _gap(c))
                  + tuple(_gap(x + y) for x, y in zip(a.L, b.L))
                  + tuple(_gap(x + y) for x, y in zip(a.Q, b.Q)))
        return TM2(c, L, Q, r)
    __radd__ = __add__
    def __neg__(a): return TM2(-a.c, tuple(-x for x in a.L), tuple(-x for x in a.Q), a.r)
    def __sub__(a, b):
        if type(b) is not TM2:
            c = a.c - float(b); return TM2(c, a.L, a.Q, _usum((a.r, _gap(c))))
        return a.__add__(b.__neg__())
    def __rsub__(a, b): return (-a).__add__(b)
    def __mul__(a, b):
        if type(b) is not TM2:
            k = float(b)
            c = a.c*k
            L = tuple(x*k for x in a.L); Q = tuple(x*k for x in a.Q)
            r = _usum((ru(abs(k)*a.r), _gap(c)) + tuple(_gap(x*k) for x in a.L)
                      + tuple(_gap(x*k) for x in a.Q))
            return TM2(c, L, Q, r)
        ca, cb = a.c, b.c
        c = ca*cb
        L = [0.0]*N; gaps = [_gap(c)]
        for i in range(N):
            v = ca*b.L[i] + cb*a.L[i]
            L[i] = v; gaps += [_gap(ca*b.L[i]), _gap(cb*a.L[i]), _gap(v)]
        Q = [0.0]*15
        for k, (i, j) in enumerate(_IDX):
            v = ca*b.Q[k] + cb*a.Q[k]
            gaps += [_gap(ca*b.Q[k]), _gap(cb*a.Q[k]), _gap(v)]
            if i == j:
                w = a.L[i]*b.L[i]
                v += w; gaps.append(_gap(w))
            else:
                w1 = a.L[i]*b.L[j]; w2 = a.L[j]*b.L[i]
                v += w1 + w2; gaps += [_gap(w1), _gap(w2), _gap(w1 + w2)]
            Q[k] = v; gaps.append(_gap(v))
        # remainder: all products of degree >= 3 plus cross-remainder terms
        bla, bqa, ra = a._bl(), a._bq(), a.r
        blb, bqb, rb = b._bl(), b._bq(), b.r
        maga = _usum((abs(ca), bla, bqa, ra))
        magb = _usum((abs(cb), blb, bqb, rb))
        rr = _usum((ru(bla*bqb), ru(blb*bqa), ru(bqa*bqb),
                    ru(ra*magb), ru(rb*maga)))
        r = _usum((rr,) + tuple(gaps))
        return TM2(c, L, Q, r)
    __rmul__ = __mul__
    # ---- nonlinear scalar via degree-2 Taylor at center with Lagrange R3 ----
    def _taylor2(s, f0, f1, f2, d3max):
        """f(x) enclosure: f0,f1,f2 FI enclosures of f(xc), f'(xc), f''(xc)/2;
        d3max >= sup_cell |f'''|/6 * rad^3 is charged by caller?  No: charged here:
        R3 <= d3max * rad(u)^3 with d3max >= sup|f'''|/6."""
        u = TM2(0.0, s.L, s.Q, s.r)          # deviation from center
        u2 = u*u
        f1c = 0.5*(f1.lo + f1.hi); f1r = ru(max(ru(f1.hi - f1c), ru(f1c - f1.lo)))
        f2c = 0.5*(f2.lo + f2.hi); f2r = ru(max(ru(f2.hi - f2c), ru(f2c - f2.lo)))
        f0c = 0.5*(f0.lo + f0.hi); f0r = ru(max(ru(f0.hi - f0c), ru(f0c - f0.lo)))
        radu = u.rad()
        out = u*f1c + u2*f2c + f0c
        extra = _usum((f0r, ru(f1r*radu), ru(f2r*ru(radu*radu)),
                       ru(d3max*ru(ru(radu*radu)*radu))))
        return TM2(out.c, out.L, out.Q, _usum((out.r, extra)))
    def sin(s):
        f0 = F.trig_fi(s.c, s.c)[1]
        f1 = F.trig_fi(s.c, s.c)[0]
        f2 = FI(0.0) - F.trig_fi(s.c, s.c)[1]*FI(0.5)
        return s._taylor2(f0, f1, f2, ru(1.0/6.0))     # |f'''| <= 1
    def cos(s):
        f0 = F.trig_fi(s.c, s.c)[0]
        f1 = FI(0.0) - F.trig_fi(s.c, s.c)[1]
        f2 = FI(0.0) - F.trig_fi(s.c, s.c)[0]*FI(0.5)
        return s._taylor2(f0, f1, f2, ru(1.0/6.0))

# ---------- Rodrigues series (polynomial in TM2; alternating tails) ----------
_FCOEF = [1.0, -1/6, 1/120, -1/5040, 1/362880]
_GCOEF = [0.5, -1/24, 1/720, -1/40320, 1/3628800]
_FTAIL = 1/39916800.0
_GTAIL = 1/479001600.0

def _series(z, coef, tailc, nonneg=False):
    zfi = z.to_FI()
    # When the caller guarantees z >= 0 (z is a sum of squares, e.g. |eta|^2 in
    # rot_from_vec), the TM2 range's lower bound can dip slightly negative purely from
    # enclosure slack even though true z >= 0.  Flooring lo at 0 is unconditionally sound
    # (it only tightens a provable fact) and keeps the alternating-series tail valid.
    lo_eff = max(zfi.lo, 0.0) if nonneg else zfi.lo
    assert zfi.hi <= 1.0 and lo_eff >= -1e-9, "series domain guard"
    acc = TM2.const(coef[-1])
    for c in reversed(coef[:-1]):
        acc = acc*z + c
    Z = max(abs(lo_eff), abs(zfi.hi))
    tail = ru(1.1*ru(tailc*ru(Z**5)))
    return TM2(acc.c, acc.L, acc.Q, _usum((acc.r, tail)))

def sinc_sq(z): return _series(z, _FCOEF, _FTAIL, nonneg=True)
def oneminuscos_over(z): return _series(z, _GCOEF, _GTAIL, nonneg=True)

def rot_from_vec(eta):
    z = eta[0]*eta[0] + eta[1]*eta[1] + eta[2]*eta[2]
    f = sinc_sq(z); g = oneminuscos_over(z)
    ex, ey, ez = eta
    outer = [[ex*ex, ex*ey, ex*ez], [ey*ex, ey*ey, ey*ez], [ez*ex, ez*ey, ez*ez]]
    sk = [[TM2.const(0), -(f*ez), f*ey], [f*ez, TM2.const(0), -(f*ex)],
          [-(f*ey), f*ex, TM2.const(0)]]
    R = [[None]*3 for _ in range(3)]
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
