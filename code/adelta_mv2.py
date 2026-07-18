"""adelta_mv2.py -- BIVARIATE (delta,phi) mean-value bound on A_delta, for the low-delta pentA/pentB
short-edge region where the delta-only mean-value (adelta_mv) fails (cancellation in BOTH directions).

Over a cell [dc+-dhw] x [pc+-phw]:
    D2(d,p) in D2(dc,pc)  (+)  [d_delta D2 over cell]*[-dhw,dhw]  (+)  [d_phi D2 over cell]*[-phw,phw]
(rigorous gradient/mean-value bound).  D2(dc,pc) is a POINT eval (tight, no cancellation).  The two
correction slopes are LOOSE (they only multiply the small half-widths).  d_phi D2 is obtained by
carrying a phi-DUAL coefficient (value, d/dphi) through a degree-3 delta-jet J3D: one dual evaluation
over the cell yields  D2=.dd.v, d_delta D2=.ddd.v, d_phi D2=.dd.dp.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_interval as F
from fast_interval import FI
import adelta_mv as MV
from adelta_mv import jdot3, jcross3, jnorm3, jnormalize3, flags_jet3
from adelta_jet import VERTS, R2, flag_struct

Z = F.FI(0.0)

class Dual:
    """(value, d/dphi) with FI components."""
    __slots__ = ('v', 'dp')
    def __init__(s, v, dp=None):
        if isinstance(v, Dual):
            s.v = v.v; s.dp = v.dp if dp is None else dp; return
        s.v = v; s.dp = Z if dp is None else dp
    @staticmethod
    def _c(o):
        return o if isinstance(o, Dual) else Dual(o)
    def __add__(s, o):
        o = Dual._c(o); return Dual(s.v + o.v, s.dp + o.dp)
    def __sub__(s, o):
        o = Dual._c(o); return Dual(s.v - o.v, s.dp - o.dp)
    def __mul__(s, o):
        o = Dual._c(o); return Dual(s.v * o.v, s.dp * o.v + s.v * o.dp)
    def __truediv__(s, o):
        o = Dual._c(o); q = s.v / o.v; return Dual(q, (s.dp - q * o.dp) / o.v)
    def sqr(s):
        return Dual(s.v.sqr(), F.FI(2.0) * s.v * s.dp)
    def sqrt(s):
        r = s.v.sqrt(); return Dual(r, s.dp / (F.FI(2.0) * r))

ZD = Dual(FI(0.0)); TWO = Dual(FI(2.0)); THREE = Dual(FI(3.0)); SIX = Dual(FI(6.0))

class J3D:
    """degree-3 delta-jet with Dual (phi-carrying) coefficients."""
    __slots__ = ('v', 'd', 'dd', 'ddd')
    def __init__(s, v, d=None, dd=None, ddd=None):
        s.v = Dual._c(v); s.d = ZD if d is None else Dual._c(d)
        s.dd = ZD if dd is None else Dual._c(dd); s.ddd = ZD if ddd is None else Dual._c(ddd)
    def __add__(s, o):
        if not isinstance(o, J3D): o = J3D(o)
        return J3D(s.v + o.v, s.d + o.d, s.dd + o.dd, s.ddd + o.ddd)
    def __sub__(s, o):
        if not isinstance(o, J3D): o = J3D(o)
        return J3D(s.v - o.v, s.d - o.d, s.dd - o.dd, s.ddd - o.ddd)
    def __mul__(s, o):
        if not isinstance(o, J3D): o = J3D(o)
        v = s.v * o.v
        d = s.d * o.v + s.v * o.d
        dd = s.dd * o.v + TWO * s.d * o.d + s.v * o.dd
        ddd = s.ddd * o.v + THREE * s.dd * o.d + THREE * s.d * o.dd + s.v * o.ddd
        return J3D(v, d, dd, ddd)
    def __truediv__(s, o):
        if not isinstance(o, J3D): o = J3D(o)
        v = s.v / o.v
        d = (s.d - v * o.d) / o.v
        dd = (s.dd - TWO * d * o.d - v * o.dd) / o.v
        ddd = (s.ddd - THREE * dd * o.d - THREE * d * o.dd - v * o.ddd) / o.v
        return J3D(v, d, dd, ddd)
    def sqrt(s):
        r = s.v.sqrt(); tr = TWO * r
        d = s.d / tr
        dd = (s.dd - TWO * d.sqr()) / tr
        ddd = (s.ddd - SIX * d * dd) / tr
        return J3D(r, d, dd, ddd)

E3J3D = [J3D(FI(0.0)), J3D(FI(0.0)), J3D(FI(1.0))]
def frame_jet3d(u):
    u = jnormalize3(u); f1 = jnormalize3(jcross3(E3J3D, u)); f2 = jcross3(u, f1); return [f1, f2, u]

def junit2d3d(mx, my):
    D = mx.v.sqr() + my.v.sqr(); L = D.sqrt()
    Dp = TWO * (mx.v * mx.d + my.v * my.d)
    Dpp = TWO * (mx.d.sqr() + mx.v * mx.dd + my.d.sqr() + my.v * my.dd)
    Dppp = TWO * (THREE * mx.d * mx.dd + mx.v * mx.ddd + THREE * my.d * my.dd + my.v * my.ddd)
    N = mx.v * my.d - my.v * mx.d
    Np = mx.v * my.dd - my.v * mx.dd
    Npp = mx.v * my.ddd - my.v * mx.ddd + mx.d * my.dd - my.d * mx.dd
    psip = N / D
    P = Np * D - N * Dp; psipp = P / D.sqr()
    Pp = Npp * D - N * Dpp; psippp = (Pp * D - TWO * P * Dp) / (D * D.sqr())
    hx = mx.v / L; hy = my.v / L
    hxp = ZD - psip * hy; hyp = psip * hx
    hxpp = (ZD - psip.sqr() * hx) - psipp * hy; hypp = (ZD - psip.sqr() * hy) + psipp * hx
    t3 = psippp - psip * psip.sqr()
    hxppp = (ZD - THREE * psip * psipp * hx) - t3 * hy
    hyppp = (ZD - THREE * psip * psipp * hy) + t3 * hx
    return J3D(hx, hxp, hxpp, hxppp), J3D(hy, hyp, hypp, hyppp)

def u2_jet3d(dlo, dhi, plo, phi):
    cd, sd = F.trig_fi(dlo, dhi); cp, sp = F.trig_fi(plo, phi)
    cosd = J3D(Dual(cd), Dual(Z - sd), Dual(Z - cd), Dual(sd))    # d/dphi = 0 (delta,phi independent)
    sind = J3D(Dual(sd), Dual(cd), Dual(Z - sd), Dual(Z - cd))
    cpr = J3D(Dual(cp * R2, Z - sp * R2))                         # v=cp*r2, d/dphi=-sp*r2
    spj = J3D(Dual(sp, cp))                                       # v=sp, d/dphi=cp
    r2 = Dual(R2)
    ux = cosd * r2 + sind * cpr
    uy = cosd * (Dual(Z) - r2) + sind * cpr
    uz = sind * spj
    return [ux, uy, uz]

def flags_jet3d(dlo, dhi, plo, phi, struct):
    u2 = u2_jet3d(dlo, dhi, plo, phi); W = frame_jet3d(u2)
    Wv = [[jdot3(W[r], [J3D(x) for x in V]) for r in range(3)] for V in VERTS]
    q = [[wv[1], J3D(FI(0.0)) - wv[0]] for wv in Wv]
    zc = [wv[2] for wv in Wv]
    flags = []
    for (p_, r_, sign) in struct:
        dq = [q[r_][0] - q[p_][0], q[r_][1] - q[p_][1]]
        rawx = J3D(FI(0.0)) - dq[1]; rawy = dq[0]
        nx, ny = junit2d3d(rawx, rawy)
        if sign < 0: nx = J3D(FI(0.0)) - nx; ny = J3D(FI(0.0)) - ny
        for j in range(8):
            Jq = nx * (J3D(FI(0.0)) - q[j][1]) + ny * q[j][0]
            flags.append([zc[j] * nx, zc[j] * ny, Jq, nx, ny])
    return flags

def cell_Adelta_mv2(dlo, dhi, plo, phi, struct=None):
    """rigorous sup|d^2 r_P/ddelta^2| over the cell via the BIVARIATE mean-value form."""
    dc = 0.5 * (dlo + dhi); dhw = 0.5 * (dhi - dlo)
    pc = 0.5 * (plo + phi); phw = 0.5 * (phi - plo)
    if struct is None:
        s = flag_struct(dc, pc)
        if s is None: return None
        struct = s[0]
    cres = flags_jet3(dc, dc, pc, pc, struct)      # tight center-point D2 (FI delta-jet at a point)
    if cres is None: return None
    dflags = flags_jet3d(dlo, dhi, plo, phi, struct)
    dI = FI(-dhw, dhw); pI = FI(-phw, phw)
    m = 0.0
    for fc, fd in zip(cres, dflags):
        for c2, cd in zip(fc, fd):
            val = c2.dd + cd.ddd.v * dI + cd.dd.dp * pI   # D2(dc,pc) + d_delta*dhw + d_phi*phw
            m = max(m, abs(val.lo), abs(val.hi))
    return m

if __name__ == '__main__':
    d = 1.5e-3
    print('bivariate mean-value in the low-delta pentA short-edge region (true ~4-5):')
    for phi, dhw, phw in [(0.1, 1e-6, 5e-3), (0.1, 1e-7, 5e-3), (0.3, 1e-7, 5e-3), (0.5, 1e-7, 2e-3)]:
        e = cell_Adelta_mv2(d - dhw, d + dhw, phi - phw, phi + phw)
        print(f'  phi={phi} dhw={dhw:.0e} phw={phw:.0e}: {e}')
