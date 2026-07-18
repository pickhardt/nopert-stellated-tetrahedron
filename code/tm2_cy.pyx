# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# Cython port of tm2.py's TM2 (degree-2 Taylor model, N=5 symbols, 15 quadratic pairs).
# BIT-IDENTICAL to the pure version: every remainder mirrors _usum EXACTLY (s=0.0, then ru(s+term) in
# the pure term order — note ru(0+t)=ru(t) inflates the first term). Compile with -ffp-contract=off.
import math
from libc.math cimport nextafter, INFINITY, fabs, pow as _cpow
import fast_interval as F
from fast_interval import FI

cdef double _NINF = -INFINITY
cdef double _PINF = INFINITY
cdef inline double rd(double x): return nextafter(x, _NINF)
cdef inline double ru(double x): return nextafter(x, _PINF)
cdef inline double _gap(double z): return ru(ru(z) - rd(z))
cdef inline double _dmax(double a, double b): return a if a > b else b

cdef int[15] _II
cdef int[15] _JJ
cdef int _k = 0
cdef int _i, _j
for _i in range(5):
    for _j in range(_i, 5):
        _II[_k] = _i; _JJ[_k] = _j; _k += 1

cdef class TM2:
    cdef public double c, r
    cdef double L[5]
    cdef double Q[15]
    def __init__(self, c, L=None, Q=None, r=0.0):
        cdef int i
        self.c = <double>c; self.r = <double>r
        for i in range(5): self.L[i] = 0.0 if L is None else <double>L[i]
        for i in range(15): self.Q[i] = 0.0 if Q is None else <double>Q[i]
    @property
    def Lt(self): return tuple(self.L[i] for i in range(5))
    @property
    def Qt(self): return tuple(self.Q[i] for i in range(15))
    @staticmethod
    def sym(c, k, hw):
        cdef TM2 x = _new()
        x.c = <double>c; x.L[<int>k] = <double>hw
        return x
    @staticmethod
    def const(x):
        cdef TM2 t = _new()
        t.c = <double>x
        return t
    @staticmethod
    def from_FI(fi):
        cdef double lo = fi.lo
        cdef double hi = fi.hi
        cdef double c = 0.5*(lo+hi)
        cdef TM2 t = _new()
        t.c = c
        t.r = ru(_dmax(ru(hi-c), ru(c-lo)))
        return t
    cdef double _bl(self):
        cdef double s = 0.0
        cdef int i
        for i in range(5): s = ru(s + fabs(self.L[i]))
        return s
    cdef double _bq(self):
        cdef double s = 0.0
        cdef int i
        for i in range(15): s = ru(s + fabs(self.Q[i]))
        return s
    cpdef double rad(self):
        cdef double s = 0.0
        s = ru(s + self._bl()); s = ru(s + self._bq()); s = ru(s + self.r)
        return s
    cpdef to_FI(self):
        cdef double bl = self._bl()
        cdef double qpos = 0.0, qneg = 0.0, qcross = 0.0, v
        cdef int k
        for k in range(15):
            v = self.Q[k]
            if _II[k] == _JJ[k]:
                if v > 0: qpos = ru(qpos + v)
                else: qneg = ru(qneg - v)
            else:
                qcross = ru(qcross + fabs(v))
        cdef double sp = 0.0
        sp = ru(sp + bl); sp = ru(sp + qpos); sp = ru(sp + qcross); sp = ru(sp + self.r)
        cdef double sn = 0.0
        sn = ru(sn + bl); sn = ru(sn + qneg); sn = ru(sn + qcross); sn = ru(sn + self.r)
        return FI(rd(self.c - sn), ru(self.c + sp))
    def hi(self): return self.to_FI().hi
    def lo(self): return self.to_FI().lo
    def __add__(a, b):
        if type(a) is TM2 and type(b) is TM2: return _add_tm(<TM2>a, <TM2>b)
        elif type(a) is TM2: return (<TM2>a)._addscalar(<double>b)
        else: return (<TM2>b)._addscalar(<double>a)
    def __radd__(a, b): return (<TM2>a)._addscalar(<double>b)
    cdef TM2 _addscalar(self, double k):
        cdef TM2 o = _new()
        cdef int i
        o.c = self.c + k
        for i in range(5): o.L[i] = self.L[i]
        for i in range(15): o.Q[i] = self.Q[i]
        cdef double s = 0.0
        s = ru(s + self.r); s = ru(s + _gap(o.c))
        o.r = s
        return o
    def __neg__(a):
        cdef TM2 x = <TM2>a
        cdef TM2 o = _new()
        cdef int i
        o.c = -x.c; o.r = x.r
        for i in range(5): o.L[i] = -x.L[i]
        for i in range(15): o.Q[i] = -x.Q[i]
        return o
    def __sub__(a, b):
        if type(a) is TM2 and type(b) is TM2: return _add_tm(<TM2>a, (<TM2>b).__neg__())
        elif type(a) is TM2: return (<TM2>a)._addscalar(-(<double>b))
        else: return ((<TM2>b).__neg__())._addscalar(<double>a)
    def __rsub__(a, b): return ((<TM2>a).__neg__())._addscalar(<double>b)
    def __mul__(a, b):
        if type(a) is TM2 and type(b) is TM2: return _mul_tm(<TM2>a, <TM2>b)
        elif type(a) is TM2: return (<TM2>a)._mulscalar(<double>b)
        else: return (<TM2>b)._mulscalar(<double>a)
    def __rmul__(a, b): return (<TM2>a)._mulscalar(<double>b)
    cdef TM2 _mulscalar(self, double k):
        cdef TM2 o = _new()
        cdef int i
        o.c = self.c*k
        for i in range(5): o.L[i] = self.L[i]*k
        for i in range(15): o.Q[i] = self.Q[i]*k
        cdef double s = 0.0
        s = ru(s + ru(fabs(k)*self.r)); s = ru(s + _gap(o.c))
        for i in range(5): s = ru(s + _gap(o.L[i]))
        for i in range(15): s = ru(s + _gap(o.Q[i]))
        o.r = s
        return o
    def _taylor2(s, f0, f1, f2, d3max):
        cdef TM2 ss = <TM2>s
        cdef TM2 u = _new()
        cdef int i
        u.r = ss.r
        for i in range(5): u.L[i] = ss.L[i]
        for i in range(15): u.Q[i] = ss.Q[i]
        cdef TM2 u2 = _mul_tm(u, u)
        cdef double f1c = 0.5*(f1.lo+f1.hi)
        cdef double f1r = ru(_dmax(ru(f1.hi-f1c), ru(f1c-f1.lo)))
        cdef double f2c = 0.5*(f2.lo+f2.hi)
        cdef double f2r = ru(_dmax(ru(f2.hi-f2c), ru(f2c-f2.lo)))
        cdef double f0c = 0.5*(f0.lo+f0.hi)
        cdef double f0r = ru(_dmax(ru(f0.hi-f0c), ru(f0c-f0.lo)))
        cdef double radu = u.rad()
        cdef TM2 out = (u._mulscalar(f1c)).__add__(u2._mulscalar(f2c)).__add__(f0c)
        cdef double ex = 0.0
        ex = ru(ex + f0r); ex = ru(ex + ru(f1r*radu))
        ex = ru(ex + ru(f2r*ru(radu*radu)))
        ex = ru(ex + ru((<double>d3max)*ru(ru(radu*radu)*radu)))
        cdef TM2 o = <TM2>out
        cdef double sr = 0.0
        sr = ru(sr + o.r); sr = ru(sr + ex)
        o.r = sr
        return o
    def sin(s):
        cdef TM2 ss = <TM2>s
        f0 = F.trig_fi(ss.c, ss.c)[1]; f1 = F.trig_fi(ss.c, ss.c)[0]
        f2 = FI(0.0) - F.trig_fi(ss.c, ss.c)[1]*FI(0.5)
        return ss._taylor2(f0, f1, f2, ru(1.0/6.0))
    def cos(s):
        cdef TM2 ss = <TM2>s
        f0 = F.trig_fi(ss.c, ss.c)[0]; f1 = FI(0.0) - F.trig_fi(ss.c, ss.c)[1]
        f2 = FI(0.0) - F.trig_fi(ss.c, ss.c)[0]*FI(0.5)
        return ss._taylor2(f0, f1, f2, ru(1.0/6.0))

cdef inline TM2 _new():
    cdef TM2 t = TM2.__new__(TM2)
    cdef int i
    t.c = 0.0; t.r = 0.0
    for i in range(5): t.L[i] = 0.0
    for i in range(15): t.Q[i] = 0.0
    return t

cdef TM2 _add_tm(TM2 x, TM2 y):
    cdef TM2 o = _new()
    cdef int i
    o.c = x.c + y.c
    for i in range(5): o.L[i] = x.L[i] + y.L[i]
    for i in range(15): o.Q[i] = x.Q[i] + y.Q[i]
    cdef double s = 0.0
    s = ru(s + x.r); s = ru(s + y.r); s = ru(s + _gap(o.c))
    for i in range(5): s = ru(s + _gap(o.L[i]))
    for i in range(15): s = ru(s + _gap(o.Q[i]))
    o.r = s
    return o

cdef TM2 _mul_tm(TM2 a, TM2 b):
    cdef double ca = a.c, cb = b.c
    cdef TM2 o = _new()
    o.c = ca*cb
    cdef int i, k, ii, jj
    cdef double v, w, w1, w2
    # gaps accumulator starts with rr (computed below), so collect gaps into a running sum AFTER rr.
    # We must _usum([rr, _gap(c), <L gaps...>, <Q gaps...>]) in pure order -> defer: store gap sum start.
    cdef double bla = a._bl(), bqa = a._bq(), ra = a.r
    cdef double blb = b._bl(), bqb = b._bq(), rb = b.r
    cdef double maga = 0.0
    maga = ru(maga + fabs(ca)); maga = ru(maga + bla); maga = ru(maga + bqa); maga = ru(maga + ra)
    cdef double magb = 0.0
    magb = ru(magb + fabs(cb)); magb = ru(magb + blb); magb = ru(magb + bqb); magb = ru(magb + rb)
    cdef double rr = 0.0
    rr = ru(rr + ru(bla*bqb)); rr = ru(rr + ru(blb*bqa)); rr = ru(rr + ru(bqa*bqb))
    rr = ru(rr + ru(ra*magb)); rr = ru(rr + ru(rb*maga))
    # r = _usum([rr, _gap(c), <all gaps in pure order>])
    cdef double s = 0.0
    s = ru(s + rr); s = ru(s + _gap(o.c))
    for i in range(5):
        v = ca*b.L[i] + cb*a.L[i]
        o.L[i] = v
        s = ru(s + _gap(ca*b.L[i])); s = ru(s + _gap(cb*a.L[i])); s = ru(s + _gap(v))
    for k in range(15):
        ii = _II[k]; jj = _JJ[k]
        v = ca*b.Q[k] + cb*a.Q[k]
        s = ru(s + _gap(ca*b.Q[k])); s = ru(s + _gap(cb*a.Q[k])); s = ru(s + _gap(v))
        if ii == jj:
            w = a.L[ii]*b.L[ii]
            v = v + w; s = ru(s + _gap(w))
        else:
            w1 = a.L[ii]*b.L[jj]; w2 = a.L[jj]*b.L[ii]
            v = v + (w1 + w2)          # pure does v += w1 + w2  (= v + (w1+w2)), not (v+w1)+w2
            s = ru(s + _gap(w1)); s = ru(s + _gap(w2)); s = ru(s + _gap(w1 + w2))
        o.Q[k] = v; s = ru(s + _gap(v))
    o.r = s
    return o

_FCOEF = [1.0, -1.0/6.0, 1.0/120.0, -1.0/5040.0, 1.0/362880.0]
_GCOEF = [0.5, -1.0/24.0, 1.0/720.0, -1.0/40320.0, 1.0/3628800.0]
_FTAIL = 1.0/39916800.0
_GTAIL = 1.0/479001600.0

def _series(z, coef, tailc, bint nonneg=False):
    zfi = z.to_FI()
    # nonneg: caller guarantees z >= 0 (z is a sum of squares, e.g. |eta|^2 in
    # rot_from_vec); flooring lo at 0 is unconditionally sound (tightens a provable
    # fact) and prevents spurious enclosure-slack rejections. Bit-identical when lo>=0.
    cdef double lo_eff = zfi.lo
    if nonneg and lo_eff < 0.0: lo_eff = 0.0
    assert zfi.hi <= 1.0 and lo_eff >= -1e-9, "series domain guard"
    cdef int nc = len(coef)
    acc = TM2.const(coef[nc-1])          # NB: coef[-1] is wrong under wraparound=False
    cdef int i
    for i in range(nc-2, -1, -1):
        acc = acc*z + coef[i]
    cdef double Zm = _dmax(fabs(lo_eff), fabs(zfi.hi))
    cdef double tail = ru(1.1*ru((<double>tailc)*ru(_cpow(Zm, 5.0))))
    cdef TM2 o = <TM2>acc
    cdef double s = 0.0
    s = ru(s + o.r); s = ru(s + tail)
    o.r = s
    return o

def sinc_sq(z): return _series(z, _FCOEF, _FTAIL, True)
def oneminuscos_over(z): return _series(z, _GCOEF, _GTAIL, True)

def rot_from_vec(eta):
    z = eta[0]*eta[0] + eta[1]*eta[1] + eta[2]*eta[2]
    f = sinc_sq(z); g = oneminuscos_over(z)
    ex, ey, ez = eta
    outer = [[ex*ex, ex*ey, ex*ez], [ey*ex, ey*ey, ey*ez], [ez*ex, ez*ey, ez*ez]]
    Z0 = TM2.const(0.0)
    sk = [[Z0, -(f*ez), f*ey], [f*ez, Z0, -(f*ex)], [-(f*ey), f*ex, Z0]]
    R = [[None]*3 for _ in range(3)]
    cdef int i, j
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
