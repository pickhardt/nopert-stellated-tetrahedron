# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# Cython port of g_deriv_aa's AA class + _gap/_usum (the hot arithmetic). FAITHFUL/bit-identical:
# same float ops + directed rounding as the pure-Python version. DA/glam_G stay in Python and call
# this fast AA. Compile with -ffp-contract=off.
import math
from libc.math cimport nextafter, sqrt as _csqrt, fabs, INFINITY
import fast_interval as F
from fast_interval import FI

cdef double _NINF = -INFINITY
cdef double _PINF = INFINITY
cdef inline double _rd(double x): return nextafter(x, _NINF)
cdef inline double _ru(double x): return nextafter(x, _PINF)

cpdef double _gap(double z):
    return _ru(_ru(z) - _rd(z))
def _usum(xs):
    cdef double s = 0.0, x
    for x in xs: s = _ru(s + x)
    return s
cdef inline double _usum2(double a, double b): return _ru(_ru(a) + b)          # = _usum((a,b))
cdef inline double _usum3(double a, double b, double c): return _ru(_ru(_ru(a) + b) + c)  # = _usum((a,b,c))

cdef class AA:
    cdef public double c, cth, cph, r
    def __init__(self, c, cth=0.0, cph=0.0, r=0.0):
        self.c = <double>c; self.cth = <double>cth; self.cph = <double>cph; self.r = <double>r
    @staticmethod
    def const(x): return _mkAA(<double>x, 0.0, 0.0, 0.0)
    @staticmethod
    def from_FI(fi):
        cdef double lo = fi.lo, hi = fi.hi
        cdef double c = 0.5*(lo+hi)
        cdef double r = _ru(_dmax(_ru(hi-c), _ru(c-lo)))
        return _mkAA(c, 0.0, 0.0, r)
    cpdef double rad(AA s):
        return _usum3(fabs(s.cth), fabs(s.cph), s.r)
    cpdef to_FI(AA s):
        cdef double hw = s.rad()
        return FI(_rd(s.c - hw), _ru(s.c + hw))
    def __add__(a, b):
        cdef AA x, y
        cdef double c, cth, cph, bc
        if type(a) is AA and type(b) is AA:
            x = <AA>a; y = <AA>b
            c = x.c + y.c; cth = x.cth + y.cth; cph = x.cph + y.cph
            return _mkAA(c, cth, cph, _usum((x.r, y.r, _gap(c), _gap(cth), _gap(cph))))
        elif type(a) is AA:
            x = <AA>a; bc = <double>b; c = x.c + bc
            return _mkAA(c, x.cth, x.cph, _usum2(x.r, _gap(c)))
        else:
            y = <AA>b; bc = <double>a; c = y.c + bc
            return _mkAA(c, y.cth, y.cph, _usum2(y.r, _gap(c)))
    def __radd__(a, b):
        cdef AA x = <AA>a
        cdef double bc = <double>b, c = x.c + bc
        return _mkAA(c, x.cth, x.cph, _usum2(x.r, _gap(c)))
    def __neg__(a):
        cdef AA x = <AA>a
        return _mkAA(-x.c, -x.cth, -x.cph, x.r)
    def __sub__(a, b):
        cdef AA x, y
        cdef double c, cth, cph, bc
        if type(a) is AA and type(b) is AA:
            x = <AA>a; y = <AA>b
            c = x.c - y.c; cth = x.cth - y.cth; cph = x.cph - y.cph
            return _mkAA(c, cth, cph, _usum((x.r, y.r, _gap(c), _gap(cth), _gap(cph))))
        elif type(a) is AA:
            x = <AA>a; bc = <double>b; c = x.c - bc
            return _mkAA(c, x.cth, x.cph, _usum2(x.r, _gap(c)))
        else:  # scalar - AA
            y = <AA>b; bc = <double>a
            return (y.__neg__()).__add__(bc)
    def __rsub__(a, b):
        return (a.__neg__()).__add__(b)
    def __mul__(a, b):
        cdef AA x, y
        cdef double k, xc, yc, c, cth, cph, radx, rady
        if type(a) is AA and type(b) is AA:
            x = <AA>a; y = <AA>b
            xc = x.c; yc = y.c
            c = xc*yc; cth = xc*y.cth + yc*x.cth; cph = xc*y.cph + yc*x.cph
            radx = x.rad(); rady = y.rad()
            return _mkAA(c, cth, cph, _usum((
                _ru(fabs(xc)*y.r), _ru(fabs(yc)*x.r), _ru(radx*rady), _gap(c),
                _gap(xc*y.cth), _gap(yc*x.cth), _gap(cth),
                _gap(xc*y.cph), _gap(yc*x.cph), _gap(cph))))
        elif type(a) is AA:
            x = <AA>a; k = <double>b
            c = x.c*k; cth = x.cth*k; cph = x.cph*k
            return _mkAA(c, cth, cph, _usum((_ru(fabs(k)*x.r), _gap(c), _gap(cth), _gap(cph))))
        else:
            y = <AA>b; k = <double>a
            c = y.c*k; cth = y.cth*k; cph = y.cph*k
            return _mkAA(c, cth, cph, _usum((_ru(fabs(k)*y.r), _gap(c), _gap(cth), _gap(cph))))
    def __rmul__(a, b):
        return AA.__mul__(a, b)
    cpdef AA sqr(AA s):
        cdef double xc = s.c, two = xc + xc
        cdef double c = xc*xc, cth = two*s.cth, cph = two*s.cph, radx = s.rad()
        return _mkAA(c, cth, cph, _usum((_ru(fabs(two)*s.r), _ru(radx*radx), _gap(c), _gap(cth), _gap(cph))))
    def _cheb(s, fc, p0i, d2max):
        cdef AA ss = <AA>s
        cdef double p0 = 0.5*(p0i.lo + p0i.hi)
        cdef double p0r = _ru(_dmax(_ru(p0i.hi - p0), _ru(p0 - p0i.lo)))
        cdef double cc = 0.5*(fc.lo + fc.hi)
        cdef double cr = _ru(_dmax(_ru(fc.hi - cc), _ru(cc - fc.lo)))
        cdef double radx = ss.rad()
        cdef double cth = p0*ss.cth, cph = p0*ss.cph
        cdef double curv = _ru(0.5*_ru((<double>d2max)*_ru(radx*radx)))
        return _mkAA(cc, cth, cph, _usum((cr, _ru(fabs(p0)*ss.r), _ru(p0r*radx), curv, _gap(cth), _gap(cph))))
    def sqrt(s):
        cdef AA ss = <AA>s
        fi = ss.to_FI(); a = fi.lo; b = fi.hi
        if a <= 0.0:
            return AA.from_FI(fi.sqrt())
        fc = FI(_rd(_csqrt(ss.c)) if ss.c > 0 else 0.0, _ru(_csqrt(ss.c)))
        p0 = FI(0.5) / FI(_rd(_csqrt(ss.c)), _ru(_csqrt(ss.c)))
        a15 = (FI(a, a).sqrt() * FI(a, a)).lo
        d2 = _ru(0.25 / a15)
        return ss._cheb(fc, p0, d2)
    def recip(s):
        cdef AA ss = <AA>s
        fi = ss.to_FI(); a = fi.lo; b = fi.hi
        if a <= 0.0 <= b: raise ZeroDivisionError('AA.recip over interval containing 0')
        fc = FI(1.0) / FI(ss.c, ss.c)
        p0 = FI(0.0) - (FI(1.0) / FI(ss.c, ss.c).sqr())
        cdef double m = fabs(a) if fabs(a) < fabs(b) else fabs(b)
        m3 = (FI(m, m).sqr() * FI(m, m)).lo
        d2 = _ru(2.0 / m3)
        return ss._cheb(fc, p0, d2)
    def __truediv__(a, b):
        if type(b) is not AA: return a.__mul__(1.0/float(b))
        return a.__mul__(b.recip())
    def cos(s):
        cdef AA ss = <AA>s
        fi = ss.to_FI()
        fc = F.trig_fi(ss.c, ss.c)[0]
        p0 = FI(0.0) - F.trig_fi(ss.c, ss.c)[1]
        cc = F.trig_fi(fi.lo, fi.hi)[0]; d2 = _dmax(fabs(cc.lo), fabs(cc.hi))
        return ss._cheb(fc, p0, d2)
    def sin(s):
        cdef AA ss = <AA>s
        fi = ss.to_FI()
        fc = F.trig_fi(ss.c, ss.c)[1]
        p0 = F.trig_fi(ss.c, ss.c)[0]
        sv = F.trig_fi(fi.lo, fi.hi)[1]; d2 = _dmax(fabs(sv.lo), fabs(sv.hi))
        return ss._cheb(fc, p0, d2)

cdef inline double _dmax(double a, double b): return a if a > b else b
cdef inline AA _mkAA(double c, double cth, double cph, double r):
    cdef AA x = AA.__new__(AA)
    x.c = c; x.cth = cth; x.cph = cph; x.r = r
    return x
