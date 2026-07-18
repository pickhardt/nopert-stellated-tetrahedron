# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# Cython port of g_deriv_aa's DA (dual-affine forward-mode AD: value + d/dsym1, d/dsym2, each a cython AA).
# FAITHFUL: identical op sequence to the pure DA, so bounds are bit-identical. Only the Python-level DA
# object churn (the ~55% profile bottleneck of glam_lip_dir/s) is moved to native code; the AA arithmetic
# is already cython. Compile with the same flags as g_aa_cy.
from fast_interval import FI
from g_aa_cy import AA

cdef inline DA _mk(object v, object dth, object dph):
    cdef DA x = DA.__new__(DA)
    x.v = v; x.dth = dth; x.dph = dph
    return x

cdef object _AZ = AA.const(0.0)

cdef class DA:
    cdef public object v, dth, dph
    def __init__(self, v, dth=None, dph=None):
        if type(v) is AA: self.v = v
        elif type(v) is FI: self.v = AA.from_FI(v)
        else: self.v = AA.const(v)
        self.dth = dth if dth is not None else AA.const(0.0)
        self.dph = dph if dph is not None else AA.const(0.0)
    @staticmethod
    def const(x):
        cdef object v = AA.from_FI(x) if type(x) is FI else AA.const(x)
        return _mk(v, AA.const(0.0), AA.const(0.0))
    def __add__(a, b):
        cdef DA x, y
        if type(a) is DA and type(b) is DA: x = <DA>a; y = <DA>b
        elif type(a) is DA: x = <DA>a; y = <DA>DA.const(b)
        else: x = <DA>DA.const(a); y = <DA>b
        return _mk(x.v + y.v, x.dth + y.dth, x.dph + y.dph)
    def __radd__(a, b):
        cdef DA x = <DA>a, y = <DA>DA.const(b)
        return _mk(x.v + y.v, x.dth + y.dth, x.dph + y.dph)
    def __neg__(a):
        cdef DA x = <DA>a
        return _mk(-x.v, -x.dth, -x.dph)
    def __sub__(a, b):
        cdef DA x, y
        if type(a) is DA and type(b) is DA: x = <DA>a; y = <DA>b
        elif type(a) is DA: x = <DA>a; y = <DA>DA.const(b)
        else: x = <DA>DA.const(a); y = <DA>b
        return _mk(x.v - y.v, x.dth - y.dth, x.dph - y.dph)
    def __rsub__(a, b):
        cdef DA x = <DA>a, y = <DA>DA.const(b)
        return _mk(y.v - x.v, y.dth - x.dth, y.dph - x.dph)
    def __mul__(a, b):
        cdef DA x, y
        if type(a) is DA and type(b) is DA: x = <DA>a; y = <DA>b
        elif type(a) is DA: x = <DA>a; y = <DA>DA.const(b)
        else: x = <DA>DA.const(a); y = <DA>b
        return _mk(x.v * y.v, x.dth * y.v + x.v * y.dth, x.dph * y.v + x.v * y.dph)
    def __rmul__(a, b):
        cdef DA x = <DA>a, y = <DA>DA.const(b)
        return _mk(x.v * y.v, x.dth * y.v + x.v * y.dth, x.dph * y.v + x.v * y.dph)
    def __truediv__(a, b):
        cdef DA x, y
        if type(a) is DA and type(b) is DA: x = <DA>a; y = <DA>b
        elif type(a) is DA: x = <DA>a; y = <DA>DA.const(b)
        else: x = <DA>DA.const(a); y = <DA>b
        g2 = y.v * y.v
        return _mk(x.v / y.v, (x.dth * y.v - x.v * y.dth) / g2, (x.dph * y.v - x.v * y.dph) / g2)
    cpdef DA sqr(DA a):
        return _mk(a.v.sqr(), (a.v * a.dth) * 2.0, (a.v * a.dph) * 2.0)
    cpdef DA sqrt(DA a):
        r = a.v.sqrt(); tworr = r * 2.0
        return _mk(r, a.dth / tworr, a.dph / tworr)
