# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# Cython port of fast_interval.py — FI as a cdef class with native double arithmetic.
# FAITHFUL: identical IEEE-754 double ops + math.nextafter directed rounding as the pure-Python
# version (compile with -ffp-contract=off so no FMA changes intermediates). A positive .lo is a proof.
import math
from libc.math cimport nextafter, sqrt as _csqrt, ceil as _cceil, cos as _ccos, sin as _csin, INFINITY

cdef double _NINF = -INFINITY
cdef double _PINF = INFINITY
cdef inline double rd(double x): return nextafter(x, _NINF)   # toward -inf
cdef inline double ru(double x): return nextafter(x, _PINF)   # toward +inf

cdef class FI:
    cdef public double lo, hi
    def __init__(self, lo, hi=None):
        # far engine passes floats; a non-float (Fraction/int) is enclosed outward for soundness
        cdef double l, h
        if type(lo) is float:
            l = <double>lo
        else:
            l = <double>lo   # __float__; only ever the point-constant path (dyadic in practice)
        self.lo = l
        if hi is None:
            self.hi = l
        else:
            self.hi = <double>hi if type(hi) is float else <double>hi
    def __add__(a, b):
        cdef FI x, y
        cdef double blo, bhi
        if type(a) is FI:
            x = <FI>a
            if type(b) is FI: y = <FI>b; blo = y.lo; bhi = y.hi
            else: blo = <double>b; bhi = blo
            return _mk(rd(x.lo+blo), ru(x.hi+bhi))
        else:  # a scalar + FI b
            y = <FI>b; blo = <double>a
            return _mk(rd(blo+y.lo), ru(blo+y.hi))
    def __radd__(a, b):
        cdef FI x = <FI>a
        cdef double blo = <double>b
        return _mk(rd(blo+x.lo), ru(blo+x.hi))
    def __neg__(a):
        cdef FI x = <FI>a
        return _mk(-x.hi, -x.lo)
    def __sub__(a, b):
        cdef FI x, y
        cdef double blo, bhi
        if type(a) is FI:
            x = <FI>a
            if type(b) is FI: y = <FI>b; blo = y.lo; bhi = y.hi
            else: blo = <double>b; bhi = blo
            return _mk(rd(x.lo-bhi), ru(x.hi-blo))
        else:
            y = <FI>b; blo = <double>a
            return _mk(rd(blo-y.hi), ru(blo-y.lo))
    def __rsub__(a, b):
        cdef FI x = <FI>a
        cdef double blo = <double>b
        return _mk(rd(blo-x.hi), ru(blo-x.lo))
    def __mul__(a, b):
        cdef FI x, y
        cdef double alo, ahi, blo, bhi, p0, p1, p2, p3, lo, hi
        if type(a) is FI and type(b) is FI:
            x = <FI>a; y = <FI>b; alo=x.lo; ahi=x.hi; blo=y.lo; bhi=y.hi
        elif type(a) is FI:
            x = <FI>a; alo=x.lo; ahi=x.hi; blo=<double>b; bhi=blo
        else:
            y = <FI>b; blo=y.lo; bhi=y.hi; alo=<double>a; ahi=alo
        p0=alo*blo; p1=alo*bhi; p2=ahi*blo; p3=ahi*bhi
        lo=p0; hi=p0
        if p1<lo: lo=p1
        if p1>hi: hi=p1
        if p2<lo: lo=p2
        if p2>hi: hi=p2
        if p3<lo: lo=p3
        if p3>hi: hi=p3
        return _mk(rd(lo), ru(hi))
    def __rmul__(a, b):
        return FI.__mul__(a, b)
    def __truediv__(a, b):
        cdef FI x, y
        cdef double alo, ahi, blo, bhi, p0, p1, p2, p3, lo, hi
        if type(a) is FI and type(b) is FI:
            x=<FI>a; y=<FI>b; alo=x.lo; ahi=x.hi; blo=y.lo; bhi=y.hi
        elif type(a) is FI:
            x=<FI>a; alo=x.lo; ahi=x.hi; blo=<double>b; bhi=blo
        else:
            y=<FI>b; blo=y.lo; bhi=y.hi; alo=<double>a; ahi=alo
        if blo<=0<=bhi: raise ZeroDivisionError('div by interval containing 0')
        p0=alo/blo; p1=alo/bhi; p2=ahi/blo; p3=ahi/bhi
        lo=p0; hi=p0
        if p1<lo: lo=p1
        if p1>hi: hi=p1
        if p2<lo: lo=p2
        if p2>hi: hi=p2
        if p3<lo: lo=p3
        if p3>hi: hi=p3
        return _mk(rd(lo), ru(hi))
    cpdef FI sqr(FI a):
        cdef double lo=a.lo, hi=a.hi
        if lo>=0: return _mk(rd(lo*lo), ru(hi*hi))
        if hi<=0: return _mk(rd(hi*hi), ru(lo*lo))
        cdef double m = lo*lo
        if hi*hi>m: m=hi*hi
        return _mk(0.0, ru(m))
    cpdef FI sqrt(FI a):
        return _mk(rd(_csqrt(a.lo)) if a.lo>0 else 0.0, ru(_csqrt(a.hi)) if a.hi>0 else 0.0)

cdef inline FI _mk(double lo, double hi):
    cdef FI r = FI.__new__(FI)
    r.lo = lo; r.hi = hi
    return r

PI = math.pi
cdef double _PIc = math.pi

def trig_fi(lo, hi):
    cdef double L = rd(<double>lo - 1e-15), H = ru(<double>hi + 1e-15)
    def crange(double a, double b, fn, double base, double extval):
        vals=[fn(a), fn(b)]
        cdef long k = <long>_cceil((a-base)/_PIc)
        while base + k*_PIc <= b:
            vals.append(extval if k%2==0 else -extval); k+=1
        return min(vals), max(vals)
    cl, ch = crange(L, H, math.cos, 0.0, 1.0)
    sl, sh = crange(L, H, math.sin, _PIc/2, 1.0)
    return _mk(rd(cl), ru(ch)), _mk(rd(sl), ru(sh))

def vsub(u, v): return [a-b for a,b in zip(u,v)]
def dot(u, v):
    cdef FI r = _mk(0.0, 0.0)
    for a,b in zip(u,v): r = r + a*b
    return r
def cross(u, v): return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
def vnorm(u):
    cdef FI s = _mk(0.0, 0.0)
    for c in u: s = s + c.sqr()
    return s.sqrt()
def normalize(u):
    n = vnorm(u); return [a/n for a in u]
E3 = [_mk(0.0,0.0), _mk(0.0,0.0), _mk(1.0,1.0)]
def frame(u):
    u = normalize(u); f1 = normalize(cross(E3,u)); f2 = cross(u,f1); return [f1,f2,u]
def applyW(W, v): return [dot(W[0],v), dot(W[1],v), dot(W[2],v)]
def rodrigues(xi):
    th = vnorm(xi); Cc, Ss = trig_fi(th.lo, th.hi); k = [c/th for c in xi]
    K = [[_mk(0.0,0.0),-k[2],k[1]],[k[2],_mk(0.0,0.0),-k[0]],[-k[1],k[0],_mk(0.0,0.0)]]
    def mv(Mx, v): return [dot(Mx[r],v) for r in range(3)]
    omc = _mk(1.0,1.0) - Cc
    def matR(v):
        Kv = mv(K,v); KKv = mv(K,Kv); return [v[r]+Ss*Kv[r]+omc*KKv[r] for r in range(3)]
    return matR
def proj(w): return [w[1], _mk(0.0,0.0)-w[0]]
_V4 = [[1.,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
def verts(a):
    return [[FI(x) for x in r] for r in _V4] + [[FI(-a*x) for x in r] for r in _V4]
R2 = _mk(rd(_csqrt(0.5)), ru(_csqrt(0.5)))
def iu2(thlo, thhi, plo, phi):
    ct, st = trig_fi(thlo, thhi); cp, sp = trig_fi(plo, phi)
    return [st*cp, st*sp, ct]
def idir(dtlo, dthi, dplo, dphi):
    ct, st = trig_fi(dtlo, dthi); cp, sp = trig_fi(dplo, dphi); return [st*cp, st*sp, ct]
def G_lambda_mv(u2, dirv, s0, hs, aa, edge_order, lam):
    W = frame(u2); V = verts(aa); qo = [proj(applyW(W,v)) for v in V]; edges = []
    for (p_, r_, sgn) in edge_order:
        d = vsub(qo[r_], qo[p_]); n = [_mk(0.0,0.0)-d[1], d[0]]
        if sgn<0: n = [-n[0], -n[1]]
        nn = vnorm(n); n = [x/nn for x in n]; ci = None
        for v in qo:
            val = dot(n,v); ci = val if ci is None else _mk(min(ci.lo,val.lo), min(ci.hi,val.hi))
        edges.append((n, ci))
    dirI = [x if type(x) is FI else FI(x) for x in dirv]
    Rc = rodrigues([FI(s0)*x for x in dirI]); sC = FI(s0-hs, s0+hs); Rcell = rodrigues([sC*x for x in dirI])
    wc = [Rc(applyW(W,v)) for v in V]; wcell = [Rcell(applyW(W,v)) for v in V]; hsI = FI(-hs, hs)
    G = _mk(0.0,0.0)
    for k,w in lam:
        ei, vj = k//8, k%8; n, ci = edges[ei]; w = FI(w)
        gc = ci - dot(n, proj(wc[vj])); dg = _mk(0.0,0.0) - dot(n, proj(cross(dirI, wcell[vj])))
        G = G + w*(gc + dg*hsI)
    return G
