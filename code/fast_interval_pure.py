"""FAST rigorous interval core: float64 intervals with math.nextafter directed rounding.
Rigorous (each op's rounding error is bounded by pushing endpoints outward 1 ULP), ~100-1000× faster
than exact-Fraction RI. Ports the flag geometry + mean-value G_λ. A positive lo is still a PROOF."""
import math
from math import nextafter, inf, sqrt as _sqrt
def rd(x): return nextafter(x,-inf)      # round toward −∞
def ru(x): return nextafter(x, inf)      # round toward +∞
class FI:
    __slots__=('lo','hi')
    def __init__(s,lo,hi=None): s.lo=lo; s.hi=(lo if hi is None else hi)
    def __add__(a,b):
        b=b if type(b) is FI else FI(b); return FI(rd(a.lo+b.lo),ru(a.hi+b.hi))
    __radd__=__add__
    def __neg__(a): return FI(-a.hi,-a.lo)
    def __sub__(a,b):
        b=b if type(b) is FI else FI(b); return FI(rd(a.lo-b.hi),ru(a.hi-b.lo))
    def __rsub__(a,b): return (b if type(b) is FI else FI(b)).__sub__(a)
    def __mul__(a,b):
        b=b if type(b) is FI else FI(b)
        p0=a.lo*b.lo;p1=a.lo*b.hi;p2=a.hi*b.lo;p3=a.hi*b.hi
        lo=p0;hi=p0
        if p1<lo:lo=p1
        if p1>hi:hi=p1
        if p2<lo:lo=p2
        if p2>hi:hi=p2
        if p3<lo:lo=p3
        if p3>hi:hi=p3
        return FI(rd(lo),ru(hi))
    __rmul__=__mul__
    def __truediv__(a,b):
        b=b if type(b) is FI else FI(b)
        if b.lo<=0<=b.hi: raise ZeroDivisionError('div by interval containing 0')
        p0=a.lo/b.lo;p1=a.lo/b.hi;p2=a.hi/b.lo;p3=a.hi/b.hi
        lo=min(p0,p1,p2,p3);hi=max(p0,p1,p2,p3); return FI(rd(lo),ru(hi))
    def sqr(a):
        if a.lo>=0: return FI(rd(a.lo*a.lo),ru(a.hi*a.hi))
        if a.hi<=0: return FI(rd(a.hi*a.hi),ru(a.lo*a.lo))
        return FI(0.0, ru(max(a.lo*a.lo,a.hi*a.hi)))
    def sqrt(a): return FI(rd(_sqrt(a.lo)) if a.lo>0 else 0.0, ru(_sqrt(a.hi)) if a.hi>0 else 0.0)
PI=math.pi
def trig_fi(lo,hi):
    lo=rd(lo-1e-15); hi=ru(hi+1e-15)                 # tiny pad for extrema-detection safety
    def crange(a,b,fn,ext_at,exts):                   # exts: list of (angle_of_extremum_base, value)
        vals=[fn(a),fn(b)]
        for base,val in exts:
            k=math.ceil((a-base)/PI)
            while base+k*PI<=b:
                vals.append(val if k%2==0 else -val); k+=1
        return min(vals),max(vals)
    cl,ch=crange(lo,hi,math.cos,0,[(0.0,1.0)])        # cos extrema at k·π: +1,−1
    sl,sh=crange(lo,hi,math.sin,PI/2,[(PI/2,1.0)])    # sin extrema at π/2+k·π
    return FI(rd(cl),ru(ch)),FI(rd(sl),ru(sh))
# vectors
def vsub(u,v): return [a-b for a,b in zip(u,v)]
def dot(u,v):
    r=FI(0.0)
    for a,b in zip(u,v): r=r+a*b
    return r
def cross(u,v): return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
def vnorm(u):
    s=FI(0.0)
    for c in u: s=s+c.sqr()
    return s.sqrt()
def normalize(u):
    n=vnorm(u); return [a/n for a in u]
E3=[FI(0.0),FI(0.0),FI(1.0)]
def frame(u):
    u=normalize(u); f1=normalize(cross(E3,u)); f2=cross(u,f1); return [f1,f2,u]
def applyW(W,v): return [dot(W[0],v),dot(W[1],v),dot(W[2],v)]
def rodrigues(xi):
    th=vnorm(xi); Cc,Ss=trig_fi(th.lo,th.hi); k=[c/th for c in xi]
    K=[[FI(0.0),-k[2],k[1]],[k[2],FI(0.0),-k[0]],[-k[1],k[0],FI(0.0)]]
    def mv(Mx,v): return [dot(Mx[r],v) for r in range(3)]
    omc=FI(1.0)-Cc
    def matR(v):
        Kv=mv(K,v); KKv=mv(K,Kv); return [v[r]+Ss*Kv[r]+omc*KKv[r] for r in range(3)]
    return matR
def proj(w): return [w[1],FI(0.0)-w[0]]
_V4=[[1.,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
def verts(a):
    return [[FI(x) for x in r] for r in _V4]+[[FI(-a*x) for x in r] for r in _V4]
_R2=FI(0.0).__class__  # placeholder
R2=FI(rd(_sqrt(0.5)),ru(_sqrt(0.5)))
def iu2(thlo,thhi,plo,phi):
    # STANDARD SPHERICAL: θ=polar from +z, φ=azimuth ⇒ u=(sinθcosφ, sinθsinφ, cosθ), unit by
    # construction (no division ⇒ tight). Fundamental domain F lives in these coords.
    ct,st=trig_fi(thlo,thhi); cp,sp=trig_fi(plo,phi)
    return [st*cp, st*sp, ct]
def idir(dtlo,dthi,dplo,dphi):
    ct,st=trig_fi(dtlo,dthi); cp,sp=trig_fi(dplo,dphi); return [st*cp,st*sp,ct]
def G_lambda_mv(u2,dirv,s0,hs,aa,edge_order,lam):
    W=frame(u2); V=verts(aa); qo=[proj(applyW(W,v)) for v in V]; edges=[]
    for (p_,r_,sgn) in edge_order:
        d=vsub(qo[r_],qo[p_]); n=[FI(0.0)-d[1],d[0]]
        if sgn<0: n=[-n[0],-n[1]]
        nn=vnorm(n); n=[x/nn for x in n]; ci=None
        for v in qo:
            val=dot(n,v); ci=val if ci is None else FI(min(ci.lo,val.lo),min(ci.hi,val.hi))
        edges.append((n,ci))
    dirI=[x if type(x) is FI else FI(x) for x in dirv]
    Rc=rodrigues([FI(s0)*x for x in dirI]); sC=FI(s0-hs,s0+hs); Rcell=rodrigues([sC*x for x in dirI])
    wc=[Rc(applyW(W,v)) for v in V]; wcell=[Rcell(applyW(W,v)) for v in V]; hsI=FI(-hs,hs)
    G=FI(0.0)
    for k,w in lam:
        ei,vj=k//8,k%8; n,ci=edges[ei]; w=FI(w)
        gc=ci-dot(n,proj(wc[vj])); dg=FI(0.0)-dot(n,proj(cross(dirI,wcell[vj])))
        G=G+w*(gc+dg*hsI)
    return G
