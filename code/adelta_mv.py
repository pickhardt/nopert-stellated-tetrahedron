"""adelta_mv.py -- MEAN-VALUE bound on A_δ that certifies the transition slivers with COARSE cells,
where the direct interval jet (adelta_jet) needs ~1e9 cells.  Key facts:
  * d²r_P/dδ² at a POINT (δc, φc) has no δ-interval cancellation -> the tight jet gives it directly;
  * the true third derivative d³r_P/dδ³ near the transitions is small (~19), so over a cell
        D2(δ,φ) ∈ D2(δc, φcell)  ⊕  D3(δcell, φcell) · [-δhw, δhw]      (Taylor-Lagrange in δ)
    is a rigorous enclosure with a NEGLIGIBLE D3·δhw correction even at coarse δhw.
D2(δc,·) is computed by the validated tight jet (adelta_jet.flags_jet at point δ, using det2_tight);
D3 by a plain third-order jet here (looseness is harmless -- it only multiplies δhw).  φ is kept as a
small interval (dyadic-refined near a transition); the jet accounts for φ-variation rigorously.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_interval as F
from fast_interval import FI
import adelta_jet as AJ
from adelta_jet import J2, flags_jet, flag_struct, VERTS, R2, frame_np

Z3 = FI(0.0)
TWO = FI(2.0); THREE = FI(3.0); SIX = FI(6.0)

class J3:
    """degree-3 Taylor jet in δ (v,d,dd,ddd), plain FI arithmetic (used only for the loose D3)."""
    __slots__ = ('v','d','dd','ddd')
    def __init__(s, v, d=None, dd=None, ddd=None):
        s.v=v; s.d=Z3 if d is None else d; s.dd=Z3 if dd is None else dd; s.ddd=Z3 if ddd is None else ddd
    def __add__(s,o):
        if not isinstance(o,J3): o=J3(o)
        return J3(s.v+o.v, s.d+o.d, s.dd+o.dd, s.ddd+o.ddd)
    def __sub__(s,o):
        if not isinstance(o,J3): o=J3(o)
        return J3(s.v-o.v, s.d-o.d, s.dd-o.dd, s.ddd-o.ddd)
    def __neg__(s): return J3(Z3-s.v, Z3-s.d, Z3-s.dd, Z3-s.ddd)
    def __mul__(s,o):
        if not isinstance(o,J3): o=J3(o)
        v=s.v*o.v
        d=s.d*o.v+s.v*o.d
        dd=s.dd*o.v+TWO*s.d*o.d+s.v*o.dd
        ddd=s.ddd*o.v+THREE*s.dd*o.d+THREE*s.d*o.dd+s.v*o.ddd
        return J3(v,d,dd,ddd)
    def __truediv__(s,o):
        if not isinstance(o,J3): o=J3(o)
        v=s.v/o.v
        d=(s.d-v*o.d)/o.v
        dd=(s.dd-TWO*d*o.d-v*o.dd)/o.v
        ddd=(s.ddd-THREE*dd*o.d-THREE*d*o.dd-v*o.ddd)/o.v
        return J3(v,d,dd,ddd)
    def sqrt(s):
        r=s.v.sqrt(); tr=TWO*r
        d=s.d/tr
        dd=(s.dd-TWO*d.sqr())/tr
        ddd=(s.ddd-SIX*d*dd)/tr
        return J3(r,d,dd,ddd)

def jdot3(u,v):
    r=u[0]*v[0]
    for i in (1,2): r=r+u[i]*v[i]
    return r
def jcross3(u,v):
    return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
def jnorm3(u):
    ssum=u[0].__mul__(u[0])
    for i in (1,2): ssum=ssum+u[i]*u[i]
    return ssum.sqrt()
def jnormalize3(u):
    n=jnorm3(u); return [c/n for c in u]

E3J3=[J3(FI(0.0)),J3(FI(0.0)),J3(FI(1.0))]
def frame_jet3(u):
    u=jnormalize3(u); f1=jnormalize3(jcross3(E3J3,u)); f2=jcross3(u,f1); return [f1,f2,u]

def u2_jet3(dlo,dhi,plo,phi):
    cd,sd=F.trig_fi(dlo,dhi); cp,sp=F.trig_fi(plo,phi)
    cosd=J3(cd, Z3-sd, Z3-cd, sd)     # d/dδ: -sin,-cos,+sin
    sind=J3(sd, cd, Z3-sd, Z3-cd)     # d/dδ: +cos,-sin,-cos
    cpr=J3(cp*R2); spj=J3(sp)
    ux=cosd*R2 + sind*cpr
    uy=cosd*(Z3-R2) + sind*cpr
    uz=sind*spj
    return [ux,uy,uz]

def junit2d3(mx,my):
    """unit-normal 2-jet to 3rd order via the angle ψ=atan2(my,mx). Plain FI (loose D3 is fine)."""
    D=mx.v.sqr()+my.v.sqr(); L=D.sqrt()
    Dp=TWO*(mx.v*mx.d+my.v*my.d)
    Dpp=TWO*(mx.d.sqr()+mx.v*mx.dd+my.d.sqr()+my.v*my.dd)
    Dppp=TWO*(THREE*mx.d*mx.dd+mx.v*mx.ddd+THREE*my.d*my.dd+my.v*my.ddd)
    N=mx.v*my.d-my.v*mx.d
    Np=mx.v*my.dd-my.v*mx.dd
    Npp=mx.v*my.ddd-my.v*mx.ddd + mx.d*my.dd-my.d*mx.dd
    psip=N/D
    P=Np*D-N*Dp; psipp=P/D.sqr()
    Pp=Npp*D-N*Dpp; psippp=(Pp*D-TWO*P*Dp)/(D*D.sqr())
    hx=mx.v/L; hy=my.v/L
    hxp=Z3-psip*hy; hyp=psip*hx
    hxpp=(Z3-psip.sqr()*hx)-psipp*hy; hypp=(Z3-psip.sqr()*hy)+psipp*hx
    t3=psippp-psip*psip.sqr()          # ψ''' - ψ'³
    hxppp=(Z3-THREE*psip*psipp*hx)-t3*hy
    hyppp=(Z3-THREE*psip*psipp*hy)+t3*hx
    return J3(hx,hxp,hxpp,hxppp), J3(hy,hyp,hypp,hyppp)

def flags_jet3(dlo,dhi,plo,phi,struct):
    u2=u2_jet3(dlo,dhi,plo,phi); W=frame_jet3(u2)
    Wv=[[jdot3(W[r],[J3(x) for x in V]) for r in range(3)] for V in VERTS]
    q=[[wv[1], J3(FI(0.0))-wv[0]] for wv in Wv]
    zc=[wv[2] for wv in Wv]
    flags=[]
    for (p_,r_,sign) in struct:
        dq=[q[r_][0]-q[p_][0], q[r_][1]-q[p_][1]]
        rawx=J3(FI(0.0))-dq[1]; rawy=dq[0]
        nx,ny=junit2d3(rawx,rawy)
        if sign<0: nx=J3(FI(0.0))-nx; ny=J3(FI(0.0))-ny
        for j in range(8):
            Jq=nx*(J3(FI(0.0))-q[j][1])+ny*q[j][0]
            flags.append([zc[j]*nx, zc[j]*ny, Jq, nx, ny])
    return flags

def cell_Adelta_mv(dlo,dhi,plo,phi,struct=None):
    """rigorous sup|d²r_P/dδ²| over the cell via the δ mean-value form. None if structure degenerate."""
    dc=0.5*(dlo+dhi); dhw=0.5*(dhi-dlo)
    if struct is None:
        s=flag_struct(dc,0.5*(plo+phi))
        if s is None: return None
        struct=s[0]
    # tight center D2 (point δc, φ-interval) from the validated J2 jet
    cres=flags_jet(dc,dc,plo,phi,struct)
    if cres is None: return None
    cflags,_=cres
    # loose D3 over the full cell
    d3flags=flags_jet3(dlo,dhi,plo,phi,struct)
    dhwI=FI(-dhw,dhw)
    m=0.0
    for fc,f3 in zip(cflags,d3flags):
        for c2,c3 in zip(fc,f3):
            val=c2.dd + c3.ddd*dhwI       # D2(δc,φcell) ⊕ D3·[-δhw,δhw]
            m=max(m, abs(val.lo), abs(val.hi))
    return m

if __name__=='__main__':
    import sbbox_prove as S, numpy as np
    d=5.5e-3; t=[x for x in S.find_transitions(d) if 0<x<np.pi][1]
    print('mean-value enclosures at COARSE δhw=2.5e-4 near transition:')
    for eps in [1e-3,1e-4,1e-5,1e-6]:
        phi=t-eps; phw=min(eps*0.1,1e-4)
        e=cell_Adelta_mv(d-2.5e-4,d+2.5e-4,phi-phw,phi+phw)
        print(f'  ε={eps:.0e} (φhw={phw:.0e}): {e}')
    print('benign φ=1.0 δhw=2.5e-4 φhw=5e-3:', cell_Adelta_mv(d-2.5e-4,d+2.5e-4,1.0-5e-3,1.0+5e-3))
