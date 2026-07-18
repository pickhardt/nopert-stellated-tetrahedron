"""flagcalc2.py -- exact/interval flag calculus, v2: two-symbol Taylor forms.

Forms in t = delta-delta_c (|t|<=w) and s = phi-phi_c (|s|<=v):
    x(t,s) in c + a*t + b*s + d*t*s + r*[-1,1],  c,a,b,d intervals, r>=0.
All coefficient arithmetic is outward-rounded interval arithmetic; remainders
collect every monomial outside {1,t,s,ts} conservatively.

Provides: frame/vertex/flag forms for P_{11/20} near u*=(1,-1,0)/sqrt2, hull
config checks, and the inversion-free "p-point" membership verifier:
  for float weights lam(t,s)=lam0+lam1 t+lam2 s+lam3 ts >= 0 on the cell,
  E_k := sum_i lam_i(t,s) r_i(t,s) - (delta_c+t) g_k  is hulled; if
  ||E_k|| <= eta and the once-verified support margin of the generator set
  exceeds eta/delta_lo, then delta*K_box subset conv F on the whole cell.
Soundness: p_k = sum lam_i r_i / sum lam_i lies in conv F exactly (lam>=0),
and h_{conv{p_k}}(u) >= delta*(h_gens(u) - eta/delta) >= delta*h_K(u).
"""
import numpy as np

INF = np.inf
def _up(x): return np.nextafter(x, INF)
def _dn(x): return np.nextafter(x, -INF)
def infl(x):
    x = np.asarray(x, dtype=float)
    return _up(_up(np.abs(x)) * (1 + 1e-12) + 1e-300)

def ithin(x):
    x = np.asarray(x, dtype=float); return (x, x)
def ival(lo,hi): return (np.asarray(lo,dtype=float), np.asarray(hi,dtype=float))
def iadd(a,b): return (_dn(a[0]+b[0]), _up(a[1]+b[1]))
def isub(a,b): return (_dn(a[0]-b[1]), _up(a[1]-b[0]))
def ineg(a):   return (-a[1], -a[0])
def imul(a,b):
    c1,c2,c3,c4 = a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]
    return (_dn(np.minimum(np.minimum(c1,c2),np.minimum(c3,c4))),
            _up(np.maximum(np.maximum(c1,c2),np.maximum(c3,c4))))
def iinv(a):
    assert np.all(a[0]*a[1] > 0)
    return (_dn(np.minimum(1/a[0],1/a[1])), _up(np.maximum(1/a[0],1/a[1])))
def imag(a):   return np.maximum(np.abs(a[0]), np.abs(a[1]))
def isin(a):
    m=0.5*(a[0]+a[1]); r=_up(0.5*(a[1]-a[0])); sn=np.sin(m)
    sl=infl(r+4e-16*(1+np.abs(sn))); return (_dn(sn-sl),_up(sn+sl))

class F2:
    __slots__=('c','a','b','d','r')
    def __init__(s,c,a,b,d,r): s.c,s.a,s.b,s.d,s.r = c,a,b,d,np.asarray(r,dtype=float)
def f2const(x):
    z=ithin(np.zeros(np.shape(np.asarray(x)))); return F2(ithin(x),z,z,z,np.zeros(np.shape(np.asarray(x))))
def f2ival(c):
    z=ithin(np.zeros(np.shape(c[0]))); return F2(c,z,z,z,np.zeros(np.shape(c[0])))
def f2add(x,y): return F2(iadd(x.c,y.c),iadd(x.a,y.a),iadd(x.b,y.b),iadd(x.d,y.d),infl(x.r+y.r))
def f2sub(x,y): return F2(isub(x.c,y.c),isub(x.a,y.a),isub(x.b,y.b),isub(x.d,y.d),infl(x.r+y.r))
def f2neg(x):   return F2(ineg(x.c),ineg(x.a),ineg(x.b),ineg(x.d),x.r)
def f2mul(x,y,w,v):
    c=imul(x.c,y.c)
    a=iadd(imul(x.c,y.a),imul(x.a,y.c))
    b=iadd(imul(x.c,y.b),imul(x.b,y.c))
    d=iadd(iadd(imul(x.c,y.d),imul(x.d,y.c)), iadd(imul(x.a,y.b),imul(x.b,y.a)))
    mxc,mxa,mxb,mxd = imag(x.c),imag(x.a),imag(x.b),imag(x.d)
    myc,mya,myb,myd = imag(y.c),imag(y.a),imag(y.b),imag(y.d)
    T,S=w,v
    hx = mxc+mxa*T+mxb*S+mxd*T*S+x.r
    hy = myc+mya*T+myb*S+myd*T*S+y.r
    rem = ( mxa*mya*T*T + mxb*myb*S*S
          + (mxa*myd+mxd*mya)*T*T*S + (mxb*myd+mxd*myb)*T*S*S
          + mxd*myd*T*T*S*S
          + x.r*hy + y.r*hx + x.r*y.r )
    return F2(c,a,b,d,infl(rem))
def f2hull(x,w,v):
    sl=_up(imag(x.a)*w + imag(x.b)*v + imag(x.d)*w*v + x.r)
    return (_dn(x.c[0]-sl), _up(x.c[1]+sl))
def _split(x,w,v):
    m=0.5*(x.c[0]+x.c[1])
    um=infl(np.maximum(np.abs(x.c[0]-m),np.abs(x.c[1]-m)) + imag(x.a)*w+imag(x.b)*v+imag(x.d)*w*v + x.r)
    return m,um
def f2sqrt(x,w,v):
    X=f2hull(x,w,v); assert np.all(X[0]>0), "f2sqrt range"
    m,um=_split(x,w,v)
    sm=ival(_dn(np.sqrt(m)),_up(np.sqrt(m))); inv2s=iinv(iadd(sm,sm))
    c=iadd(sm, imul(isub(x.c,ithin(m)),inv2s))
    a=imul(x.a,inv2s); b=imul(x.b,inv2s); d=imul(x.d,inv2s)
    rem=infl(x.r*imag(inv2s) + um*um/(8.0*_dn(X[0]*np.sqrt(X[0]))))
    return F2(c,a,b,d,rem)
def f2inv(x,w,v):
    X=f2hull(x,w,v); assert np.all(X[0]*X[1]>0), "f2inv range"
    m,um=_split(x,w,v)
    im=ival(_dn(1.0/m),_up(1.0/m)); im2=imul(im,im)
    c=isub(im, imul(isub(x.c,ithin(m)),im2))
    a=ineg(imul(x.a,im2)); b=ineg(imul(x.b,im2)); d=ineg(imul(x.d,im2))
    lo3=_dn(np.minimum(np.abs(X[0]),np.abs(X[1]))**3)
    rem=infl(x.r*imag(im2) + um*um/lo3)
    return F2(c,a,b,d,rem)
def f2div(x,y,w,v): return f2mul(x,f2inv(y,w,v),w,v)
# sin/cos of (delta_c + t): forms in t; of (phi_c + s): forms in s
def f2sin_t(dc,w):
    sn,cs=np.sin(dc),np.cos(dc); z=ithin(0.0)
    return F2(ival(_dn(sn-3e-16),_up(sn+3e-16)), ival(_dn(cs-3e-16),_up(cs+3e-16)), z,z, infl(w*w/2))
def f2cos_t(dc,w):
    sn,cs=np.sin(dc),np.cos(dc); z=ithin(0.0)
    return F2(ival(_dn(cs-3e-16),_up(cs+3e-16)), ival(_dn(-sn-3e-16),_up(-sn+3e-16)), z,z, infl(w*w/2))
def f2sin_s(fc,v):
    sn,cs=np.sin(fc),np.cos(fc); z=ithin(0.0)
    return F2(ival(_dn(sn-3e-16),_up(sn+3e-16)), z, ival(_dn(cs-3e-16),_up(cs+3e-16)), z, infl(v*v/2))
def f2cos_s(fc,v):
    sn,cs=np.sin(fc),np.cos(fc); z=ithin(0.0)
    return F2(ival(_dn(cs-3e-16),_up(cs+3e-16)), z, ival(_dn(-sn-3e-16),_up(-sn+3e-16)), z, infl(v*v/2))

# ------------------- geometry -------------------
VV=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=float)
A_=11.0/20.0
VERTS=np.vstack([VV,-A_*VV])
SQ2=ival(_dn(np.sqrt(2.0)),_up(np.sqrt(2.0)))
PHI_W=np.arctan(11*np.sqrt(2.0)/9)
HULLS={'pentA':[0,3,4,1,7],'hex1':[0,3,4,2,1,7],'pentB':[0,3,4,2,7]}
INTER={'pentA':[2,5,6],'hex1':[5,6],'pentB':[1,5,6]}

def frame_forms(dlo,dhi,flo,fhi):
    dc=0.5*(dlo+dhi); w=_up(max(_up(dhi-dc),_up(dc-dlo)))
    fc=0.5*(flo+fhi); v=_up(max(_up(fhi-fc),_up(fc-flo)))
    sph,cph=f2sin_s(fc,v),f2cos_s(fc,v)
    isq2=iinv(SQ2)
    wx=f2neg(f2mul(cph,f2ival(isq2),w,v)); wy=wx; wz=f2neg(sph)
    sd,cd=f2sin_t(dc,w),f2cos_t(dc,w)
    one=f2const(1.0); omc=f2sub(one,cd)
    W3=[wx,wy,wz]; zf=f2const(0.0)
    K=[[zf,f2neg(W3[2]),W3[1]],[W3[2],zf,f2neg(W3[0])],[f2neg(W3[1]),W3[0],zf]]
    K2=[[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            sacc=f2const(0.0)
            for k in range(3): sacc=f2add(sacc,f2mul(K[i][k],K[k][j],w,v))
            K2[i][j]=sacc
    Rf=[[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            Rf[i][j]=f2add(f2const(1.0 if i==j else 0.0),
                           f2add(f2mul(sd,K[i][j],w,v),f2mul(omc,K2[i][j],w,v)))
    e1i=[f2const(0.0),f2const(0.0),f2const(1.0)]
    e2i=[f2ival(ineg(isq2)),f2ival(ineg(isq2)),f2const(0.0)]
    ui =[f2ival(isq2),f2ival(ineg(isq2)),f2const(0.0)]
    def mv(vec):
        out=[]
        for i in range(3):
            sacc=f2const(0.0)
            for k in range(3): sacc=f2add(sacc,f2mul(Rf[i][k],vec[k],w,v))
            out.append(sacc)
        return out
    E1R,E2R,UR=mv(e1i),mv(e2i),mv(ui)
    q=[[None,None] for _ in range(8)]; z=[None]*8
    for j in range(8):
        vj=VERTS[j]
        def dot(row):
            sacc=f2const(0.0)
            for k in range(3): sacc=f2add(sacc,f2mul(row[k],f2const(vj[k]),w,v))
            return sacc
        q[j][0],q[j][1],z[j]=dot(E1R),dot(E2R),dot(UR)
    return q,z,w,v,dc,fc

def cross_form(pA,pB,pC,w,v):
    ax,ay=f2sub(pB[0],pA[0]),f2sub(pB[1],pA[1])
    bx,by=f2sub(pC[0],pB[0]),f2sub(pC[1],pB[1])
    return f2sub(f2mul(ax,by,w,v),f2mul(ay,bx,w,v))
def side_form(pA,pB,pS,w,v):
    ax,ay=f2sub(pB[0],pA[0]),f2sub(pB[1],pA[1])
    sx,sy=f2sub(pS[0],pA[0]),f2sub(pS[1],pA[1])
    return f2sub(f2mul(ax,sy,w,v),f2mul(ay,sx,w,v))

def flags_forms(q,z,hull,w,v):
    """Flags with L-hat scaled normals: n = J(edge)/Lhat, Lhat a float constant
    >= sup of the edge length over the cell (verified: Lhat^2 >= hull-sup of
    |edge|^2).  Each flag equals beta*(unit flag), beta = len/Lhat in (0,1],
    which is SOUND for the box-kill certificate (see notes, Lemma WS-1).
    Returns (F, Lhats)."""
    F=[]; Lh=[]; m=len(hull)
    for i in range(m):
        Aj,Bj=hull[i],hull[(i+1)%m]
        ex,ey=f2sub(q[Bj][0],q[Aj][0]),f2sub(q[Bj][1],q[Aj][1])
        nx,ny=f2neg(ey),ex
        n2=f2add(f2mul(nx,nx,w,v),f2mul(ny,ny,w,v))
        H=f2hull(n2,w,v)
        assert H[0] > 0, "degenerate edge in cell"
        Lhat=_up(np.sqrt(H[1]))          # float constant >= sup edge length
        ivL=f2ival(iinv(ithin(Lhat)))    # thin interval 1/Lhat
        nx,ny=f2mul(nx,ivL,w,v),f2mul(ny,ivL,w,v)
        Lh.append(Lhat)
        for jj in (Aj,Bj):
            qx,qy=q[jj][0],q[jj][1]
            spin=f2add(f2mul(nx,f2neg(qy),w,v),f2mul(ny,qx,w,v))
            F.append([f2mul(z[jj],nx,w,v),f2mul(z[jj],ny,w,v),spin,nx,ny])
    return F, Lh

def verify_ppoint(F, cand, G, dc, w, v, dlo, eta_budget):
    """cand: dict subset(tuple)->(gi, lam0,lam1,lam2,lam3) each (6,k).
    Verifies lam>=0 on cell (exact float ineq) and ||E_k||_2 <= eta_budget*dlo.
    Returns (ok, min_pos_slack, max_eta_rel) with eta_rel = eta/(dlo)."""
    max_eta = 0.0; min_pos = np.inf
    for S,(gi,l0,l1,l2,l3) in cand.items():
        # positivity of lam(t,s) over cell
        slack = _dn(l0 - _up(np.abs(l1)*w + np.abs(l2)*v + np.abs(l3)*w*v))
        m=float(slack.min()); min_pos=min(min_pos,m)
        if m < 0: return False, m, np.inf
        k=len(gi); Gk=G[gi]
        lamf=[F2(ithin(l0[j]),ithin(l1[j]),ithin(l2[j]),ithin(l3[j]),np.zeros(k)) for j in range(6)]
        # sum of lam (should be ~1 with zero t,s parts)
        ssum=f2const(np.zeros(k))
        for j in range(6): ssum=f2add(ssum,lamf[j])
        sh=f2hull(ssum,w,v); slo=float(sh[0].min())
        if slo<=0.5: return False, m, np.inf
        eta2=np.zeros(k)
        for i in range(5):
            yc=ival(_dn(dc*Gk[:,i]),_up(dc*Gk[:,i]))
            acc=F2(ineg(yc), ithin(-Gk[:,i]), ithin(np.zeros(k)), ithin(np.zeros(k)), np.zeros(k))
            for j in range(6):
                acc=f2add(acc,f2mul(F[S[j]][i],lamf[j],w,v))
            h=f2hull(acc,w,v)
            eta2=_up(eta2+_up(np.maximum(np.abs(h[0]),np.abs(h[1]))**2))
        eta=_up(np.sqrt(eta2))
        # normalization correction: p = num/den; ||p-dg|| <= (eta+|den-1|*|dg|)/den_lo
        dgnorm=_up(np.sqrt((_up((dc+w)))**2*np.array([g@g for g in Gk])))
        denerr=_up(np.maximum(np.abs(sh[0]-1),np.abs(sh[1]-1)))
        etatot=_up((eta+denerr*dgnorm)/slo)
        rel=float((etatot/ _dn(np.array(dlo))).max())
        max_eta=max(max_eta,rel)
        if rel > eta_budget: return False, m, max_eta
    return True, min_pos, max_eta
