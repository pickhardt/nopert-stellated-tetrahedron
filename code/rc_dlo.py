"""
rc_dlo.py -- certified depth lower bounds d_lo(u2) over (delta,phi) windows in the
DK chart u2(delta,phi) = exp(delta[w(phi)]) u*, u* = (1,-1,0)/sqrt2, for P_{11/20};
and the RC-table generator RC = d_lo/(M0*delta_hi) for the DK sweep corner-skip.

Soundness architecture (Lemma WD-1 / WD-2, see notes_round_91_walldip_handoff.md):
  * outward-rounded interval arithmetic (libm-axiom for sin/cos endpoint values,
    inflated by 4 ulp, standard for this program);
  * flag families given by explicit edge lists; an edge's incidences are ACTIVE at a
    parameter as soon as all 8 projected points lie weakly inside its halfplane
    (supporting line => hull edge => active incidences); we verify STRICT side tests
    for all non-exempt points by intervals over the whole window;
  * ball certificate: cube generators g = c*sigma, sigma in {-1,1}^5
    (conv{g} contains B(0,c)); float LP weights lambda >= 0 at window center;
    interval residual eta_m >= sup_W || (sum_i lambda_i r_i)/(sum lambda) - g_m ||_2.
    Then B(0, c - max eta) subset conv F(param) for EVERY param in the window,
    PROVIDED the family's validity tests pass over the window.  (The true point
    P(param) = normalized combination of TRUE flags lies in conv F(param) and within
    eta of g_m; support function h(x) >= c||x||_1 - eta||x||_2 >= (c-eta)||x||_2.)
  * two-case wall windows: family A = full pent cycle [0,3,4,1,7] (thin side),
    family B = hex cycle minus short edge = edges (0,3),(3,4),(4,2),(1,7),(7,0).
    Exempt scalar: slack of point 2 (v3) against pent edge (4,1) =: sA, and slack of
    point 1 (v2) against edge (4,2) =: sB satisfy  sA*|q1-q4| = -sB*|q2-q4|
    (both are +/- cross(q2-q4, q1-q4)), so at every parameter sA>=0 or sB>=0.
    If sA>=0 all pent side tests hold weakly => F_A active; if sB>=0 all B-side
    tests hold weakly => F_B active.  Hence d >= min(d_lo(F_A), d_lo(F_B)).
    (The mirror wall pentB/hex at phi near pi - phi_w is handled by the same code
    with the mirrored cycles, auto-detected from the float hull.)
"""
import math
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

# ---------------- interval arithmetic ----------------
INF = float('inf')
def _up(x):
    for _ in range(1): x = math.nextafter(x, INF)
    return x
def _dn(x):
    for _ in range(1): x = math.nextafter(x, -INF)
    return x

class I:
    __slots__=('lo','hi')
    def __init__(self,lo,hi=None):
        if hi is None: hi=lo
        self.lo=lo; self.hi=hi
        assert lo<=hi
    def __repr__(self): return "[%.17g,%.17g]"%(self.lo,self.hi)
    def __add__(s,o):
        o=asI(o); return I(_dn(s.lo+o.lo), _up(s.hi+o.hi))
    __radd__=__add__
    def __neg__(s): return I(-s.hi,-s.lo)
    def __sub__(s,o):
        o=asI(o); return I(_dn(s.lo-o.hi), _up(s.hi-o.lo))
    def __rsub__(s,o): return asI(o)-s
    def __mul__(s,o):
        o=asI(o)
        ps=(s.lo*o.lo, s.lo*o.hi, s.hi*o.lo, s.hi*o.hi)
        return I(_dn(min(ps)), _up(max(ps)))
    __rmul__=__mul__
    def __truediv__(s,o):
        o=asI(o)
        assert o.lo>0 or o.hi<0, "division by interval containing 0"
        ps=(s.lo/o.lo, s.lo/o.hi, s.hi/o.lo, s.hi/o.hi)
        return I(_dn(min(ps)), _up(max(ps)))
    def sq(s):
        if s.lo>=0: return I(_dn(s.lo*s.lo), _up(s.hi*s.hi))
        if s.hi<=0: return I(_dn(s.hi*s.hi), _up(s.lo*s.lo))
        m=max(-s.lo,s.hi); return I(0.0,_up(m*m))
    def sqrt(s):
        lo=max(s.lo,0.0)   # sound: used only on enclosures of true squares/norms
        assert s.hi>=0
        return I(_dn(math.sqrt(lo)), _up(math.sqrt(s.hi)))
    def mag(s): return max(abs(s.lo),abs(s.hi))
def asI(x): return x if isinstance(x,I) else I(float(x))

def _trig_widen(x, k=4):
    lo,hi=x
    for _ in range(k):
        lo=math.nextafter(lo,-INF); hi=math.nextafter(hi,INF)
    return I(max(lo,-1.0),min(hi,1.0))
def isin(x):
    a,b=x.lo,x.hi
    assert b-a < math.pi, "interval too wide for sin"
    vals=[math.sin(a),math.sin(b)]
    # critical points pi/2 + k*pi
    k0=math.ceil((a-math.pi/2)/math.pi); k1=math.floor((b-math.pi/2)/math.pi)
    for k in range(k0,k1+1):
        vals.append(1.0 if k%2==0 else -1.0)
    return _trig_widen((min(vals),max(vals)))
def icos(x):
    a,b=x.lo,x.hi
    assert b-a < math.pi, "interval too wide for cos"
    vals=[math.cos(a),math.cos(b)]
    k0=math.ceil(a/math.pi); k1=math.floor(b/math.pi)
    for k in range(k0,k1+1):
        vals.append(1.0 if k%2==0 else -1.0)
    return _trig_widen((min(vals),max(vals)))

def ivec(*xs): return [asI(x) for x in xs]
def vadd(u,v): return [a+b for a,b in zip(u,v)]
def vsub(u,v): return [a-b for a,b in zip(u,v)]
def vscale(c,u): return [asI(c)*a for a in u]
def vdot(u,v):
    s=asI(0.0)
    for a,b in zip(u,v): s=s+a*b
    return s
def mvec(M,v): return [vdot(row,v) for row in M]
def mmul(A,B):
    n=len(A); m=len(B[0]); k=len(B)
    return [[vdot(A[i],[B[t][j] for t in range(k)]) for j in range(m)] for i in range(n)]

# ---------------- geometry (exact constants as intervals) ----------------
SQ2 = I(2.0,2.0).sqrt()                # sqrt(2) enclosure
ISQ2 = asI(1.0)/SQ2
A_ST = asI(11.0)/asI(20.0)             # 11/20 enclosure (exact: 0.55 rounded outward)
E1T = [asI(0.0),asI(0.0),asI(1.0)]
E2T = [-ISQ2,-ISQ2,asI(0.0)]
WSTAR = [E1T, E2T, [ISQ2,-ISQ2,asI(0.0)]]
VT=[[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]
def vertices_I():
    Vs=[]
    for r in VT: Vs.append([asI(float(x)) for x in r])
    for r in VT: Vs.append([-(A_ST*asI(float(x))) for x in r])
    return Vs
VI = vertices_I()

def skew_I(w):
    z=asI(0.0)
    return [[z,-w[2],w[1]],[w[2],z,-w[0]],[-w[1],w[0],z]]

def frame_I(dl, ph):
    """Interval frame W(delta,phi) (3x3 interval matrix), dl, ph intervals."""
    sph, cph = isin(ph), icos(ph)
    w=[ -(sph*E1T[i]) + cph*E2T[i] for i in range(3)]
    K=skew_I(w); K2=mmul(K,K)
    sd, cd = isin(dl), icos(dl)
    one_m_cd = asI(1.0)-cd
    R=[[ (asI(1.0) if i==j else asI(0.0)) + sd*K[i][j] + one_m_cd*K2[i][j]
         for j in range(3)] for i in range(3)]
    # W = WSTAR @ R^T
    RT=[[R[j][i] for j in range(3)] for i in range(3)]
    return mmul(WSTAR,RT)

def shadow_I(dl,ph):
    W=frame_I(dl,ph)
    q=[]; z=[]
    for v in VI:
        p=mvec(W,v)
        q.append([p[0],p[1]]); z.append(p[2])
    return q,z

def cross2(u,v): return u[0]*v[1]-u[1]*v[0]

def flags_I(q,z,edges):
    """Unit-normalized interval flags for the given (ccw) edge list."""
    F=[]
    for (A,B) in edges:
        e=vsub(q[B],q[A])
        nraw=[-e[1],e[0]]                      # J e  (inward for ccw)
        ln=(nraw[0].sq()+nraw[1].sq()).sqrt()
        n=[nraw[0]/ln, nraw[1]/ln]
        for j in (A,B):
            Jq=[-q[j][1],q[j][0]]
            F.append([z[j]*n[0], z[j]*n[1], n[0]*Jq[0]+n[1]*Jq[1], n[0], n[1]])
    return F

def side_slacks_I(q,edges,exempt=()):
    """For each edge, interval slack cross(qB-qA, qk-qA) for every k not in edge.
       Returns list of ((edge,k), slack). exempt: set of (edge,k) to skip."""
    out=[]
    for (A,B) in edges:
        e=vsub(q[B],q[A])
        for k in range(8):
            if k==A or k==B or ((A,B),k) in exempt: continue
            s=cross2(e,vsub(q[k],q[A]))
            out.append((((A,B),k),s))
    return out

# ---------------- float center machinery ----------------
Vf = np.vstack([np.array(VT,float), -0.55*np.array(VT,float)])
Wstar_f = np.array([[0,0,1],[-1/math.sqrt(2),-1/math.sqrt(2),0],
                    [1/math.sqrt(2),-1/math.sqrt(2),0]])
def frame_f(delta,phi):
    w = -math.sin(phi)*Wstar_f[0] + math.cos(phi)*Wstar_f[1]
    K = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    return Wstar_f @ (np.eye(3)+math.sin(delta)*K+(1-math.cos(delta))*(K@K)).T
def shadow_f(delta,phi):
    P=Vf@frame_f(delta,phi).T
    return P[:,:2],P[:,2]
Jf=np.array([[0,-1],[1,0]])
def flags_f(q,z,edges):
    F=[]
    for (A,B) in edges:
        e=q[B]-q[A]; n=Jf@e; n/=np.linalg.norm(n)
        for j in (A,B): F.append([z[j]*n[0],z[j]*n[1],n@(Jf@q[j]),n[0],n[1]])
    return np.array(F)


CUBE=None  # retired: cube generators c*sigma lie outside the hull (norm c*sqrt5)

AXDIRS=[]
for j in range(5):
    for s in (1.0,-1.0):
        v=[0.0]*5; v[j]=s; AXDIRS.append(np.array(v))

def ball_lambdas(Fc, shrink=0.85):
    """Per-axis-direction float reaches and LP weights.
       Returns list of (g, lam) for generators g = shrink*reach*dir; None if any LP fails."""
    n=len(Fc)
    out=[]
    for x in AXDIRS:
        # float reach: max t s.t. t*x in conv Fc  <=>  LP max t, sum lam Fc = t x, lam>=0, sum lam=1
        A_eq=np.vstack([np.hstack([Fc.T, -x.reshape(5,1)]), np.append(np.ones(n),0.0)])
        b_eq=np.append(np.zeros(5),1.0)
        cobj=np.zeros(n+1); cobj[-1]=-1.0
        r=linprog(cobj,A_eq=A_eq,b_eq=b_eq,bounds=[(0,None)]*n+[(0,None)],method='highs')
        if not r.success or r.x[-1]<=0: return None
        t=shrink*r.x[-1]
        g=t*x
        r2=linprog(np.ones(n),A_eq=np.vstack([Fc.T,np.ones((1,n))]),
                   b_eq=np.append(g,1.0),bounds=[(0,None)]*n,method='highs')
        if not r2.success: return None
        out.append((g,r2.x))
    return out

def ball_cert_I(FI, gens):
    """Certified d_lo from interval flags + float (g,lam) generator witnesses.
       For each generator: true point P=sum lam_i r_i / sum lam in conv F(param),
       ||P-g||<=eta (interval). rho_j := min over the two +-e_j generators of
       (|g_j| - eta).  d_lo = (sum rho_j^{-2})^{-1/2}, computed with outward rounding."""
    rho=[INF]*5
    for (g,lam) in gens:
        S=asI(0.0); P=[asI(0.0)]*5
        for li,r in zip(lam,FI):
            if li<=0: continue
            liI=asI(float(li)); S=S+liI
            for j in range(5): P[j]=P[j]+liI*r[j]
        if S.lo<=0: return None
        nrm2=asI(0.0)
        for j in range(5):
            nrm2=nrm2+(P[j]/S-asI(float(g[j]))).sq()
        eta=nrm2.sqrt().hi
        j=int(np.argmax(np.abs(g)))
        rj=abs(g[j])-eta
        if rj<=0: return None
        rho[j]=min(rho[j],rj)
    # d_lo = (sum rho_j^-2)^(-1/2) outward-rounded downward
    ssum=asI(0.0)
    for rj in rho:
        ssum=ssum+asI(1.0)/asI(rj).sq()
    dlo=(asI(1.0)/ssum).sqrt().lo
    return dlo if dlo>0 else None

# ---------------- window certifiers ----------------
PENT_A=[(0,3),(3,4),(4,1),(1,7),(7,0)]      # thin side, phi < phi_w  (v3=2 interior)
FAM_B =[(0,3),(3,4),(4,2),(1,7),(7,0)]      # hex minus short edge (2,1)
EXEMPT_A={((4,1),2)}                        # sA: slack of v3 vs pent edge (p1,v2)
EXEMPT_B={((4,2),1)}                        # sB: slack of v2 vs edge (p1,v3)
# mirrored wall (pentB/hex near pi-phi_w): transitioning vertex v2<->v3 swap roles
PENT_A2=[(0,3),(3,4),(4,2),(2,7),(7,0)]     # thin side, phi > pi-phi_w (v2=1 interior)
FAM_B2 =[(0,3),(3,4),(4,2),(1,7),(7,0)]     # hex minus short edge (2,1)
EXEMPT_A2={((2,7),1)}                       # sA2 = slack of v2 vs pent edge (v3,p4)
EXEMPT_B2={((1,7),2)}                       # sB2 = slack of v3 vs edge (v2,p4); sA2_raw = -sB2_raw

def _validate(q, edges, exempt):
    for key,s in side_slacks_I(q,edges,exempt):
        if not s.lo>0: return key
    return None

def dlo_family(dl, ph, edges, exempt, c_frac=0.8):
    """Certified d_lo for one family over window; None on any failure."""
    dc,pc=0.5*(dl.lo+dl.hi),0.5*(ph.lo+ph.hi)
    qf,zf=shadow_f(dc,pc)
    Fc=flags_f(qf,zf,edges)
    try:
        d_float=-ConvexHull(Fc).equations[:,-1].max()
    except Exception:
        return None
    if d_float<=0: return None
    q,z=shadow_I(dl,ph)
    if _validate(q,edges,exempt) is not None: return None
    FI=flags_I(q,z,edges)
    gens=ball_lambdas(Fc)
    if gens is None: return None
    return ball_cert_I(FI,gens)

def dlo_window(dl, ph):
    """Certified d_lo over window; auto single-family or two-case. None => split."""
    dc,pc=0.5*(dl.lo+dl.hi),0.5*(ph.lo+ph.hi)
    qf,zf=shadow_f(dc,pc)
    cyc=list(ConvexHull(qf).vertices)
    wall1 = abs(pc-math.atan(11*math.sqrt(2)/9))
    wall2 = abs(pc-(math.pi-math.atan(11*math.sqrt(2)/9)))
    near1 = wall1 < wall2
    if len(cyc)==6 and min(wall1,wall2)>0.2:
        # deep hex: single family = hex minus short edge (still subset-safe)
        fam = FAM_B if near1 else FAM_B2
        return dlo_family(dl,ph,fam,set())
    # near a wall or pent: two-case (also sound deep in pent: case B never fires)
    if near1:
        dA=dlo_family(dl,ph,PENT_A,EXEMPT_A)
        if dA is None: return None
        dB=dlo_family(dl,ph,FAM_B,EXEMPT_B)
        if dB is None: return None
    else:
        dA=dlo_family(dl,ph,PENT_A2,EXEMPT_A2)
        if dA is None: return None
        dB=dlo_family(dl,ph,FAM_B2,EXEMPT_B2)
        if dB is None: return None
    return min(dA,dB)

def dlo_shard(d0,d1,p0,p1, min_dw_rel=1e-3, min_pw=1e-5, max_cells=20000, quality=0.4):
    """Adaptive certified min of d_lo over shard [d0,d1]x[p0,p1].
       quality: keep splitting a passing window until d_lo >= quality * float-truth
       at its center (or width floors reached; then any positive d_lo is accepted).
       Returns (d_lo_min, ncells, nstuck); d_lo_min=None if stuck cells remain."""
    stack=[(d0,d1,p0,p1)]
    dmin=INF; ncells=0; stuck=[]
    def truth(a,b,pA,pB):
        from scipy.spatial import ConvexHull as CH
        qf,zf=shadow_f(0.5*(a+b),0.5*(pA+pB))
        cyc=list(CH(qf).vertices)
        F=flags_f(qf,zf,[(cyc[k],cyc[(k+1)%len(cyc)]) for k in range(len(cyc))])
        return -CH(F).equations[:,-1].max()
    while stack:
        a,b,pA,pB=stack.pop()
        if ncells>max_cells: return None,ncells,len(stack)+len(stuck)
        r=dlo_window(I(a,b),I(pA,pB))
        ncells+=1
        rd=(b-a)/(0.5*(a+b)); pw=(pB-pA)
        at_floor = (rd<=min_dw_rel and pw<=min_pw)
        if r is not None:
            if at_floor or r >= quality*max(truth(a,b,pA,pB),1e-300):
                dmin=min(dmin,r); continue
            # else: fall through to split for a tighter value
        # split the wider direction: relative delta-width vs phi-width (both target ~2e-3)
        rd=(b-a)/(0.5*(a+b)); pw=(pB-pA)
        if rd>=pw and rd>min_dw_rel:
            m=0.5*(a+b); stack.append((a,m,pA,pB)); stack.append((m,b,pA,pB))
        elif pw>min_pw:
            m=0.5*(pA+pB); stack.append((a,b,pA,m)); stack.append((a,b,m,pB))
        else:
            stuck.append((a,b,pA,pB))
    if stuck: return None,ncells,len(stuck)
    return dmin,ncells,0

M0=1.0
def rc_of_shard(d0,d1,p0,p1,**kw):
    dlo,nc,nst=dlo_shard(d0,d1,p0,p1,**kw)
    if dlo is None: return None,nc,nst
    return _dn(dlo/(M0*d1)), nc, nst


def dlo_shard_rows(d0,d1,p0,p1,**kw):
    """Like dlo_shard but also returns the leaf rows (a,b,pA,pB,d_lo|None).
       Replay verification = rerun dlo_window on each row; no stored witnesses
       needed (the certificate is deterministically re-derivable from the window)."""
    stack=[(d0,d1,p0,p1)]; rows=[]; ncells=0
    min_dw_rel=kw.get('min_dw_rel',1e-3); min_pw=kw.get('min_pw',1e-5)
    quality=kw.get('quality',0.4); max_cells=kw.get('max_cells',200000)
    from scipy.spatial import ConvexHull as CH
    def truth(a,b,pA,pB):
        qf,zf=shadow_f(0.5*(a+b),0.5*(pA+pB))
        cyc=list(CH(qf).vertices)
        F=flags_f(qf,zf,[(cyc[k],cyc[(k+1)%len(cyc)]) for k in range(len(cyc))])
        return -CH(F).equations[:,-1].max()
    while stack:
        a,b,pA,pB=stack.pop(); ncells+=1
        if ncells>max_cells: return None,rows
        r=dlo_window(I(a,b),I(pA,pB))
        rd=(b-a)/(0.5*(a+b)); pw=(pB-pA)
        at_floor=(rd<=min_dw_rel and pw<=min_pw)
        if r is not None and (at_floor or r>=quality*max(truth(a,b,pA,pB),1e-300)):
            rows.append((a,b,pA,pB,r)); continue
        if rd>=pw and rd>min_dw_rel:
            m=0.5*(a+b); stack.append((a,m,pA,pB)); stack.append((m,b,pA,pB))
        elif pw>min_pw:
            m=0.5*(pA+pB); stack.append((a,b,pA,m)); stack.append((a,b,m,pB))
        else:
            rows.append((a,b,pA,pB,None))   # stuck
    return min((x[4] for x in rows if x[4] is not None),default=None), rows

def emit_rc_manifest(path, shards, **kw):
    """shards: list of (d0,d1,p0,p1). Writes tab-separated manifest with header.
       Row types: RC (shard summary) and W (window leaf).  RC = d_lo/(M0*d1)."""
    with open(path,'w') as f:
        f.write("# rc_dlo manifest v1; M0=1; chart u2(delta,phi)=exp(delta[w(phi)])u*\n")
        f.write("# W a b phiA phiB dlo   |   RC d0 d1 p0 p1 dlo_min RC nstuck\n")
        for (d0,d1,p0,p1) in shards:
            dmin,rows=dlo_shard_rows(d0,d1,p0,p1,**kw)
            nst=sum(1 for r in rows if r[4] is None)
            for r in rows:
                f.write("W\t%.17g\t%.17g\t%.17g\t%.17g\t%s\n"%(r[0],r[1],r[2],r[3],
                        "STUCK" if r[4] is None else "%.17g"%r[4]))
            rc = None if (dmin is None or nst>0) else _dn(dmin/d1)
            f.write("RC\t%.17g\t%.17g\t%.17g\t%.17g\t%s\t%s\t%d\n"%(
                    d0,d1,p0,p1, "NA" if dmin is None else "%.17g"%dmin,
                    "NA" if rc is None else "%.17g"%rc, nst))
    return path
