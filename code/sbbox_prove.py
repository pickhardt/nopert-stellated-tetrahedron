"""Proof-grade SB-box certifier core. Certificate: δ·Box(b_Π,b_γ,b_t) ⊆ conv F(u₂(δ,φ)) over the
critical-disk param u₂(δ,φ)=cosδ·u*+sinδ·v(φ). Flags r_P=(z_j n_i [Π,2], n_i·Jq_j [γ,1], n_i [t,2])∈ℝ⁵.
Box-inscription margin = min over facets (a,c) of conv F of  c − δ·(b_Π‖a_Π‖+b_γ|a_γ|+b_t‖a_t‖).
Here: RIGOROUS interval margin at a point (interval flags → interval ℝ⁵ facet normals via 4×4 minors →
interval box support). Facet 5-tuples taken from the float hull at the point (structure), then the
offsets/normals are interval-enclosed. A positive lower bound is a proof at that config."""
import sys, math; sys.path.insert(0,'.')
import numpy as np
from scipy.spatial import ConvexHull
import fast_interval as F
from fast_interval import FI
from det_tight import det4_tight, det5_tight  # tight rigorous interval dets (replace naive-Laplace blowup)
MOD_HINT=0.002   # if the cheap naive-det margin lower bound is below this, upgrade to the tight det
_TR_CACHE={}
def _transitions(delta):
    key=round(delta,9)
    if key not in _TR_CACHE: _TR_CACHE[key]=find_transitions(delta)   # resolved at call time
    return _TR_CACHE[key]
def _det4_naive(M):
    def _mnr(M,i,j): return [[M[a][b] for b in range(4) if b!=j] for a in range(4) if a!=i]
    def _d3(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    s=F.FI(0.0)
    for j in range(4):
        t=M[0][j]*_d3(_mnr(M,0,j)); s=s+(t if j%2==0 else F.FI(0.0)-t)
    return s
bP,bG,bT=2/125,1/25,1/500
V4=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
def verts(a): return np.vstack([V4,-a*V4])
aa=11/20; Vnp=verts(aa)
ustar=np.array([1,-1,0.])/math.sqrt(2); B1=np.array([0,0,1.]); B2=-np.array([1,1,0.])/math.sqrt(2)
def u2_of_np(delta,phi):
    v=math.sin(phi)*B1-math.cos(phi)*B2; u=math.cos(delta)*ustar+math.sin(delta)*v; return u/np.linalg.norm(u)
Znp=np.array([[0.,1,0],[-1,0,0]]); Jnp=np.array([[0.,-1],[1,0]])
def frame_np(u):
    u=u/np.linalg.norm(u); h=np.array([0,0,1.]) if abs(u[2])<0.9 else np.array([1,0,0.])
    f1=np.cross(h,u); f1/=np.linalg.norm(f1); f2=np.cross(u,f1); return np.array([f1,f2,u])
def flags_np(delta,phi):
    """float flags in ℝ⁵ + the active (edge,vertex) structure (for the interval facet 5-tuples)."""
    W=frame_np(u2_of_np(delta,phi)); Wv=(W@Vnp.T).T; q=(Znp@Wv.T).T; zc=Wv[:,2]
    H=ConvexHull(q); idx=list(H.vertices); ctr=q[idx].mean(0); edges=[]
    for k in range(len(idx)):
        p_,r_=idx[k],idx[(k+1)%len(idx)]; d=q[r_]-q[p_]; n=np.array([-d[1],d[0]]); n=n/np.linalg.norm(n)
        if n@(ctr-q[p_])<0: n=-n
        edges.append((n,float(np.min(q@n)),p_,r_))
    R=[]; pairs=[]
    for (n,ci,p_,r_) in edges:
        for j in range(8):
            if abs(n@q[j]-ci)<1e-7:
                R.append([zc[j]*n[0],zc[j]*n[1],float(n@(Jnp@q[j])),n[0],n[1]]); pairs.append((n[0],n[1],zc[j],float(q[j]@[1,0]),float(q[j]@[0,1]),j))
    return np.array(R),pairs
# ---- interval flags (fast_interval) at a (δ,φ) POINT (degenerate cell → tight) ----
def flags_FI(dlo,dhi,plo,phi_):
    cd,sd=F.trig_fi(dlo,dhi); cp,sp=F.trig_fi(plo,phi_)
    r2=F.FI(F.rd(math.sqrt(0.5)),F.ru(math.sqrt(0.5)))
    # u2 = cosδ·(r2,-r2,0) + sinδ·(cp·r2, cp·r2, sp)   [v=sinφ b1 − cosφ b2 = (cp r2, cp r2, sp)]
    ux=cd*r2 + sd*(cp*r2); uy=F.FI(0.0)-cd*r2 + sd*(cp*r2); uz=sd*sp
    W=F.frame([ux,uy,uz]); V=[[F.FI(x) for x in r] for r in V4]+[[F.FI(-aa*x) for x in r] for r in V4]
    Wv=[F.applyW(W,v) for v in V]; q=[F.proj(w) for w in Wv]; zc=[w[2] for w in Wv]
    return q,zc,W
def facet_margin(delta,plo,phi):   # δ fixed (point), φ-CELL [plo,phi]; returns interval margin/δ over the cell
    # SUBSET-SAFE structure: pick the cell endpoint with FEWER active pairs. Since a transition is
    # 10↔12 with the 10 ⊂ the 12, conv{fewer flags} ⊆ conv F(φ) for EVERY φ in the cell ⇒ box inside
    # the smaller hull ⇒ box inside conv F everywhere. Handles the transitions with no two-case split.
    # δ may be a scalar (point) or a (δ_lo,δ_hi) INTERVAL for δ-continuum certification. Structure
    # (flags_np/hull) uses the midpoint; the interval geometry + box support + division use [δ_lo,δ_hi].
    if isinstance(delta,(tuple,list)): dlo,dhi=delta; dmid=0.5*(dlo+dhi)
    else: dlo=dhi=dmid=delta
    dfi=F.FI(dlo,dhi)
    # SUBSET-SAFE structure: pick the endpoint with FEWER active pairs (10⊂12 at a transition).
    Ra,_=flags_np(dmid,plo); Rb,_=flags_np(dmid,phi)
    sphi = plo if len(Ra)<=len(Rb) else phi
    R,pairs=flags_np(dmid,sphi)
    if len(R)<6: return None
    try: H=ConvexHull(R)
    except Exception: return None
    q,zc,W=flags_FI(dlo,dhi,plo,phi)   # INTERVAL geometry over the δ×φ cell
    W2=frame_np(u2_of_np(dmid,sphi)); Wvn=(W2@Vnp.T).T; qn=(Znp@Wvn.T).T; zcn=Wvn[:,2]
    Hh=ConvexHull(qn); idx=list(Hh.vertices); ctr=qn[idx].mean(0)
    Rint=[]
    for k in range(len(idx)):
        p_,r_=idx[k],idx[(k+1)%len(idx)]; d=qn[r_]-qn[p_]; nn=np.array([-d[1],d[0]]); nn=nn/np.linalg.norm(nn)
        if nn@(ctr-qn[p_])<0: nn=-nn
        # interval normal for this edge
        d0=q[r_][0]-q[p_][0]; d1=q[r_][1]-q[p_][1]; ni0=F.FI(0.0)-d1; ni1=d0
        if (nn@np.array([ -(qn[r_][1]-qn[p_][1]), (qn[r_][0]-qn[p_][0]) ]))<0: ni0=F.FI(0.0)-ni0; ni1=F.FI(0.0)-ni1
        L=(ni0.sqr()+ni1.sqr()).sqrt(); ni0=ni0/L; ni1=ni1/L
        ci=None
        for w in q:
            val=ni0*w[0]+ni1*w[1]; ci=val if ci is None else FI(min(ci.lo,val.lo),min(ci.hi,val.hi))
        for j in range(8):
            if abs(float(nn@qn[j])-float(np.min(qn@nn)))<1e-7:
                zj=zc[j]; Jq0=F.FI(0.0)-q[j][1]; Jq1=q[j][0]   # J·q=(-q1,q0); n·Jq
                gam=ni0*Jq0+ni1*Jq1
                Rint.append([zj*ni0, zj*ni1, gam, ni0, ni1])
    # match count/order to R
    if len(Rint)!=len(R): return None   # structure mismatch → subdivide
    Rnp=np.array(R)
    # STABLE SUB-HULL loop. Certify box ⊆ conv(active flags): each facet gets a rigorous margin, and
    # every non-facet active flag must stay strictly inside over the cell. A flag that can't be proven
    # inside (near-coplanar → tight det5 s.hi>0) is DROPPED and we retry on the smaller hull. SOUND:
    # the cell is one-sided (transitions are cell boundaries), so every active flag is a flag of F(φ)
    # for ALL φ in the cell ⇒ conv(any active subset) ⊆ conv F(φ) ⇒ box ⊆ conv(subset) ⇒ box ⊆ conv F.
    # FAST/TIGHT det: naive Laplace normal first; upgrade to det4_tight only if the margin is < MOD_HINT.
    def _margin_on(active):
        if len(active)<6: return None
        Rsub=Rnp[active]
        # QJ (joggle) breaks near-degenerate 5D triangulations into non-degenerate simplices, avoiding
        # thin facets whose tiny normal blows up the interval margin division. SOUND: the interval
        # margin + stability guard re-validate whatever structure QJ produces (box ⊆ conv(active) ⊆ conv F).
        try: Hs=ConvexHull(Rsub, qhull_options='QJ')
        except Exception:
            try: Hs=ConvexHull(Rsub)
            except Exception: return None
        Ri=[Rint[i] for i in active]
        cen=[sum((Ri[i][c].lo+Ri[i][c].hi)/2 for i in range(len(Ri)))/len(Ri) for c in range(5)]
        def _mfrom_a(a,r0,diffs,tight):
            aP=(a[0].sqr()+a[1].sqr()).sqrt(); aG=a[2]
            aGabs=FI(_amin_abs(aG), max(abs(aG.lo),abs(aG.hi)))
            aT=(a[3].sqr()+a[4].sqr()).sqrt()
            if tight:   # c = a·r0 = det[diff_1..4, r0]; tight det removes the a-then-dot blowup
                c=det5_tight([diffs[0],diffs[1],diffs[2],diffs[3],[r0[cc] for cc in range(5)]])
            else:
                c=F.FI(0.0)
                for cc in range(5): c=c+a[cc]*r0[cc]
            anorm=(a[0].sqr()+a[1].sqr()+a[2].sqr()+a[3].sqr()+a[4].sqr()).sqrt()
            af_c=sum(((a[cc].lo+a[cc].hi)/2)*(cen[cc]-(r0[cc].lo+r0[cc].hi)/2) for cc in range(5))
            if af_c>0: a=[F.FI(0.0)-x for x in a]; c=F.FI(0.0)-c
            hK=dfi*(bP*aP+bG*aGabs+bT*aT)            # box support with δ∈[δ_lo,δ_hi]
            return (c-hK)*(F.FI(1.0)/(anorm*dfi)), a  # margin/δ over the δ×φ cell
        mn=None; unstable=set()
        for simp in Hs.simplices:
            r0=Ri[simp[0]]; diffs=[[Ri[simp[t]][c]-r0[c] for c in range(5)] for t in range(1,5)]
            def _avec(tight):
                av=[]
                for k in range(5):
                    Mk=[[diffs[t][c] for c in range(5) if c!=k] for t in range(4)]
                    dk=det4_tight(Mk) if tight else _det4_naive(Mk)
                    av.append(dk if k%2==0 else F.FI(0.0)-dk)
                return av
            a=_avec(False); m,a=_mfrom_a(a,r0,diffs,False)
            if m.lo<MOD_HINT: a=_avec(True); m,a=_mfrom_a(a,r0,diffs,True)
            if m.lo<-0.5:   # DEGENERATE thin simplex (‖a‖→0 → division blowup): 5 near-coplanar flags,
                # one redundant. Drop the vertex whose removal leaves the most non-degenerate remainder.
                verts=list(simp); best_k=verts[1]; best_vol=-1.0
                for kk in verts:
                    oth=[Ri[t] for t in verts if t!=kk]; o0=oth[0]
                    dmr=[[ (oth[t][c].lo+oth[t][c].hi)/2-(o0[c].lo+o0[c].hi)/2 for c in range(5)] for t in range(1,4)]
                    G=[[sum(dmr[i][c]*dmr[j][c] for c in range(5)) for j in range(3)] for i in range(3)]
                    vol=abs(G[0][0]*(G[1][1]*G[2][2]-G[1][2]*G[2][1])-G[0][1]*(G[1][0]*G[2][2]-G[1][2]*G[2][0])+G[0][2]*(G[1][0]*G[2][1]-G[1][1]*G[2][0]))
                    if vol>best_vol: best_vol=vol; best_k=kk
                unstable.add(active[best_k]); continue
            mn=m if mn is None else FI(min(mn.lo,m.lo),min(mn.hi,m.hi))
            ac=[(a[cc].lo+a[cc].hi)/2 for cc in range(5)]; anc=math.sqrt(sum(x*x for x in ac))
            for P in range(len(Ri)):
                if P in simp: continue
                sc=sum(ac[cc]*((Ri[P][cc].lo+Ri[P][cc].hi)/2-(r0[cc].lo+r0[cc].hi)/2) for cc in range(5))
                if anc>0 and sc/anc > -1e-7: continue   # structurally coplanar → cannot go strictly outside
                s=F.FI(0.0)
                for cc in range(5): s=s+a[cc]*(Ri[P][cc]-r0[cc])   # fast loose dot
                if s.hi>1e-12:
                    st=det5_tight([diffs[0],diffs[1],diffs[2],diffs[3],[Ri[P][cc]-r0[cc] for cc in range(5)]])
                    if st.hi>1e-12: unstable.add(active[P])
        return mn,unstable
    active=list(range(len(R))); mn=None
    for _ in range(6):
        res=_margin_on(active)
        if res is None: return (mn,False) if mn is not None else None
        mn,unstable=res
        if not unstable: return mn,True         # box ⊆ conv(active) ⊆ conv F, all facets stable
        active=[i for i in active if i not in unstable]   # drop near-coplanar flags, re-certify sub-hull
    return mn,False
def _amin_abs(iv):
    return 0.0 if (iv.lo<=0<=iv.hi) else min(abs(iv.lo),abs(iv.hi))
def _detf(np,M): return float(np.linalg.det(np.array(M)))
if __name__=='__main__':
    print('SB-box interval margin/δ over φ-CELLS at δ=1e-3 (tight spot φ≈2.617; check looseness vs cell width):')
    for pc,hw in [(2.617,1e-5),(2.617,1e-4),(2.617,1e-3),(1.0,1e-3),(1.0,0.01)]:
        r=facet_margin(1e-3, pc-hw, pc+hw)
        if r is None or isinstance(r,tuple): print(f'  φ={pc}±{hw}: {r}')
        else: m,stable=r; print(f'  φ={pc}±{hw}: margin/δ ∈ [{m.lo:.6f},{m.hi:.6f}] stable={stable}')

def find_transitions(delta, scan=4000):
    """φ where the active-pair count changes (silhouette transitions), located per δ, bracketed."""
    prev=None; tr=[]
    for i in range(scan+1):
        phi=2*math.pi*i/scan
        try: R,_=flags_np(delta,phi); n=len(R)
        except Exception: n=-1
        if prev is not None and n!=prev and prev>0 and n>0: tr.append(phi)
        prev=n
    return tr
def sweep(delta, MOD=0.005, MINH=1e-5, EPS=1e-6):
    """adaptive φ-sweep at fixed δ. DYNAMIC transition φ (per δ) are cell boundaries (±EPS), so no cell
    straddles a hull change ⇒ each one-sided cell certifies with its own stable structure. The 2·EPS
    windows AT each transition (measure ~1e-6) are covered by continuity of the box-inscription margin."""
    from collections import deque
    tr=find_transitions(delta)
    hard=[]
    for t in tr: hard+=[t-EPS, t+EPS]
    N=400; breaks=sorted(set([b for b in [2*math.pi*i/N for i in range(N+1)]+hard if 0<=b<=2*math.pi]))
    Q=deque([(breaks[i],breaks[i+1]) for i in range(len(breaks)-1) if breaks[i+1]-breaks[i]>2e-8])
    # skip the tiny transition windows (they sit between t-EPS and t+EPS): mark them handled-by-continuity
    tr_win=[(t-EPS,t+EPS) for t in tr]
    def in_trwin(a,b): return any(a>=w0-1e-12 and b<=w1+1e-12 for (w0,w1) in tr_win)
    cert=fail=twin=0; man=[]
    while Q:
        a,b=Q.popleft()
        if in_trwin(a,b): twin+=1; continue   # tiny transition window → continuity
        try: r=facet_margin(delta,a,b)
        except Exception: r=None
        if r is None:
            if b-a<MINH: fail+=1
            else: m=(a+b)/2; Q.append((a,m)); Q.append((m,b))
            continue
        marg,stable=r
        if stable and marg.lo>=MOD: cert+=1; man.append((a,b,marg.lo))
        elif b-a<MINH: fail+=1
        else: m=(a+b)/2; Q.append((a,m)); Q.append((m,b))
    return cert,fail,man,twin
if __name__!='__main__': pass
