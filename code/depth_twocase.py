"""Two-case depth certificate for silhouette-transition (hull-unstable) cells.
At a transition cell the shadow-hull combinatorics change, so the plain certifier's hull_stable fails
and subdividing loops forever. Fix: enumerate the candidate edge-orders (silhouette combinatorics) that
occur in the cell, and certify d_lo>0 UNDER EACH. Sound: every u2 in the cell has its true hull among
the candidates, so d(u2)=inradius(conv F(u2)) >= min_C d_lo^C > 0.
 - dlo_cert_eo: depth_grid.dlo_cert's R^5 facet-depth cert but with a GIVEN edge-order eo (the plain
   2-D hull_stable is skipped; the R^5 flag-hull stability per facet is KEPT — that's what validates
   the cert for that combinatorics over the cell).
 - candidate_eos: the distinct edge-orders seen over the cell (corners+center+edge-midpoints). [rigor
   note: for a proof this must be the AMBIGUOUS-VERTEX enumeration; sampling first to validate.]"""
import sys, math
sys.path.insert(0,'.')
import numpy as np
from fractions import Fraction as Fr
from scipy.spatial import ConvexHull
import fast_interval as F
from fast_interval import FI
import ball_cover as BC, ball_cover5 as B5
import depth_grid as DG
from det_tight import det4_tight, det5_tight   # tight interval dets: fix wall-cell facet-degenerate/unstable
aa=Fr(11,20)

def dlo_cert_eo(th,thw,ph,phw,eo):
    """R^5 facet-depth cert with a FIXED edge-order eo. Returns (d, ok, why)."""
    W=F.frame(F.iu2(th-thw,th+thw,ph-phw,ph+phw)); V=F.verts(aa)
    Wv=[F.applyW(W,v) for v in V]; q=[F.proj(w) for w in Wv]; zc=[w[2] for w in Wv]
    u2c=BC.u2center(th,ph); W2=BC.frame_np(u2c); Wvn=(W2@BC.V.T).T; qn=(BC.Z@Wvn.T).T; zcn=Wvn[:,2]
    Rfl=[]; Rint=[]
    for (p_,r_,sgn) in eo:
        dn=qn[r_]-qn[p_]; nn=np.array([-dn[1],dn[0]])
        if sgn<0: nn=-nn
        nl=np.linalg.norm(nn)
        if nl<1e-12: return None,False,'eo-degenerate'
        nn=nn/nl
        d0=q[r_][0]-q[p_][0]; d1=q[r_][1]-q[p_][1]
        ni0=F.FI(0.0)-d1; ni1=d0
        if sgn<0: ni0=F.FI(0.0)-ni0; ni1=F.FI(0.0)-ni1
        L=(ni0.sqr()+ni1.sqr()).sqrt()
        if L.lo<=0: return None,False,'edge-degenerate'
        ni0=ni0/L; ni1=ni1/L
        for j in (p_,r_):
            Rfl.append([zcn[j]*nn[0],zcn[j]*nn[1],float(nn@np.array([-qn[j][1],qn[j][0]])),nn[0],nn[1]])
            Jq0=F.FI(0.0)-q[j][1]; Jq1=q[j][0]
            Rint.append([zc[j]*ni0, zc[j]*ni1, ni0*Jq0+ni1*Jq1, ni0, ni1])
    Rfl=np.array(Rfl)
    if len(Rfl)<6: return None,False,'few-flags'
    def _depth_on(active):
        Rsub=Rfl[active]
        if len(Rsub)<6: return ('fail','subhull-small')
        try: H=ConvexHull(Rsub)
        except Exception: return ('fail','r5-hull')
        Ri=[Rint[i] for i in active]
        cen=[sum((Ri[i][c].lo+Ri[i][c].hi)/2 for i in range(len(Ri)))/len(Ri) for c in range(5)]
        dmin=None; unstable=set()
        for simp in H.simplices:
            r0=Ri[simp[0]]
            diffs=[[Ri[simp[t]][c]-r0[c] for c in range(5)] for t in range(1,5)]
            a=[]
            for k in range(5):
                Mk=[[diffs[t][c] for c in range(5) if c!=k] for t in range(4)]
                dk=det4_tight(Mk); a.append(dk if k%2==0 else F.FI(0.0)-dk)
            c=det5_tight([diffs[0],diffs[1],diffs[2],diffs[3],[r0[cc] for cc in range(5)]])
            af_c=sum(((a[cc].lo+a[cc].hi)/2)*(cen[cc]-(r0[cc].lo+r0[cc].hi)/2) for cc in range(5))
            if af_c>0: a=[F.FI(0.0)-x for x in a]; c=F.FI(0.0)-c
            anorm=(a[0].sqr()+a[1].sqr()+a[2].sqr()+a[3].sqr()+a[4].sqr()).sqrt()
            if anorm.lo<=0: return ('fail','facet-degenerate')
            if c.lo<=0: return ('fail','origin-outside')
            dfac=c/anorm
            dmin=dfac if dmin is None else FI(min(dmin.lo,dfac.lo),min(dmin.hi,dfac.hi))
            ac=[(a[cc].lo+a[cc].hi)/2 for cc in range(5)]; anc=math.sqrt(sum(x*x for x in ac))
            for P in range(len(Ri)):
                if P in simp: continue
                sc=sum(ac[cc]*((Ri[P][cc].lo+Ri[P][cc].hi)/2-(r0[cc].lo+r0[cc].hi)/2) for cc in range(5))
                if anc>0 and sc/anc>-1e-7: continue
                s=F.FI(0.0)
                for cc in range(5): s=s+a[cc]*(Ri[P][cc]-r0[cc])
                if s.hi>1e-12:
                    st=det5_tight([diffs[0],diffs[1],diffs[2],diffs[3],[Ri[P][cc]-r0[cc] for cc in range(5)]])
                    if st.hi>1e-12: unstable.add(active[P])
        return ('ok',dmin,unstable)
    # SUB-HULL loop: drop near-coplanar unstable flags. SOUND for depth — conv(subset)⊆conv F, so
    # inradius(subset) <= d(u2); a positive sub-hull inradius is a valid d_lo lower bound.
    active=list(range(len(Rfl)))
    for _ in range(6):
        res=_depth_on(active)
        if res[0]=='fail': return None,False,res[1]
        _,dmin,unstable=res
        if dmin is None: return None,False,'no-facets'
        if not unstable: return dmin.lo,True,'ok'
        active=[i for i in active if i not in unstable]
        if len(active)<6: return None,False,'subhull-small'
    return None,False,'no-converge'

def candidate_eos(th,thw,ph,phw):
    """distinct edge-orders over the cell (SAMPLING — corners+center+edge-mids). Rigor upgrade TODO."""
    pts=[(th+s*thw, ph+t*phw) for s in (-1,-0.5,0,0.5,1) for t in (-1,-0.5,0,0.5,1)]
    seen={}
    for (tt,pp) in pts:
        eo=BC.edge_order(BC.u2center(tt,pp))
        if eo is None: continue
        key=tuple((p_,r_,sgn) for (p_,r_,sgn) in eo)
        seen[key]=eo
    return list(seen.values())

def dlo_cert_twocase(th,thw,ph,phw):
    """certify d_lo>0 under EVERY candidate combinatorics; d=min. ok only if all candidates pass."""
    cands=candidate_eos(th,thw,ph,phw)
    if not cands: return None,False,'no-cands'
    dmin=9.0
    for eo in cands:
        d,ok,why=dlo_cert_eo(th,thw,ph,phw,eo)
        if not ok: return None,False,f'cand:{why}'
        dmin=min(dmin,d)
    return dmin,True,f'twocase({len(cands)})'

def stable_edges(th,thw,ph,phw,eo):
    """RIGOROUS: the edges of eo whose every non-endpoint vertex is STRICTLY inward over the whole
    cell (interval turn test, unnormalized scale-invariant frame — as in hull_stable). These are
    PROVABLY silhouette edges of the true hull for every u2 in the cell, so their flags form a subset
    of the true active flags: conv{stable flags} ⊆ conv F(u2) ⇒ inradius(subset) ≤ d(u2)."""
    u2=F.iu2(th-thw,th+thw,ph-phw,ph+phw)
    f1=F.cross([F.FI(0.0),F.FI(0.0),F.FI(1.0)],u2); f2=F.cross(u2,f1); V=F.verts(aa)
    qt=[[F.dot(f2,v), F.FI(0.0)-F.dot(f1,v)] for v in V]
    keep=[]
    for (p_,r_,sgn) in eo:
        d0=qt[r_][0]-qt[p_][0]; d1=qt[r_][1]-qt[p_][1]; n0=F.FI(0.0)-d1; n1=d0
        if sgn<0: n0=F.FI(0.0)-n0; n1=F.FI(0.0)-n1
        ok=True
        for k in range(8):
            if k==p_ or k==r_: continue
            if (n0*(qt[k][0]-qt[p_][0])+n1*(qt[k][1]-qt[p_][1])).lo<=0: ok=False; break
        if ok: keep.append((p_,r_,sgn))
    return keep

def dlo_cert_wall(th,thw,ph,phw):
    """Subset-safe fallback for hull-unstable (transition) cells: certify d_lo>0 using only the
    PROVABLY-silhouette (stable) edges. Sound (subset ⊆ conv F ⇒ inradius bound). Returns (d,ok,why)."""
    u2c=BC.u2center(th,ph); eo=BC.edge_order(u2c)
    if eo is None: return None,False,'no-hull'
    se=stable_edges(th,thw,ph,phw,eo)
    if len(se)<3: return None,False,'too-few-stable'
    return dlo_cert_eo(th,thw,ph,phw,se)


# ==================== r79 additions: W2-A/B/C wall module ====================
# Lemmas and soundness: see notes_round_79_in2_lemmas.md.
#  W2-A structured hull case analysis (C1/C2 interval tests -> per-u true hull = coarse cycle
#        with per-edge insertions from S_i);
#  W2-B flag-perturbation kill transfer (d_lo = d_c - eps);
#  W2-C degree-invariance inradius bound (replaces the per-flag facet-stability loop:
#        (K1) exact combinatorial cycle check, (K2) interval hyperplane-distance over the cell,
#        (K3) exact integer ray-cast degree = +1 at the box-midpoint base configuration).
from collections import Counter, defaultdict

def _qt_cell(th,thw,ph,phw):
    u2=F.iu2(th-thw,th+thw,ph-phw,ph+phw)
    f1=F.cross([F.FI(0.0),F.FI(0.0),F.FI(1.0)],u2); f2=F.cross(u2,f1)
    V=F.verts(aa)
    return [[F.dot(f2,v), F.FI(0.0)-F.dot(f1,v)] for v in V]

def _turn(qt,a,v,b):
    return (qt[v][0]-qt[a][0])*(qt[b][1]-qt[v][1]) - (qt[v][1]-qt[a][1])*(qt[b][0]-qt[v][0])

def coarse_build(th,thw,ph,phw):
    """W2-A data: coarse cycle + per-edge ambiguous sets, interval-verified (C1),(C2) over the cell.
    Returns ((edges,S,orient),'ok') or (None,why)."""
    eo=BC.edge_order(BC.u2center(th,ph))
    if eo is None: return None,'no-hull'
    qt=_qt_cell(th,thw,ph,phw)
    cyc=[e[0] for e in eo]
    changed=True
    while changed and len(cyc)>=3:
        changed=False
        for idx in range(len(cyc)):
            c=_turn(qt,cyc[idx-1],cyc[idx],cyc[(idx+1)%len(cyc)])
            if c.lo<=0<=c.hi:
                cyc.pop(idx); changed=True; break
    if len(cyc)<3: return None,'cycle-collapse'
    sgns=set()
    for idx in range(len(cyc)):
        c=_turn(qt,cyc[idx-1],cyc[idx],cyc[(idx+1)%len(cyc)])
        if c.lo<=0<=c.hi: return None,'turn-straddle'
        sgns.add(1 if c.lo>0 else -1)
    if len(sgns)!=1: return None,'mixed-turns'
    orient=sgns.pop()
    edges=[]; S=[]; cnt=Counter()
    for i in range(len(cyc)):
        p_,r_=cyc[i],cyc[(i+1)%len(cyc)]
        d0=qt[r_][0]-qt[p_][0]; d1=qt[r_][1]-qt[p_][1]
        Se=[]
        for k in range(8):
            if k==p_ or k==r_: continue
            s=(F.FI(0.0)-d1)*(qt[k][0]-qt[p_][0])+d0*(qt[k][1]-qt[p_][1])
            if orient<0: s=F.FI(0.0)-s
            if s.lo>0: continue
            Se.append(k); cnt[k]+=1
        edges.append((p_,r_,orient)); S.append(Se)
    if any(v>1 for v in cnt.values()): return None,'multi-straddle'
    return (edges,S,orient),'ok'

_VN2=[3.0]*4+[3.0*(11.0/20.0)**2]*4   # |v_j|^2 exactly (float of 363/400 rounded up below)
def eps_bound(th,thw,ph,phw,edges,S):
    """W2-B eps: interval sup over the cell of the coarse-vs-fine flag pairing distances."""
    W=F.frame(F.iu2(th-thw,th+thw,ph-phw,ph+phw)); V=F.verts(aa)
    Wv=[F.applyW(W,v) for v in V]; q=[F.proj(w) for w in Wv]
    def unitn(p_,r_,sgn):
        d0=q[r_][0]-q[p_][0]; d1=q[r_][1]-q[p_][1]; n0=F.FI(0.0)-d1; n1=d0
        if sgn<0: n0=F.FI(0.0)-n0; n1=F.FI(0.0)-n1
        L=(n0.sqr()+n1.sqr()).sqrt()
        if L.lo<=0: return None
        return n0/L, n1/L
    epsmax=0.0
    for (e,Se) in zip(edges,S):
        if not Se: continue
        p_,r_,sgn=e
        ne=unitn(p_,r_,sgn)
        if ne is None: return None
        for (j,cands) in ((p_,[(p_,c) for c in Se]),(r_,[(c,r_) for c in Se])):
            fac=2.0000000001 if j<4 else 1.3811    # sqrt(|v_j|^2+1) outward-rounded: sqrt(4)=2, sqrt(763)/20<1.3811
            for (x,y) in cands:
                nf=unitn(x,y,sgn)
                if nf is None: return None
                dn0=ne[0]-nf[0]; dn1=ne[1]-nf[1]
                nrm=(dn0.sqr()+dn1.sqr()).sqrt().hi
                epsmax=max(epsmax,fac*nrm)
    return epsmax

def _idet(M):
    """Bareiss fraction-free integer determinant (exact)."""
    M=[row[:] for row in M]; n=len(M); sign=1; prev=1
    for k in range(n-1):
        if M[k][k]==0:
            for i in range(k+1,n):
                if M[i][k]!=0: M[k],M[i]=M[i],M[k]; sign=-sign; break
            else: return 0
        for i in range(k+1,n):
            for j in range(k+1,n):
                M[i][j]=(M[i][j]*M[k][k]-M[i][k]*M[k][j])//prev
        prev=M[k][k]
    return sign*M[-1][-1]

def _cof_normal_i(P):
    """integer normal a with a.y = det[y;P1-P0;...;P4-P0] (so a.(Pi-P0)=0)."""
    E=[[P[t][c]-P[0][c] for c in range(5)] for t in range(1,5)]
    a=[]
    for k in range(5):
        Mk=[[E[t][c] for c in range(5) if c!=k] for t in range(4)]
        d=_idet(Mk)
        a.append(d if k%2==0 else -d)
    return a

def _parity_sort4(t):
    perm=sorted(range(4),key=lambda i:t[i])
    par=1; seen=[False]*4
    for i in range(4):
        if seen[i]: continue
        l=0; j=i
        while not seen[j]: seen[j]=True; j=perm[j]; l+=1
        if l%2==0: par=-par
    return par

def _cycle_check(simps,fs):
    """(K1): boundary of the signed 4-chain vanishes."""
    acc=defaultdict(int)
    for tau,f in zip(simps,fs):
        for k in range(5):
            ridge=tuple(tau[:k])+tuple(tau[k+1:])
            acc[tuple(sorted(ridge))]+=f*((-1)**k)*_parity_sort4(ridge)
    return all(v==0 for v in acc.values())

def _ray_degree(Pint,simps,fs,anrs):
    """(K3): exact integer ray-cast degree of the signed cycle about the origin."""
    import random
    rnd=random.Random(20790411)
    for _ in range(60):
        e=[rnd.randint(-999,999) for _ in range(5)]
        deg=0; ok=True
        for tau,f,a in zip(simps,fs,anrs):
            P=[Pint[i] for i in tau]
            M=[[P[j][r] for j in range(5)]+[-e[r]] for r in range(5)]
            M.append([1,1,1,1,1,0])
            D=_idet(M)
            if D==0: ok=False; break
            sD=1 if D>0 else -1
            hit=True; degen=False
            for col in range(6):
                Mc=[row[:] for row in M]
                for r in range(6): Mc[r][col]=(0 if r<5 else 1)
                v=_idet(Mc)*sD
                if v==0: degen=True; break
                if v<0: hit=False; break
            if degen: ok=False; break
            if not hit: continue
            ae=sum(a[k]*e[k] for k in range(5))
            if ae==0: ok=False; break
            deg+=f*(1 if ae>0 else -1)
        if ok: return deg
    return None

def dlo_cert_eo_deg(th,thw,ph,phw,eo):
    """W2-C: R^5 inradius lower bound with a FIXED edge-order, degree-based (no facet-stability).
    Returns (d,ok,why)."""
    W=F.frame(F.iu2(th-thw,th+thw,ph-phw,ph+phw)); V=F.verts(aa)
    Wv=[F.applyW(W,v) for v in V]; q=[F.proj(w) for w in Wv]; zc=[w[2] for w in Wv]
    Rint=[]
    for (p_,r_,sgn) in eo:
        d0=q[r_][0]-q[p_][0]; d1=q[r_][1]-q[p_][1]
        ni0=F.FI(0.0)-d1; ni1=d0
        if sgn<0: ni0=F.FI(0.0)-ni0; ni1=F.FI(0.0)-ni1
        L=(ni0.sqr()+ni1.sqr()).sqrt()
        if L.lo<=0: return None,False,'edge-degenerate'
        ni0=ni0/L; ni1=ni1/L
        for j in (p_,r_):
            Jq0=F.FI(0.0)-q[j][1]; Jq1=q[j][0]
            Rint.append([zc[j]*ni0, zc[j]*ni1, ni0*Jq0+ni1*Jq1, ni0, ni1])
    m=len(Rint)
    if m<6: return None,False,'few-flags'
    # base configuration p0: dyadic rounding of box midpoints, verified inside boxes
    SC=1<<40
    P0=[]
    for i in range(m):
        row=[]
        for c in range(5):
            lo,hi=Rint[i][c].lo,Rint[i][c].hi
            x=round(((lo+hi)/2)*SC)/SC
            if not (lo<=x<=hi): x=(lo+hi)/2
            row.append(x)
        P0.append(row)
    import numpy as _np
    try: H=ConvexHull(_np.array(P0))
    except Exception: return None,False,'r5-hull'
    # exact integers for the base configuration: ONE global power-of-two scale (floats are dyadic)
    pairs=[[float(x).as_integer_ratio() for x in row] for row in P0]
    Dmax=1
    for row in pairs:
        for (_,dd) in row: Dmax=max(Dmax,dd)
    Pint=[[n*(Dmax//dd) for (n,dd) in row] for row in pairs]
    simps=[tuple(int(v) for v in s) for s in H.simplices]
    fs=[]; anrs=[]
    for tau in simps:
        P=[Pint[i] for i in tau]
        a=_cof_normal_i(P)
        ap0=sum(a[k]*P[0][k] for k in range(5))
        if ap0==0: return None,False,'base-degenerate'
        f=1 if ap0>0 else -1
        fs.append(f); anrs.append(a)
    if not _cycle_check(simps,fs): return None,False,'not-cycle'
    deg=_ray_degree(Pint,simps,fs,anrs)
    if deg!=1: return None,False,f'degree={deg}'
    # (K2): interval hyperplane distances over the whole cell
    cen=[sum((Rint[i][c].lo+Rint[i][c].hi)/2 for i in range(m))/m for c in range(5)]
    dmin=None
    for tau in simps:
        r0=Rint[tau[0]]
        diffs=[[Rint[tau[t]][c]-r0[c] for c in range(5)] for t in range(1,5)]
        a=[]
        for k in range(5):
            Mk=[[diffs[t][c] for c in range(5) if c!=k] for t in range(4)]
            dk=DG._det4(Mk); a.append(dk if k%2==0 else F.FI(0.0)-dk)
        c=F.FI(0.0)
        for cc in range(5): c=c+a[cc]*r0[cc]
        af_c=sum(((a[cc].lo+a[cc].hi)/2)*(cen[cc]-(r0[cc].lo+r0[cc].hi)/2) for cc in range(5))
        if af_c>0: a=[F.FI(0.0)-x for x in a]; c=F.FI(0.0)-c
        anorm=(a[0].sqr()+a[1].sqr()+a[2].sqr()+a[3].sqr()+a[4].sqr()).sqrt()
        if anorm.lo<=0: return None,False,'facet-degenerate'
        if c.lo<=0: return None,False,'origin-outside'
        dfac=c/anorm
        dmin=dfac.lo if dmin is None else min(dmin,dfac.lo)
    return dmin,True,'ok'

def dlo_cert_wall2(th,thw,ph,phw):
    """Full W2 wall-cell certificate: coarse cycle (W2-A) + degree cert (W2-C) + eps transfer (W2-B).
    Returns (d_lo, ok, why); sound: d(u2) >= d_lo > 0 for all u2 in the cell."""
    cb,why=coarse_build(th,thw,ph,phw)
    if cb is None: return None,False,'A:'+why
    edges,S,orient=cb
    d,ok,w2=dlo_cert_eo_deg(th,thw,ph,phw,edges)
    if not ok: return None,False,'C:'+w2
    eps=eps_bound(th,thw,ph,phw,edges,S)
    if eps is None: return None,False,'B:eps-degenerate'
    if d-eps<=0: return None,False,f'B:eps-eats(d={d:.4g},eps={eps:.4g})'
    return d-eps,True,'wall2'


# ==================== r80 additions: minimal-coarse (W2-A') + witness-K2 (W2-C') ====================
from itertools import combinations

def _check_cycle(qt,cyc):
    """interval (C2)+(C1) for a candidate cycle; returns (edges,S,orient) or None."""
    sg=set()
    for idx in range(len(cyc)):
        c=_turn(qt,cyc[idx-1],cyc[idx],cyc[(idx+1)%len(cyc)])
        if c.lo<=0<=c.hi: return None
        sg.add(1 if c.lo>0 else -1)
    if len(sg)!=1: return None
    orient=sg.pop()
    edges=[]; S=[]; cnt=Counter()
    for i in range(len(cyc)):
        p_,r_=cyc[i],cyc[(i+1)%len(cyc)]
        d0=qt[r_][0]-qt[p_][0]; d1=qt[r_][1]-qt[p_][1]
        Se=[]
        for k in range(8):
            if k==p_ or k==r_: continue
            s=(F.FI(0.0)-d1)*(qt[k][0]-qt[p_][0])+d0*(qt[k][1]-qt[p_][1])
            if orient<0: s=F.FI(0.0)-s
            if s.lo>0: continue
            Se.append(k); cnt[k]+=1
        edges.append((p_,r_,orient)); S.append(Se)
    if any(v>1 for v in cnt.values()): return None    # multi-straddle -> refuse
    return (edges,S,orient)

def coarse_build_min(th,thw,ph,phw,maxrm=5):
    """W2-A with MINIMAL removal: smallest subset T of the midpoint-hull cycle whose removal
    passes interval (C1)+(C2) over the cell. Sound for any passing cycle; minimality only
    maximizes the coarse hull (tightest d_lo)."""
    eo=BC.edge_order(BC.u2center(th,ph))
    if eo is None: return None,'no-hull'
    qt=_qt_cell(th,thw,ph,phw)
    cyc0=[e[0] for e in eo]; n=len(cyc0)
    for k in range(0,min(maxrm,n-3)+1):
        for T in combinations(range(n),k):
            cyc=[cyc0[i] for i in range(n) if i not in T]
            r=_check_cycle(qt,cyc)
            if r is not None: return r,'ok(rm=%d)'%k
    return None,'no-valid-cycle'

def dlo_cert_eo_deg2(th,thw,ph,phw,eo,tries=4):
    """W2-C with witness-vector K2: per-simplex fixed exact w (qhull facet equation) with
    interval min_i |w.r_i| >= c > 0 over the whole cell => moving simplex avoids B(0,c/|w|).
    (K1) exact cycle check and (K3) exact integer ray-cast degree unchanged. Side-free, no
    interval determinants, degenerate simplices harmless. Returns (d,ok,why)."""
    import numpy as _np, random as _rd
    W=F.frame(F.iu2(th-thw,th+thw,ph-phw,ph+phw)); V=F.verts(aa)
    Wv=[F.applyW(W,v) for v in V]; q=[F.proj(w) for w in Wv]; zc=[w[2] for w in Wv]
    Rint=[]
    for (p_,r_,sgn) in eo:
        d0=q[r_][0]-q[p_][0]; d1=q[r_][1]-q[p_][1]
        ni0=F.FI(0.0)-d1; ni1=d0
        if sgn<0: ni0=F.FI(0.0)-ni0; ni1=F.FI(0.0)-ni1
        L=(ni0.sqr()+ni1.sqr()).sqrt()
        if L.lo<=0: return None,False,'edge-degenerate'
        ni0=ni0/L; ni1=ni1/L
        for j in (p_,r_):
            Jq0=F.FI(0.0)-q[j][1]; Jq1=q[j][0]
            Rint.append([zc[j]*ni0, zc[j]*ni1, ni0*Jq0+ni1*Jq1, ni0, ni1])
    m=len(Rint)
    if m<6: return None,False,'few-flags'
    rnd=_rd.Random(808080)
    SC=1<<40
    lastwhy='?'
    for att in range(tries):
        # base configuration: dyadic point inside each box (midpoint, then random perturbations)
        P0=[]
        for i in range(m):
            row=[]
            for c in range(5):
                lo,hi=Rint[i][c].lo,Rint[i][c].hi
                t=0.5 if att==0 else 0.5+0.35*(2*rnd.random()-1)
                x=round((lo+(hi-lo)*t)*SC)/SC
                if not (lo<=x<=hi): x=(lo+hi)/2
                row.append(x)
            P0.append(row)
        try: H=ConvexHull(_np.array(P0))
        except Exception: lastwhy='r5-hull'; continue
        pairs=[[float(x).as_integer_ratio() for x in row] for row in P0]
        Dmax=1
        for row in pairs:
            for (_,dd) in row: Dmax=max(Dmax,dd)
        Pint=[[nu*(Dmax//dd) for (nu,dd) in row] for row in pairs]
        simps=[tuple(int(v) for v in s) for s in H.simplices]
        fs=[]; anrs=[]; bad=False
        for tau in simps:
            P=[Pint[i] for i in tau]
            a=_cof_normal_i(P)
            ap0=sum(a[k]*P[0][k] for k in range(5))
            if ap0==0: bad=True; lastwhy='base-degenerate'; break
            fs.append(1 if ap0>0 else -1); anrs.append(a)
        if bad: continue
        if not _cycle_check(simps,fs): lastwhy='not-cycle'; continue
        deg=_ray_degree(Pint,simps,fs,anrs)
        if deg!=1: lastwhy='degree=%s'%deg; continue
        # K2': witness half-space per simplex over the WHOLE cell
        dmin=None; ok=True
        for ti,tau in enumerate(simps):
            nvec=[float(x) for x in H.equations[ti][:5]]
            nFI=[F.FI(x) for x in nvec]
            los=[]; his=[]
            for i in tau:
                s=F.FI(0.0)
                for cc in range(5): s=s+nFI[cc]*Rint[i][cc]
                los.append(s.lo); his.append(s.hi)
            c=max(min(los), -max(his))     # side-free: all >= c or all <= -c
            if c<=0: ok=False; lastwhy='plane-near-origin'; break
            nn=(nFI[0].sqr()+nFI[1].sqr()+nFI[2].sqr()+nFI[3].sqr()+nFI[4].sqr()).sqrt()
            if nn.hi<=0: ok=False; lastwhy='w-degenerate'; break
            dfac=(F.FI(c)/nn).lo
            dmin=dfac if dmin is None else min(dmin,dfac)
        if not ok: continue
        if dmin is None or dmin<=0: lastwhy='no-positive-d'; continue
        return dmin,True,'ok'
    return None,False,lastwhy

def dlo_cert_wall2b(th,thw,ph,phw):
    """r80 wall-cell certificate: minimal coarse cycle (W2-A') + witness-K2 degree cert (W2-C')
    + eps transfer (W2-B). Returns (d_lo, ok, why)."""
    cb,why=coarse_build_min(th,thw,ph,phw)
    if cb is None: return None,False,'A:'+why
    edges,S,orient=cb
    d,ok,w2=dlo_cert_eo_deg2(th,thw,ph,phw,edges)
    if not ok: return None,False,'C:'+w2
    eps=eps_bound(th,thw,ph,phw,edges,S)
    if eps is None: return None,False,'B:eps-degenerate'
    if d-eps<=0: return None,False,'B:eps-eats(d=%.4g,eps=%.4g)'%(d,eps)
    return d-eps,True,'wall2b'


# ==================== r80b: tight side tests + config enumeration (W2-D) ====================
def _cx(a0,a1,b0,b1): return a0*b1-a1*b0

def _side3(qt,p_,r_,k):
    """interval of cross(r-p, k-p): intersection of three algebraically equal forms
    (kills dependency slop; sound since all equal the same real number)."""
    A=_cx(qt[r_][0]-qt[p_][0],qt[r_][1]-qt[p_][1], qt[k][0]-qt[p_][0],qt[k][1]-qt[p_][1])
    B=_cx(qt[r_][0]-qt[p_][0],qt[r_][1]-qt[p_][1], qt[k][0]-qt[r_][0],qt[k][1]-qt[r_][1])
    C=_cx(qt[r_][0]-qt[k][0],qt[r_][1]-qt[k][1], qt[p_][0]-qt[k][0],qt[p_][1]-qt[k][1])
    lo=max(A.lo,B.lo,-C.hi); hi=min(A.hi,B.hi,-C.lo)
    return FI(lo,hi)   # nonempty since the true value lies in all three

def _check_cycle2(qt,cyc):
    """(C2)+(C1) with tight _side3 for both turns and side tests."""
    sg=set()
    for idx in range(len(cyc)):
        c=_side3(qt,cyc[idx-1],cyc[idx],cyc[(idx+1)%len(cyc)])
        if c.lo<=0<=c.hi: return None
        sg.add(1 if c.lo>0 else -1)
    if len(sg)!=1: return None
    orient=sg.pop()
    edges=[]; S=[]; cnt=Counter()
    for i in range(len(cyc)):
        p_,r_=cyc[i],cyc[(i+1)%len(cyc)]
        Se=[]
        for k in range(8):
            if k==p_ or k==r_: continue
            s=_side3(qt,p_,r_,k)
            if orient<0: s=FI(-s.hi,-s.lo)
            if s.lo>0: continue
            Se.append(k); cnt[k]+=1
        edges.append((p_,r_,orient)); S.append(Se)
    if any(v>1 for v in cnt.values()): return None
    # a cycle vertex must not be ambiguous for a non-incident edge (breaks the case analysis)
    for v in cnt:
        if v in cyc: return None
    return (edges,S,orient)

def coarse_build_min2(th,thw,ph,phw,maxrm=5):
    eo=BC.edge_order(BC.u2center(th,ph))
    if eo is None: return None,'no-hull'
    qt=_qt_cell(th,thw,ph,phw)
    cyc0=[int(e[0]) for e in eo]; n=len(cyc0)
    for k in range(0,min(maxrm,n-3)+1):
        for T in combinations(range(n),k):
            cyc=[cyc0[i] for i in range(n) if i not in T]
            r=_check_cycle2(qt,cyc)
            if r is not None: return r,'ok(rm=%d)'%k
    return None,'no-valid-cycle'

def enum_configs(th,thw,ph,phw,edges,S,cap=12):
    """W2-D: enumerate edge-orders such that for EVERY u in the cell the true active flag set
    contains the flags of at least one enumerated config. Per edge with S_i={x1..xm}: every
    subset of S_i, inserted in certified along-edge order (interval-strict; refuse if ambiguous).
    Returns list of eos or None."""
    qt=_qt_cell(th,thw,ph,phw)
    from itertools import product
    per_edge=[]
    for (e,Se) in zip(edges,S):
        p_,r_,orient=e
        opts=[[]]
        if Se:
            if len(Se)>1:
                # certified along-edge strict order of Se (dot with edge direction)
                d0=qt[r_][0]-qt[p_][0]; d1=qt[r_][1]-qt[p_][1]
                ts={}
                for x in Se: ts[x]=d0*(qt[x][0]-qt[p_][0])+d1*(qt[x][1]-qt[p_][1])
                srt=sorted(Se,key=lambda x:(ts[x].lo+ts[x].hi)/2)
                for a,b in zip(srt,srt[1:]):
                    if not (ts[b].lo>ts[a].hi): return None   # order ambiguous -> refuse
                Se=srt
            opts=[]
            for mask in range(1<<len(Se)):
                sub=[Se[t] for t in range(len(Se)) if mask>>t&1]
                opts.append(sub)
        per_edge.append((p_,r_,orient,opts))
    total=1
    for (_,_,_,opts) in per_edge: total*=len(opts)
    if total>cap: return None
    cfgs=[]
    for choice in product(*[opts for (_,_,_,opts) in per_edge]):
        eo=[]
        for ((p_,r_,orient,_),sub) in zip(per_edge,choice):
            chain=[p_]+list(sub)+[r_]
            for a,b in zip(chain,chain[1:]): eo.append((a,b,orient))
        cfgs.append(eo)
    return cfgs

def dlo_cert_wall2c(th,thw,ph,phw):
    """r80 final wall certificate: minimal coarse cycle (tight tests) + W2-D config enumeration,
    each config certified by the witness-K2 degree cert; d_lo = min over configs. Fallback: the
    eps-transfer path. Returns (d_lo, ok, why)."""
    cb,why=coarse_build_min2(th,thw,ph,phw)
    if cb is None: return None,False,'A:'+why
    edges,S,orient=cb
    cfgs=enum_configs(th,thw,ph,phw,edges,S)
    if cfgs is not None:
        dmin=None; allok=True; whys=[]
        for eo in cfgs:
            d,ok,w2=dlo_cert_eo_deg2(th,thw,ph,phw,eo)
            if not ok: allok=False; whys.append(w2); break
            dmin=d if dmin is None else min(dmin,d)
        if allok and dmin is not None and dmin>0:
            return dmin,True,'wall2c(%d)'%len(cfgs)
    # fallback: eps transfer
    d,ok,w2=dlo_cert_eo_deg2(th,thw,ph,phw,edges)
    if not ok: return None,False,'C:'+w2
    eps=eps_bound(th,thw,ph,phw,edges,S)
    if eps is None: return None,False,'B:eps-degenerate'
    if d-eps<=0: return None,False,'B:eps-eats(d=%.4g,eps=%.4g)'%(d,eps)
    return d-eps,True,'wall2b'

def dlo_cert_wall2r(th,thw,ph,phw,depth=0,maxdepth=5):
    """wall2c with recursive subdivision (for near-critical-annulus cells where interval slop
    at the given width exceeds the genuinely small depth). Sound: subcells tile the cell."""
    d,ok,why=dlo_cert_wall2c(th,thw,ph,phw)
    if ok or depth>=maxdepth: return d,ok,why,depth
    dmin=None
    for sa in (-0.5,0.5):
        for sb in (-0.5,0.5):
            dd,ok2,w2,dp=dlo_cert_wall2r(th+sa*thw,thw/2,ph+sb*phw,phw/2,depth+1,maxdepth)
            if not ok2: return dd,False,w2,dp
            dmin=dd if dmin is None else min(dmin,dd)
    return dmin,True,'subdiv',depth


# ==================== r80c: SOUND config-activeness (Lemmas SUP + DICH) ====================
def _check_cycle3(qt,cyc):
    """(ii) base-edge support tests + S assignment + (D2a) inserted-edge support tests.
    Soundness: Lemma SUP (support line => active flags) + Lemma DICH (sign dichotomy with the
    exact identities cross(x-ci, c_{i+1}-ci) = -s, cross(c_{i+1}-x, ci-x) = -s). Requires
    |S_i| <= 1 per edge; each candidate in at most one edge; no cycle vertex ambiguous."""
    sg=set()
    for idx in range(len(cyc)):
        c=_side3(qt,cyc[idx-1],cyc[idx],cyc[(idx+1)%len(cyc)])
        if c.lo<=0<=c.hi: return None
        sg.add(1 if c.lo>0 else -1)
    if len(sg)!=1: return None
    orient=sg.pop()
    def sided(p_,r_,k):
        s=_side3(qt,p_,r_,k)
        return FI(-s.hi,-s.lo) if orient<0 else s
    edges=[]; S=[]; cnt=Counter()
    for i in range(len(cyc)):
        p_,r_=cyc[i],cyc[(i+1)%len(cyc)]
        Se=[]
        for k in range(8):
            if k==p_ or k==r_: continue
            if sided(p_,r_,k).lo>0: continue
            Se.append(k); cnt[k]+=1
        if len(Se)>1: return None                  # multi-candidate edge: refuse (subdivide)
        edges.append((p_,r_,orient)); S.append(Se)
    if any(v>1 for v in cnt.values()): return None # candidate ambiguous for two edges
    for v in cnt:
        if v in cyc: return None                   # cycle vertex ambiguous: refuse
    # (D2a): for each edge with S_i={x}, the two inserted edges must support strictly
    # against every l except the exact-identity partners
    for (e,Se) in zip(edges,S):
        if not Se: continue
        p_,r_,_=e; x=Se[0]
        for (a,b,skip) in ((p_,x,r_),(x,r_,p_)):
            for l in range(8):
                if l in (a,b,skip): continue
                if sided(a,b,l).lo<=0: return None
    return (edges,S,orient)

def coarse_build_min3(th,thw,ph,phw,maxrm=5):
    eo=BC.edge_order(BC.u2center(th,ph))
    if eo is None: return None,'no-hull'
    qt=_qt_cell(th,thw,ph,phw)
    cyc0=[int(e[0]) for e in eo]; n=len(cyc0)
    for k in range(0,min(maxrm,n-3)+1):
        for T in combinations(range(n),k):
            cyc=[cyc0[i] for i in range(n) if i not in T]
            r=_check_cycle3(qt,cyc)
            if r is not None: return r,'ok(rm=%d)'%k
    return None,'no-valid-cycle'

def dlo_cert_wall2d(th,thw,ph,phw):
    """PROOF-GRADE wall certificate: SUP/DICH-verified configs (each |S_i|<=1, branch
    dichotomy exact) + witness-K2 degree certificate per config; d_lo = min over configs.
    Fallback: eps transfer (sound under the same DICH pairing). Returns (d_lo, ok, why)."""
    cb,why=coarse_build_min3(th,thw,ph,phw)
    if cb is None: return None,False,'A:'+why
    edges,S,orient=cb
    cfgs=enum_configs(th,thw,ph,phw,edges,S,cap=16)
    if cfgs is not None:
        dmin=None; allok=True
        for eo in cfgs:
            d,ok,w2=dlo_cert_eo_deg2(th,thw,ph,phw,eo)
            if not ok: allok=False; break
            dmin=d if dmin is None else min(dmin,d)
        if allok and dmin is not None and dmin>0:
            return dmin,True,'wall2d(%d)'%len(cfgs)
    d,ok,w2=dlo_cert_eo_deg2(th,thw,ph,phw,edges)
    if not ok: return None,False,'C:'+w2
    eps=eps_bound(th,thw,ph,phw,edges,S)
    if eps is None: return None,False,'B:eps-degenerate'
    if d-eps<=0: return None,False,'B:eps-eats(d=%.4g,eps=%.4g)'%(d,eps)
    return d-eps,True,'wall2b'
# ==================== r84: W5 collision-cap certificate (codim-2 stellation-edge points) ====
# Closes cells near a stellation-edge collision direction (T_d-orbit of u0=(v1-p2)/|v1-p2| =
# (31,9,9)/sqrt(1123); two copies meet FBOX: (1.298891,0.282555) and (1.842701,0.282555)),
# where two vertices A,B project to the SAME silhouette corner and every constant-
# combinatorics certificate fails at every scale (combinatorial singularity; depth healthy).
#
# Lemma W5-COV (soundness). Fix a parameter cell C. Suppose the following STRICT outward-
# rounded interval tests hold over C, with a common orientation sign 'orient':
#  (T1) for every BASE edge (p,r) (consecutive pairs of the stable chain S, gap excluded):
#       side(p,r,k) strictly inward for ALL k not in {p,r} (including A, B);
#  (T2') for each corner line in {(P,A),(A,Nx)}: side strictly inward for all k not in
#       {P,A,Nx,B}; for each of {(P,B),(B,Nx)}: same with A,B swapped;
#  (T4) A and B strictly OUTSIDE the chord (P,Nx)  (P,Nx = the gap neighbors).
# Then for every u in C the shadow-hull cycle is [S-chain, corner-seq] with corner-seq in
# {(A),(B),(A,B),(B,A)}, and the true metric-active flag set at u CONTAINS the used flag
# set of the matching config:
#   C1 = base+[(P,A),(A,Nx)], C2 = base+[(P,B),(B,Nx)],
#   C3 = base+[(P,A),(B,Nx)], C4 = base+[(P,B),(A,Nx)]   (short edge (A,B) NEVER used).
# Proof: (T1) makes every base edge a strict support edge, so the hull cycle contains the
# S-chain in order and extra hull vertices can occur only in the gap. (T1)+(T2') make all
# edges of conv(S+{A}) (resp. conv(S+{B})) strict support lines against every vertex except
# B (resp. A); hence every k outside {S,A,B} lies in int conv(S+{A}) subset int Hull(u), so
# the only possible gap hull vertices are A,B. (T4) forces A outside conv(S), so the gap
# seq is nonempty. Case check: seq=(A): hull edges are base+(P,A),(A,Nx) >= C1's used edges
# (if B lies on one of them, metric activity only ADDS flags); seq=(B): C2; seq=(A,B):
# hull edges >= {(P,A),(B,Nx)} = C3's corner edges; seq=(B,A): C4. In each case the config's
# 2(nbase+2) flags are all truly active, so by subset-safety (Remark 4.2)
# d(u) >= r_config(u) >= min over the 4 configs of the certified K2' radius.  QED
# All tested lines have endpoint separation O(1) over the cap (only the degenerate short
# edge (A,B) is excluded everywhere), so the interval tests are well-conditioned.

def w5_collision_cert(th,thw,ph,phw,tol=2e-3):
    """W5 collision-cap certificate. Generic over the u0 orbit: colliding pair auto-detected
    at the cell center (detection is heuristic; ALL soundness rests on the interval tests).
    Returns (d_lo, ok, why)."""
    u2c=BC.u2center(th,ph)
    eo0=BC.edge_order(u2c)
    if eo0 is None: return None,False,'w5:no-hull'
    cyc0=[int(e[0]) for e in eo0]
    qt=_qt_cell(th,thw,ph,phw)
    qc=[((qt[k][0].lo+qt[k][0].hi)/2,(qt[k][1].lo+qt[k][1].hi)/2) for k in range(8)]
    pair=None
    for A_ in cyc0:
        for B_ in range(8):
            if B_==A_: continue
            if math.hypot(qc[A_][0]-qc[B_][0],qc[A_][1]-qc[B_][1])<tol: pair=(A_,B_); break
        if pair: break
    if pair is None: return None,False,'w5:no-collision'
    A,B=pair
    S=[v for v in cyc0 if v not in (A,B)]
    n=len(S)
    if n<3: return None,False,'w5:small-S'
    idx0={v:i for i,v in enumerate(cyc0)}
    gaps=[i for i in range(n) if (idx0[S[(i+1)%n]]-idx0[S[i]])%len(cyc0)!=1]
    # if B was interior (not in cyc0) removing A alone must leave exactly one gap
    if len(gaps)!=1: return None,False,'w5:gap!=1'
    gi=gaps[0]; P,Nx=S[gi],S[(gi+1)%n]
    base=[(S[i],S[(i+1)%n]) for i in range(n) if i!=gi]
    # orientation from strict base side tests (T1)
    sg=set()
    for (p_,r_) in base:
        for k in range(8):
            if k in (p_,r_): continue
            s=_side3(qt,p_,r_,k)
            if s.lo<=0<=s.hi: return None,False,'w5:T1(%d,%d;%d)'%(p_,r_,k)
            sg.add(1 if s.lo>0 else -1)
    if len(sg)!=1: return None,False,'w5:T1-orient'
    orient=sg.pop()
    def sided(p_,r_,k):
        s=_side3(qt,p_,r_,k)
        return FI(-s.hi,-s.lo) if orient<0 else s
    # (T2') corner support lines strict against every non-partner vertex
    for (p_,r_,skip) in ((P,A,B),(A,Nx,B),(P,B,A),(B,Nx,A)):
        for k in range(8):
            if k in (p_,r_,skip): continue
            if sided(p_,r_,k).lo<=0: return None,False,'w5:T2(%d,%d;%d)'%(p_,r_,k)
    # (T4) A,B strictly outside the chord (P,Nx)
    for k in (A,B):
        if sided(P,Nx,k).hi>=0: return None,False,'w5:T4(%d)'%k
    # the four corner configs; per-edge inward sign via strict test against interior ref
    interior=[k for k in range(8) if k not in S and k not in (A,B)]
    if not interior: return None,False,'w5:no-ref'
    ref=interior[0]
    cfgs=[base+[(P,A),(A,Nx)], base+[(P,B),(B,Nx)], base+[(P,A),(B,Nx)], base+[(P,B),(A,Nx)]]
    dmin=None
    for ci,edges_ in enumerate(cfgs):
        eo=[]
        for (p_,r_) in edges_:
            s=_side3(qt,p_,r_,ref)
            if s.lo<=0<=s.hi: return None,False,'w5:sgn(%d,%d)'%(p_,r_)
            eo.append((p_,r_,1 if s.lo>0 else -1))
        d,ok,why=dlo_cert_eo_deg2(th,thw,ph,phw,eo)
        if not ok: return None,False,'w5:C%d:%s'%(ci+1,why)
        dmin=d if dmin is None else min(dmin,d)
    return dmin,True,'w5(4)'
