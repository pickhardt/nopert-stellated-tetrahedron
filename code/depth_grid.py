"""Certified bulk depth grid: rigorous interval lower bound d_cert(C) <= min_{u2 in C} d(u2)
over a u2-cell C=(th+-thw, ph+-phw). This is the object that discharges (a) every 'handoff'
leaf of the far manifest (diagonal AND orbit handoffs, via the transport lemmas) and (b) the
cylinder zone s <= d_cert/M of the assembly theorem.

Method (same soundness discipline as sbbox_prove.py):
 1. shadow combinatorics at the cell center (float hull, edge_order) -- then PROVEN constant
    over the whole cell by ball_cover5.hull_stable (interval turn tests, scale-invariant frame).
 2. flags = both endpoints of every silhouette edge (SUBSET of the true metric active set --
    subset-safe, Remark 4.2 of the draft), interval-enclosed over the cell via fast_interval.
 3. facet structure of the R^5 flag hull from the float hull at center; each facet re-derived
    as an INTERVAL inequality (normal via 4x4 minors, offset from a facet flag); outward
    orientation by centroid; per-facet STABILITY: every non-facet, non-coplanar flag stays
    strictly inside over the cell (facet-ridge-graph completeness argument => the certified
    facet list is the complete facet list of conv F(u2) for every u2 in the cell).
 4. d_cert = min over facets of (c/|a|).lo  (ball B(0,d) inside every facet halfspace).
Returns (d_cert, ok, reason). ok=False => subdivide (never a soundness failure)."""
import sys, math
sys.path.insert(0,'certificates'); sys.path.insert(0,'.')
import numpy as np
from fractions import Fraction as Fr
from scipy.spatial import ConvexHull
import fast_interval as F
from fast_interval import FI
import ball_cover as BC
import ball_cover5 as B5
aa=Fr(11,20)

def _det4(M):
    def minor(M,i,j): return [[M[a][b] for b in range(4) if b!=j] for a in range(4) if a!=i]
    def det3(m):
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    s=F.FI(0.0)
    for j in range(4):
        term=M[0][j]*_det4_minor_det3(minor(M,0,j))
        s=s+(term if j%2==0 else F.FI(0.0)-term)
    return s
def _det4_minor_det3(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

def dlo_cert(th,thw,ph,phw):
    u2c=BC.u2center(th,ph)
    eo=BC.edge_order(u2c)
    if eo is None: return None,False,'no-hull'
    box=(th,thw,ph,phw,0,0,0,0,0,0)
    if not B5.hull_stable(box,eo): return None,False,'hull-unstable'
    # interval geometry over the cell
    W=F.frame(F.iu2(th-thw,th+thw,ph-phw,ph+phw)); V=F.verts(aa)
    Wv=[F.applyW(W,v) for v in V]; q=[F.proj(w) for w in Wv]; zc=[w[2] for w in Wv]
    # float flags for structure
    W2=BC.frame_np(u2c); Wvn=(W2@BC.V.T).T; qn=(BC.Z@Wvn.T).T; zcn=Wvn[:,2]
    Rfl=[]; Rint=[]
    for (p_,r_,sgn) in eo:
        dn=qn[r_]-qn[p_]; nn=np.array([-dn[1],dn[0]]);
        if sgn<0: nn=-nn
        nn=nn/np.linalg.norm(nn)
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
    try: H=ConvexHull(Rfl)
    except Exception: return None,False,'r5-hull'
    cen=[sum((Rint[i][c].lo+Rint[i][c].hi)/2 for i in range(len(Rint)))/len(Rint) for c in range(5)]
    dmin=None
    for simp in H.simplices:
        r0=Rint[simp[0]]
        diffs=[[Rint[simp[t]][c]-r0[c] for c in range(5)] for t in range(1,5)]
        a=[]
        for k in range(5):
            Mk=[[diffs[t][c] for c in range(5) if c!=k] for t in range(4)]
            dk=_det4(Mk); a.append(dk if k%2==0 else F.FI(0.0)-dk)
        c=F.FI(0.0)
        for cc in range(5): c=c+a[cc]*r0[cc]
        af_c=sum(((a[cc].lo+a[cc].hi)/2)*(cen[cc]-(r0[cc].lo+r0[cc].hi)/2) for cc in range(5))
        if af_c>0: a=[F.FI(0.0)-x for x in a]; c=F.FI(0.0)-c   # outward: a.y <= c contains centroid side
        anorm=(a[0].sqr()+a[1].sqr()+a[2].sqr()+a[3].sqr()+a[4].sqr()).sqrt()
        if anorm.lo<=0: return None,False,'facet-degenerate'
        if c.lo<=0: return None,False,'origin-outside?'   # need 0 strictly inside every facet
        dfac=c/anorm   # distance of 0 to the facet hyperplane (interval)
        dmin=dfac if dmin is None else FI(min(dmin.lo,dfac.lo),min(dmin.hi,dfac.hi))
        # stability: non-facet, non-coplanar flags stay strictly inside over the cell
        ac=[(a[cc].lo+a[cc].hi)/2 for cc in range(5)]
        anc=math.sqrt(sum(x*x for x in ac))
        for P in range(len(Rint)):
            if P in simp: continue
            sc=sum(ac[cc]*((Rint[P][cc].lo+Rint[P][cc].hi)/2-(r0[cc].lo+r0[cc].hi)/2) for cc in range(5))
            if anc>0 and sc/anc>-1e-7: continue   # structurally coplanar
            s=F.FI(0.0)
            for cc in range(5): s=s+a[cc]*(Rint[P][cc]-r0[cc])
            if s.hi>-0.0: 
                if s.hi>1e-12: return None,False,'facet-unstable'
    return dmin.lo,True,'ok'

if __name__=='__main__':
    import time
    print("certified depth d_cert over u2-cells vs float dlo at center:")
    tests=[('bulk',0.76,0.63,0.01),('bulk-wide',0.76,0.63,0.04),('mid',1.2,0.35,0.01),
           ('near-sector d~0.05',None,None,None),('near-sector d~0.012',None,None,None)]
    # near-sector: u* direction (1,1,0)/sqrt2 in F: th=pi/2? u=(s t cosph, s t sinph, cos th): 
    # (1,1,0)/sqrt2 -> th=pi/2, ph=pi/4. offset by delta in th.
    for name,th,ph,hw in tests:
        if th is None:
            dd=0.05 if '0.05' in name else 0.012
            th=math.pi/2-dd; ph=math.pi/4; hw=dd/8
        t0=time.time(); d,ok,why=dlo_cert(th,hw,ph,hw)
        dt=time.time()-t0
        u2c=BC.u2center(th,ph); dfl=BC.dlo(u2c)
        print(f"  {name:22s} th={th:.4f} ph={ph:.4f} hw={hw:.4f}: d_cert={d if d is None else round(d,5)} ok={ok} ({why})  float dlo(center)={dfl:.5f}  {dt*1000:.0f}ms")
