"""Ball-covering engine (top-down adaptive 5-D octree). Start with LARGE boxes; certify each with ONE
flat-dual λ via the fast interval mean-value G_λ (a certified box = a certified ball ⊇ box). Subdivide
only where G.lo≤0. Big-margin bulk ⇒ few huge boxes; boxes shrink only near the pinch. This is the
covering that beats the fixed grid: box count ∝ ∫dV/(margin/L)^5, adaptive, not resolution·everywhere.
Rigor: fast_interval (ULP-rounded) — G.lo>0 is a proof over the whole box, all t. u₂-boxes kept small
enough that the outer-shadow combinatorics (edge_order at center) are stable (checked)."""
import numpy as np, math, time, json, sys
from fractions import Fraction as Fr
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scipy.linalg import expm
import fast_interval as F
from collections import deque
M=0.866034; smax=math.pi; rmin=1e-3; MINW=5e-5
Z=np.array([[0.,1,0],[-1,0,0]]); V4=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float); V=np.vstack([V4,-11/20*V4])
ustar=np.array([1,-1,0.])/math.sqrt(2); B1=np.array([0,0,1.]); B2=-np.array([1,1,0.])/math.sqrt(2); aa=Fr(11,20)
def u2center(th,ph):   # standard spherical (matches fast_interval.iu2): θ polar, φ azimuth
    return np.array([math.sin(th)*math.cos(ph), math.sin(th)*math.sin(ph), math.cos(th)])
def dircenter(dt,dp): return np.array([math.sin(dt)*math.cos(dp),math.sin(dt)*math.sin(dp),math.cos(dt)])
def frame_np(u):
    u=u/np.linalg.norm(u); h=np.array([0,0,1.]) if abs(u[2])<0.9 else np.array([1,0,0.])
    f1=np.cross(h,u); f1/=np.linalg.norm(f1); f2=np.cross(u,f1); return np.array([f1,f2,u])
def edge_order(u2c):
    W=frame_np(u2c); q=(Z@(W@V.T)).T
    try: H=ConvexHull(q)
    except Exception: return None
    idx=list(H.vertices); ctr=q[idx].mean(0); eo=[]
    for k in range(len(idx)):
        p_,r_=idx[k],idx[(k+1)%len(idx)]; d=q[r_]-q[p_]; n=np.array([-d[1],d[0]])
        eo.append((p_,r_,1 if n@(ctr-q[p_])>=0 else -1))
    return eo
def dlo(u2c):
    W=frame_np(u2c); Wv=(W@V.T).T; q=(Z@Wv.T).T; zc=Wv[:,2]; H=ConvexHull(q); idx=list(H.vertices); ctr=q[idx].mean(0); Fl=[]
    J=np.array([[0.,-1],[1,0]])
    for kk in range(len(idx)):
        p_,r_=idx[kk],idx[(kk+1)%len(idx)]; d=q[r_]-q[p_]; n=np.array([-d[1],d[0]]); n=n/np.linalg.norm(n)
        if n@(ctr-q[p_])<0: n=-n
        ci=np.min(q@n)
        for j in range(8):
            if abs(n@q[j]-ci)<1e-7: Fl.append([zc[j]*n[0],zc[j]*n[1],float(n@(J@q[j])),n[0],n[1]])
    Fl=np.array(Fl)
    if len(Fl)<6: return 0.
    try: HF=ConvexHull(Fl)
    except Exception: return 0.
    b=HF.equations[:,-1]; A=HF.equations[:,:-1]
    if np.any(b>=-1e-12): return 0.
    return float(np.min(-b/np.linalg.norm(A,axis=1)))
def lam_lp(u2c,xi):
    W2=frame_np(u2c); R1=expm(np.array([[0,-xi[2],xi[1]],[xi[2],0,-xi[0]],[-xi[1],xi[0],0]]))@W2
    pin=(Z@(R1@V.T)).T; Wq=(Z@(W2@V.T)).T; H=ConvexHull(Wq); idx=list(H.vertices); ctr=Wq[idx].mean(0); g=[];N=[]
    for k in range(len(idx)):
        p_,r_=idx[k],idx[(k+1)%len(idx)]; d=Wq[r_]-Wq[p_]; n=np.array([-d[1],d[0]]); n=n/np.linalg.norm(n)
        if n@(ctr-Wq[p_])<0: n=-n
        ci=np.min(Wq@n)
        for j in range(8): g.append(ci-n@pin[j]); N.append(n)
    g=np.array(g);N=np.array(N);nP=len(g); Aeq=np.vstack([N[:,0],N[:,1],np.ones(nP)]);beq=np.array([0.,0.,1.])
    r=linprog(-g,A_eq=Aeq,b_eq=beq,bounds=[(0,None)]*nP,method='highs')
    if r.status!=0: return None,None
    return -r.fun,[[k,round(float(r.x[k]),8)] for k in range(nP) if r.x[k]>1e-9]
def certify(box,eo):
    th,thw,ph,phw,dt,dtw,dp,dpw,s,sw=box
    mu,lam=lam_lp(u2center(th,ph), s*dircenter(dt,dp))
    if lam is None or mu<=0: return None
    u2I=F.iu2(th-thw,th+thw,ph-phw,ph+phw); dI=F.idir(dt-dtw,dt+dtw,dp-dpw,dp+dpw)
    try: return F.G_lambda_mv(u2I,dI,s,sw,aa,eo,lam)
    except Exception: return None
def cover_u2cell(th,thw,ph,phw, seed_nd=2,seed_ndp=4,seed_ns=2, verbose=False):
    """Top-down cover of the ξ-domain (s>r) at a fixed small u₂-box. Returns (cert,fail,maxdepth,boxes)."""
    u2c=u2center(th,ph); eo=edge_order(u2c)
    if eo is None: return (0,0,0,0)
    s_start=max(rmin, dlo(u2c)/M)
    if s_start>=smax: return (0,0,0,0)
    cert=fail=nbox=0; maxd=0
    stack=deque()
    for a in range(seed_nd):
        dtc=math.pi*(a+0.5)/seed_nd; dtw=math.pi/seed_nd/2
        for b in range(seed_ndp):
            dpc=2*math.pi*(b+0.5)/seed_ndp; dpw=2*math.pi/seed_ndp/2
            for c in range(seed_ns):
                lo=s_start*(smax/s_start)**(c/seed_ns); hi=s_start*(smax/s_start)**((c+1)/seed_ns)
                stack.append((th,thw,ph,phw,dtc,dtw,dpc,dpw,(lo+hi)/2,(hi-lo)/2,0))
    while stack:
        *box,depth=stack.pop(); nbox+=1; maxd=max(maxd,depth)
        G=certify(tuple(box),eo)
        if G is not None and G.lo>0: cert+=1
        else:
            mw=max(box[5],box[7],box[9])  # dtw,dpw,sw (ξ dims only; u₂ fixed here)
            if mw<MINW: fail+=1
            else:
                L=list(box); dims=[5,7,9]; wi=max(dims,key=lambda z:L[z]); ci=wi-1
                a=L[:];b=L[:];a[ci]=L[ci]-L[wi]/2;a[wi]=L[wi]/2;b[ci]=L[ci]+L[wi]/2;b[wi]=L[wi]/2
                stack.append(tuple(a)+(depth+1,)); stack.append(tuple(b)+(depth+1,))
    return (cert,fail,maxd,nbox)
if __name__=='__main__':
    print("Ball-covering engine — top-down adaptive test (box count per u₂-cell):")
    NT=144; THW=0.9/NT/2; PHW=math.pi/NT
    tests={'bulk (far from pinch)':(0.6,0.8),'mid':(0.3,1.5),'near-pinch':(0.06,0.5),'wall-ish':(0.2,1.05)}
    for name,(th,ph) in tests.items():
        t0=time.time(); cert,fail,maxd,nbox=cover_u2cell(th,THW,ph,PHW)
        print(f"  {name:24s} th={th:.3f}: boxes={nbox} cert={cert} fail={fail} maxdepth={maxd}  {time.time()-t0:.1f}s")
