"""wall_sweep2.py -- adaptive certified sweep for (SB-box) bulk, v2:
inversion-free p-point certificates over two-symbol Taylor forms.
Proves: delta*K_box subset conv F(u2(delta,phi)) for delta in [1e-5,1.28e-3],
phi in [0,pi], with two-case wall dichotomy at both silhouette walls."""
import numpy as np, itertools, json, time, sys
sys.path.insert(0,'certificates')
from flagcalc2 import *
from flagcalc2 import _up, _dn, _split

# ---------- float helpers ----------
def _skew(w): return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
_e1=np.array([0,0,1.0]); _e2=np.array([-1,-1,0])/np.sqrt(2); _us=np.array([1,-1,0])/np.sqrt(2)
def qz_float(delta,phi):
    w=-np.sin(phi)*_e1+np.cos(phi)*_e2; K=_skew(w)
    R=np.eye(3)+np.sin(delta)*K+(1-np.cos(delta))*(K@K)
    return np.stack([VERTS@(R@_e1),VERTS@(R@_e2)],axis=1), VERTS@(R@_us)
_J=np.array([[0,-1],[1,0]])
def flags_float(delta,phi,cfg,Lh=None):
    q,z=qz_float(delta,phi); hull=HULLS[cfg]; F=[]; m=len(hull)
    for i in range(m):
        A,B=hull[i],hull[(i+1)%m]
        n=_J@(q[B]-q[A])
        n=n/(np.linalg.norm(n) if Lh is None else Lh[i])
        for jj in (A,B): F.append(np.concatenate([z[jj]*n,[n@(_J@q[jj])],n]))
    return np.array(F)

# ---------- inflated generators + once-verified support margin ----------
bPi,bg,bt = 9/500, 1/25, 1/500   # r71: bPi enlarged (annulus reach >= 1/50)
c8=np.cos(np.pi/8)
KPI, KG, KT = 1.15, 1.30, 2.00      # per-block inflation
oct8=np.array([[np.cos(k*np.pi/4),np.sin(k*np.pi/4)] for k in range(8)])
GENS=np.array([np.concatenate([o1,[sg],o2])
    for o1 in oct8*(KPI*bPi/c8) for sg in (KG*bg,-KG*bg) for o2 in oct8*(KT*bt/c8)])

def octagon_inradius_lb(verts):
    """verified lower bound on inradius of conv(verts) (contains 0), interval."""
    lb=np.inf
    m=len(verts)
    for i in range(m):
        w1,w2=verts[i],verts[(i+1)%m]
        cr=isub(imul(ithin(w1[0]),ithin(w2[1])),imul(ithin(w1[1]),ithin(w2[0])))
        dx=isub(ithin(w2[0]),ithin(w1[0])); dy=isub(ithin(w2[1]),ithin(w1[1]))
        ln=(_dn(np.sqrt(max(0.0,(imul(dx,dx)[0]+imul(dy,dy)[0])))),
            _up(np.sqrt(imul(dx,dx)[1]+imul(dy,dy)[1])))
        assert cr[0]>0, "orientation"
        lb=min(lb, cr[0]/ln[1])
    return lb
RHO_PI = octagon_inradius_lb(oct8*(KPI*bPi/c8))
RHO_T  = octagon_inradius_lb(oct8*(KT*bt/c8))
ETA_BUDGET = min(RHO_PI-bPi, KG*bg-bg, RHO_T-bt)   # support margin (proved in notes)

# candidate flag pools: for hex1 exclude the short-edge (v3-v2) flags 6,7 --
# they carry O(1/delta) curvature; margins are better without them (round 57).
_POOL={'pentA':list(range(10)),'pentB':list(range(10)),
       'hex1':[i for i in range(12) if i not in (6,7)]}
SUBS={cfg: np.array(list(itertools.combinations(_POOL[cfg],6))) for cfg in HULLS}

def candidates(cfg,dc,fc,Lh=None):
    F=flags_float(dc,fc,cfg,Lh); subs=SUBS[cfg]; nG=len(GENS)
    Bb=np.concatenate([F[subs].transpose(0,2,1),np.ones((len(subs),1,6))],axis=1)
    dets=np.linalg.det(Bb); good=np.abs(dets)>1e-8
    rhs=np.vstack([(dc*GENS).T,np.ones(nG)])
    marg=np.full((len(subs),nG),-np.inf); lam=np.zeros((len(subs),6,nG))
    idx=np.where(good)[0]
    lam[idx]=np.linalg.solve(Bb[idx],np.broadcast_to(rhs,(len(idx),6,nG)))
    marg[idx]=lam[idx].min(axis=1)
    # conditioning-aware choice: among subsets within 0.6 of best margin per
    # generator, take the best-conditioned (max smallest singular value).
    sig=np.zeros(len(subs))
    sig[idx]=np.linalg.svd(Bb[idx],compute_uv=False)[:,-1]
    bestm=marg.max(axis=0)
    okm=marg >= np.maximum(0.6*bestm, bestm - 0.2*np.abs(bestm))[None,:]
    sigm=np.where(okm, sig[:,None], -1.0)
    best=sigm.argmax(axis=0)
    hd,hf=max(1e-8,0.01*dc),1e-5
    Ftp=flags_float(dc+hd,fc,cfg,Lh); Ftm=flags_float(dc-hd,fc,cfg,Lh)
    Fsp=flags_float(dc,fc+hf,cfg,Lh); Fsm=flags_float(dc,fc-hf,cfg,Lh)
    Fpp=flags_float(dc+hd,fc+hf,cfg,Lh); Fpm=flags_float(dc+hd,fc-hf,cfg,Lh)
    Fmp=flags_float(dc-hd,fc+hf,cfg,Lh); Fmm=flags_float(dc-hd,fc-hf,cfg,Lh)
    Ft=(Ftp-Ftm)/(2*hd); Fs=(Fsp-Fsm)/(2*hf); Fts=(Fpp-Fpm-Fmp+Fmm)/(4*hd*hf)
    out={}
    for si in np.unique(best):
        gi=np.where(best==si)[0]; S=tuple(int(x) for x in subs[si])
        sl=list(S)
        B=np.concatenate([F[sl].T,np.ones((1,6))],axis=0)
        Bt=np.concatenate([Ft[sl].T,np.zeros((1,6))],axis=0)
        Bs=np.concatenate([Fs[sl].T,np.zeros((1,6))],axis=0)
        Bts=np.concatenate([Fts[sl].T,np.zeros((1,6))],axis=0)
        l0=lam[si][:,gi]
        Gaug=np.vstack([GENS[gi].T,np.zeros(len(gi))])
        l1=np.linalg.solve(B,Gaug-Bt@l0)
        l2=np.linalg.solve(B,-Bs@l0)
        l3=np.linalg.solve(B,-(Bt@l2+Bs@l1+Bts@l0))
        out[S]=(gi,l0,l1,l2,l3)
    return out

# ---------- per-cell certification ----------
def _pos(f,w,v): return f2hull(f,w,v)[0]
def cell_checks(q,z,w,v,cfg,skip=None,extra=()):
    hull=HULLS[cfg]; m=len(hull); mn=np.inf
    for i in range(m):
        A,B,C=hull[i-1],hull[i],hull[(i+1)%m]
        val=_pos(cross_form(q[A],q[B],q[C],w,v),w,v)
        if val<=0: return -np.inf
        mn=min(mn,val)
    for sxx in INTER[cfg]:
        for i in range(m):
            A,B=hull[i],hull[(i+1)%m]
            if skip is not None and skip==(sxx,(A,B)): continue
            val=_pos(side_form(q[A],q[B],q[sxx],w,v),w,v)
            if val<=0: return -np.inf
            mn=min(mn,val)
    for (A,B,C) in extra:
        val=_pos(cross_form(q[A],q[B],q[C],w,v),w,v)
        if val<=0: return -np.inf
        mn=min(mn,val)
    return mn

def try_cell(dlo,dhi,flo,fhi,mode,frames=None):
    if frames is None: frames=frame_forms(dlo,dhi,flo,fhi)
    q,z,w,v,dc,fc=frames
    def member(cfg):
        F,Lh=flags_forms(q,z,HULLS[cfg],w,v)
        cand=candidates(cfg,dc,fc,Lh)
        return verify_ppoint(F,cand,GENS,dc,w,v,dlo,ETA_BUDGET)
    if mode in ('pentA','hex1','pentB'):
        if cell_checks(q,z,w,v,mode)<=0: return False,-np.inf,np.inf
        ok,sl,eta=member(mode); return ok,sl,eta
    if mode=='twoA':
        if cell_checks(q,z,w,v,'pentA',skip=(2,(4,1)))<=0: return False,-np.inf,np.inf
        g2=min(_pos(side_form(q[4],q[2],q[s],w,v),w,v) for s in (5,6))
        g3=min(_pos(side_form(q[2],q[1],q[s],w,v),w,v) for s in (5,6))
        gt=min(_pos(cross_form(q[a],q[b],q[c],w,v),w,v) for (a,b,c) in [(3,4,2),(2,1,7)])
        if min(g2,g3,gt)<=0: return False,-np.inf,np.inf
        ok1,s1,e1=member('pentA')
        if not ok1: return False,s1,e1
        ok2,s2,e2=member('hex1')
        return ok2,min(s1,s2),max(e1,e2)
    if mode=='twoB':
        if cell_checks(q,z,w,v,'pentB',skip=(1,(2,7)))<=0: return False,-np.inf,np.inf
        g2=min(_pos(side_form(q[2],q[1],q[s],w,v),w,v) for s in (5,6))
        g3=min(_pos(side_form(q[1],q[7],q[s],w,v),w,v) for s in (5,6))
        gt=min(_pos(cross_form(q[a],q[b],q[c],w,v),w,v) for (a,b,c) in [(4,2,1),(1,7,0)])
        if min(g2,g3,gt)<=0: return False,-np.inf,np.inf
        ok1,s1,e1=member('pentB')
        if not ok1: return False,s1,e1
        ok2,s2,e2=member('hex1')
        return ok2,min(s1,s2),max(e1,e2)
    raise ValueError(mode)

def modes_for(flo,fhi):
    fc=0.5*(flo+fhi)
    if fhi<PHI_W-0.02: return ['pentA']
    if flo>PHI_W+0.02 and fhi<np.pi-PHI_W-0.02: return ['hex1']
    if flo>np.pi-PHI_W+0.02: return ['pentB']
    if fc<np.pi/2: return ['pentA','hex1','twoA']
    return ['pentB','hex1','twoB']

def sweep_level(d1,d2,f1=0.0,f2=np.pi,init_n=64,maxdepth=48):
    cells=[(d1,d2,f1+(f2-f1)*k/init_n,f1+(f2-f1)*(k+1)/init_n,0) for k in range(init_n)]
    done=[]; nfail=0; minsl=np.inf; maxeta=0.0; t0=time.time()
    while cells:
        dlo,dhi,flo,fhi,dep=cells.pop()
        ok=False; sl=-np.inf; eta=np.inf; used=None
        frames=frame_forms(dlo,dhi,flo,fhi)
        for mode in modes_for(flo,fhi):
            ok,sl,eta=try_cell(dlo,dhi,flo,fhi,mode,frames)
            if ok: used=mode; break
        if ok:
            done.append((dlo,dhi,flo,fhi,used,sl,eta)); minsl=min(minsl,sl); maxeta=max(maxeta,eta); continue
        if dep>=maxdepth:
            nfail+=1; done.append((dlo,dhi,flo,fhi,'FAIL',sl,eta)); continue
        w=(dhi-dlo)/2; v=(fhi-flo)/2
        if v*v >= 0.5*w*w/dlo:   # split the dominant error contribution
            fm=0.5*(flo+fhi); cells.append((dlo,dhi,flo,fm,dep+1)); cells.append((dlo,dhi,fm,fhi,dep+1))
        else:
            dm=0.5*(dlo+dhi); cells.append((dlo,dm,flo,fhi,dep+1)); cells.append((dm,dhi,flo,fhi,dep+1))
    return done,nfail,minsl,maxeta,time.time()-t0
