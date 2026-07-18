"""(SF)(c) FLOAT PRE-FLIGHT of the final delta-free two-mode sweep decomposition.

Domain per phi-cell: in-plane N(D_Pi(phi), W_FR) x transverse box [-BT,BT]^3,
minus cells wholly inside corner disks cd_marg < RB (SB-box + iota_m territory).
Per cell, witnesses:
  [A] single tight row i: sup_cell f_i <= -eps;  kills delta <= eps/bracket_i
      (bracket_i = sup g_i + RHO0 sup|h3_i| + RHO0^2 C4); full kill if bracket<=0.
  [B] stress lambda >= 0 on ENDPOINT rows, Sum l =1, Sum l G(phi)=0
      => Sum l f == 0 identically (CS-1);  kills ALL delta in (0,RHO0] iff
      sup_cell Sum l g + RHO0 sup|Sum l h3| + RHO0^2 C4 < 0.
Cell passes iff [B] holds, or [A] holds with full kill, or [A]+[B] delta-split.
This is float EVIDENCE validating the decomposition; the exact port replaces
sampling+inflation with qinterval and lambda(phi) exact kernel bases.
"""
import os, pickle, numpy as np, sympy as sp, itertools, time
from scipy.optimize import linprog
_HERE = os.path.dirname(os.path.abspath(__file__))

RHO0 = 1e-3          # delta cap of Lemma 6' critical disks
C4   = 145.0         # provisional order-4 remainder constant (numeric est. separately)
RB   = 0.02          # inner corner cutoff (r71: SB-box BP=9/500, reach 0.0201141 > RB)
W_FR = 0.08          # localization fringe (Thm 14.3)
BT   = 0.0285        # transverse box (Thm 14.3)
PHI_W = float(np.arctan(11*np.sqrt(2)/9))

RES = pickle.load(open(os.path.join(_HERE, 'sf_rowbounds_sym.pkl'), 'rb'))
c, s = sp.symbols('c s', real=True)
Z = (0,0,0,0,0)
UNITS = [tuple(1 if q==i else 0 for q in range(5)) for i in range(5)]
QUADS = [(i,j) for i in range(5) for j in range(i,5)]
QMON  = {(i,j): tuple((2 if q==i else 0) if i==j else (1 if q in (i,j) else 0) for q in range(5)) for (i,j) in QUADS}
def is_ep(edge,j):
    a,b = edge.split('-'); return j in (a,b)

class RowEval:
    def __init__(self, cfg):
        dat = RES[cfg]['tight']
        self.labels = [f"{r['edge']}|{r['j']}" for r in dat]
        self.ep = np.array([is_ep(r['edge'], r['j']) for r in dat])
        self.n = len(dat)
        lam = lambda e: sp.lambdify((c,s), sp.sympify(e), 'numpy')
        self.drift_f = [lam(r['fd'].get(Z,0)) for r in dat]
        self.G_f  = [[lam(r['fd'].get(m,0)) for m in UNITS] for r in dat]
        self.g0_f = [lam(r['gd'].get(Z,0)) for r in dat]
        self.gl_f = [[lam(r['gd'].get(m,0)) for m in UNITS] for r in dat]
        self.gq_f = [{ij: lam(r['gd'].get(QMON[ij],0)) for ij in QUADS} for r in dat]
        self.h3_f = [{m: lam(co) for m,co in r['hd'].items()} for r in dat]
    def eval_at(self, phi):
        cc, ss = np.cos(phi), np.sin(phi)
        n = self.n
        drift = np.array([f(cc,ss) for f in self.drift_f], float)
        G = np.array([[f(cc,ss) for f in row] for row in self.G_f], float)
        g0 = np.array([f(cc,ss) for f in self.g0_f], float)
        gl = np.array([[f(cc,ss) for f in row] for row in self.gl_f], float)
        gq = np.zeros((n,5,5))
        for k in range(n):
            for (i,j) in QUADS:
                v = float(self.gq_f[k][(i,j)](cc,ss))
                if i==j: gq[k,i,i]=v
                else: gq[k,i,j]=gq[k,j,i]=v/2
        h3 = [{m: float(f(cc,ss)) for m,f in self.h3_f[k].items()} for k in range(n)]
        return dict(drift=drift, G=G, g0=g0, gl=gl, gq=gq, h3=h3)

EV = {cfg: RowEval(cfg) for cfg in RES}

def dpi_polygon(dat):
    """corners of D_Pi = {p: drift + G_Pi . p >= 0 all tight rows} (float)."""
    rows = [(dat['drift'][k], dat['G'][k,:2]) for k in range(len(dat['drift']))
            if np.linalg.norm(dat['G'][k,:2]) > 1e-9]
    pts=[]
    for (d1,g1),(d2,g2) in itertools.combinations(rows,2):
        A = np.array([g1,g2]); b = -np.array([d1,d2])
        if abs(np.linalg.det(A)) < 1e-9: continue
        p = np.linalg.solve(A,b)
        if all(dr + gr@p >= -1e-8 for dr,gr in rows): pts.append(p)
    out=[]
    for p in pts:
        if not any(np.linalg.norm(p-q) < 1e-7 for q in out): out.append(p)
    P = np.array(out)
    if len(P) >= 3:  # order by angle around centroid
        ctr = P.mean(0); ang = np.arctan2(P[:,1]-ctr[1], P[:,0]-ctr[0])
        P = P[np.argsort(ang)]
    return P

def dist_to_poly(pts, P):
    """distance from points (m,2) to polygon P (k,2) (0 if inside)."""
    if len(P) == 0: return np.full(len(pts), np.inf)
    k = len(P); m = len(pts)
    d = np.full(m, np.inf); inside = np.ones(m, bool)
    for i in range(k):
        a, b = P[i], P[(i+1)%k]; e = b-a; L2 = e@e
        t = np.clip(((pts-a)@e)/max(L2,1e-18), 0, 1)
        proj = a + t[:,None]*e
        d = np.minimum(d, np.linalg.norm(pts-proj, axis=1))
        # inside test via cross sign (assumes CCW ordering)
        cr = e[0]*(pts[:,1]-a[1]) - e[1]*(pts[:,0]-a[0])
        inside &= (cr >= -1e-12)
    return np.where(inside, 0.0, d)

# transverse sign patterns for sampling quadratics/cubics
TPATS = np.array([[0,0,0]] + [list(p) for p in itertools.product([-1,1],repeat=3)], float)*BT

def h3_abs_bound(h3w, bx):
    """sum_m |coef| * prod bound^m ; bx = per-coordinate abs bounds (5,)."""
    tot = 0.0
    for m, co in h3w.items():
        v = abs(co)
        for i,p in enumerate(m): v *= bx[i]**p
        tot += v
    return tot

def cell_check(dat, ep_idx, box, psi_cache=None):
    """box=(x0,x1,y0,y1). Returns dict with per-witness results at this phi sample."""
    x0,x1,y0,y1 = box
    corners = np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]])
    hw = max(x1-x0, y1-y0)/2
    drift, G, g0, gl, gq, h3 = dat['drift'], dat['G'], dat['g0'], dat['gl'], dat['gq'], dat['h3']
    n = len(drift)
    # --- witness A: per-row sup f over cell (linear: corners + |G_T|*BT) ---
    supf = drift + (corners@G[:,:2].T).max(0) + np.abs(G[:,2:]).sum(1)*BT
    # per-row bracket: sup g_i + RHO0*sup|h3_i| + RHO0^2*C4  (coarse abs bounds)
    bxabs = np.array([max(abs(x0),abs(x1)), max(abs(y0),abs(y1)), BT, BT, BT])
    res = dict(A=[], B=None)
    for i in range(n):
        if supf[i] < -1e-9:
            # sup of g_i over cell: samples + curvature inflation
            X = np.array([[px,py,*tp] for (px,py) in corners for tp in TPATS] +
                         [[(x0+x1)/2,(y0+y1)/2,*tp] for tp in TPATS])
            gv = g0[i] + X@gl[i] + np.einsum('mi,ij,mj->m', X, gq[i], X)
            Hn = np.linalg.norm(gq[i][:2,:2],2)
            supg = gv.max() + Hn*2*hw**2
            br = supg + RHO0*h3_abs_bound(h3[i], bxabs) + RHO0**2*C4
            dmax = np.inf if br <= 0 else (-supf[i])/br
            res['A'].append((i, supf[i], br, dmax))
    # --- witness B: stress LP over endpoint rows, objective g at cell center ---
    ctr = np.array([(x0+x1)/2,(y0+y1)/2,0,0,0])
    Ge = G[ep_idx]; ne = len(ep_idx)
    gval = g0[ep_idx] + ctr@gl[ep_idx].T + np.einsum('i,kij,j->k', ctr, gq[ep_idx], ctr)
    Aeq = np.vstack([np.ones((1,ne)), Ge.T]); beq = np.zeros(6); beq[0]=1
    r = linprog(gval, A_eq=Aeq, b_eq=beq, bounds=[(0,None)]*ne, method='highs')
    if r.status == 0:
        lamb = r.x
        # sup of Sum l g over cell: samples + inflation
        glw = lamb@gl[ep_idx]; gqw = np.einsum('k,kij->ij', lamb, gq[ep_idx]); g0w = lamb@g0[ep_idx]
        X = np.array([[px,py,*tp] for (px,py) in corners for tp in TPATS] +
                     [[(x0+x1)/2,(y0+y1)/2,*tp] for tp in TPATS])
        gv = g0w + X@glw + np.einsum('mi,ij,mj->m', X, gqw, X)
        Hn = np.linalg.norm(gqw[:2,:2],2)
        supg = gv.max() + Hn*2*hw**2
        h3w = {}
        for lk, k in zip(lamb, ep_idx):
            if lk < 1e-12: continue
            for m, co in h3[k].items(): h3w[m] = h3w.get(m,0.) + lk*co
        c3sup = h3_abs_bound(h3w, bxabs)
        killB = supg + RHO0*c3sup + RHO0**2*C4
        res['B'] = (float(-r.fun), supg, c3sup, killB, lamb)
    return res

def verdict(resL):
    """combine results at the phi samples: pass iff at every sample
    (B kills) or (A full kill) or (A dmax >= RHO0) -- delta-split allowed only
    if some A covers [0,d1) and B covers all (B is delta-uniform, so B alone suffices);
    effectively: B killB<0  OR  exists A with dmax >= RHO0."""
    out=[]
    for res in resL:
        okB = res['B'] is not None and res['B'][3] < 0
        okA = any(dm >= RHO0 for (_,_,_,dm) in res['A'])
        out.append(okB or okA)
    return all(out)
print('module ok')
