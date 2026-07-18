"""(SF)(c) exact witness sweep driver.

Per phi-cell (exact rational (c,s)-box on the unit circle, per chart) and per in-plane
cell (rational AABB, or W3 polar sector at a marginal corner), FIND a witness with
float LPs (heuristic) and EXACT-VERIFY it with sf_exact2 (all-Fraction interval
arithmetic + exact constant stresses over Q(sqrt2,sqrt323,sqrt1123)).

Witness modes (exact statements in sf_exact2):
  A  : single-row first-order kill           (wa2)
  B  : full stress, whole transverse box     (wb2)
  W2 : orthant stress, 8 transverse orthants (ww22)
  W3 : radial-sector orthant stress at the marginal corners (w32; CS-1 / CS-1-far)

Manifest rows (jsonl, one shard per phi-cell, .done markers => resumable):
  {"kind":"meta", "phicell":[cfg,pkind,lo,hi], "pairs":[[i,j],...], "h0":..., "note":...}
  {"kind":"cell", "phicell":..., "cell":{...}, "mode":"A|B|W2|W3", "data":{...}}
  {"kind":"fail", ...}   (recorded, sweep must end with 0 fails)

Usage:
  python3 sf_make_witness.py --list                    # print planned phi-cells
  python3 sf_make_witness.py --cells 12,13,14          # run specific phi-cells
  python3 sf_make_witness.py --all --procs 8           # full sweep (LONG: hours)
  python3 sf_make_witness.py --validate                # 5-cell mini validation
Shards land in sf_shards/shard_<idx>.jsonl (+ .done).
"""
import os, sys, json, time, argparse, itertools, math
from fractions import Fraction as Fr
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
import sf_exact2 as X
import sf_preflight as SF
import qinterval as QI
from scipy.optimize import linprog

RB = X.RB; TLAST = X.TLAST; BT = X.BT
FR114 = Fr(114, 1000)          # l-inf inflation >= 0.08*sqrt2 (localization tube)
RF_FAR = Fr(7, 20)             # Lemma FT far-disk radius (r69: enlarged 0.1 -> 0.35;
                               #  Sigma = sqrt(RF_FAR^2+Gamma^2) <= 0.3512, landing
                               #  dist >= 2-Sigma = 1.6488 > RF_FAR; rho0*Sigma << sigma0)
H0 = Fr(1, 25)                 # Cartesian grid step
BANDS = [RB, Fr(3,100), Fr(9,200), Fr(7,100), TLAST]   # r71: RB=1/50 (first band merged into SB-box disk)
NW0 = 8                        # sectors per half-circle (initial)
C0REQS = [Fr(1,50), Fr(3,500), Fr(1,250), Fr(1,100)] # multi-c0 for W2/W3 (r69: +1/250,1/100 -- the B0<0 / c0>=rho0*B1 window can fall between 1/50 and 3/500)
MAXDEPTH = 8                   # in-plane quadtree depth (r-final: 4->8; near-corner cart cells
                               # close at <=7 with existing certs+WC -- subdivision-depth, NOT a new cert)
W3_DEPTH = 6                   # W3 radial-sector adaptive depth (w x tau 4-way split, only on fail)
                               # r-final: 6->10; hex1 near-corner W3 marginal orthants close at +1 level
                               # (verified: 7/7 fail cells resolve with 1 extra level; existing machinery,
                               #  NOT a new lemma). Non-marginal cells still close early -> no extra cost.
SHARD_DIR = os.path.join(_D, 'sf_shards')

# ---------------------------------------------------------------- phi cells
def _sgrade(wall, coarse, fine1, fine2):
    """breakpoints 0..0.87 in s with wall grading (wall ~ 0.86564...)."""
    bks = []
    x = Fr(0)
    while x < Fr(62,100): bks.append(x); x += coarse
    # r78: wing refinement -- pentA/pentB far-corner wing cart cells (s in [0.62,0.8])
    # fail at coarse=1/50 because the phi-drift (~0.05-0.065) swamps the O(0.005-0.03)
    # first-order margins there; at width 1/250 the same boxes certify (probe r78:
    # shard-37 worst box passes W2). Also widens apo_far (cdrift ~0.011) so the
    # dfar<=0.33 annulus routes to Lemma FT.
    while x < Fr(80,100): bks.append(x); x += Fr(1,250)
    while x < Fr(84,100): bks.append(x); x += fine1
    # cdrift (far-corner motion across the phi-cell) grows approaching the wall, shrinking the cart
    # 16-gon exemption apo below the near-corner cart cells -> they fall in the cart<->W3 seam and
    # fail.  Halve the step over [0.84,0.86] so cdrift ~halves and apo re-covers them (they route to
    # W3, which is proven out to TLAST).  (triage: whole-cell apo=0.0553 MISSES 0.0575; half apo=0.074.)
    while x < Fr(86,100): bks.append(x); x += fine1 / 2
    while x < Fr(87,100): bks.append(x); x += fine2
    bks.append(Fr(87,100))
    return bks

def phi_cells():
    """list of (cfg, pkind, lo, hi) with exact rational endpoints.
    pentA/pentB parametrized by s in [0, 0.87] (0.87^2=0.7569 > 242/323 => beyond wall);
    hex1 by c in [-0.501, 0.501] (0.501^2 > 81/323 => beyond both walls)."""
    cells = []
    bks = _sgrade(None, Fr(1,50), Fr(1,100), Fr(1,500))
    for cfg in ('pentA', 'pentB'):
        for a, b in zip(bks[:-1], bks[1:]):
            cells.append((cfg, 's', a, b))
    cbks = []
    x = Fr(-501, 1000)
    cbks.append(x); x = Fr(-499,1000)
    while x < Fr(499,1000): cbks.append(x); x += Fr(1,50)
    cbks += [Fr(499,1000), Fr(501,1000)]
    for a, b in zip(cbks[:-1], cbks[1:]):
        cells.append(('hex1', 'c', a, b))
    return cells

def phis_of(cfg, pkind, lo, hi):
    """float phi samples [lo,mid,hi] of the arc."""
    if pkind == 's':
        if cfg == 'pentA':
            ph = [math.asin(float(lo)), math.asin(float(hi))]
        else:
            ph = [math.pi - math.asin(float(hi)), math.pi - math.asin(float(lo))]
    else:
        ph = [math.acos(float(hi)), math.acos(float(lo))]
    return [ph[0], (ph[0]+ph[1])/2, ph[1]]

def csbox_of(cfg, pkind, lo, hi):
    return X.csbox_of(cfg, pkind, lo, hi)

# ---------------------------------------------------------------- geometry helpers
def dpi_vertices(dat):
    """float D_Pi vertices with their defining row pairs."""
    n = len(dat['drift'])
    rows = [(k, dat['drift'][k], dat['G'][k,:2]) for k in range(n)
            if np.linalg.norm(dat['G'][k,:2]) > 1e-9]
    verts = []
    for (i,d1,g1),(j,d2,g2) in itertools.combinations(rows,2):
        Amat = np.array([g1,g2]); b = -np.array([d1,d2])
        if abs(np.linalg.det(Amat)) < 1e-9: continue
        p = np.linalg.solve(Amat,b)
        if all(dr + gr@p >= -1e-8 for (_,dr,gr) in rows):
            if not any(np.linalg.norm(p-q) < 1e-7 for q,_ in verts):
                verts.append((p,(i,j)))
    return verts

def gen_cart_cells(cfg, pkind, lo, hi):
    """rational Cartesian AABBs covering the target region minus corner 16-gons."""
    phis = phis_of(cfg, pkind, lo, hi)
    dats = [SF.EV[cfg].eval_at(p) for p in phis]
    corners = [np.array([[0.,0.],[2*math.cos(p),2*math.sin(p)]]) for p in phis]
    vsets = [dpi_vertices(d) for d in dats]
    # heuristic keep/drop region (verifier is the authority): union polygon of all
    # sample vertices + exact corners; robust at wall straddles where a sample's
    # pentagon-config polygon may be empty/degenerate beyond the wall.
    allv = [v for vs in vsets for v, _ in vs] + [cs[k] for cs in corners for k in range(2)]
    P = np.array(allv)
    def order(Q):
        ctr = Q.mean(0); ang = np.arctan2(Q[:,1]-ctr[1], Q[:,0]-ctr[0])
        return Q[np.argsort(ang)]
    try:
        from scipy.spatial import ConvexHull
        P = P[ConvexHull(P).vertices]
    except Exception:
        P = order(P)
    polys = [P]
    polysO = [P]
    cdrift = max(np.linalg.norm(corners[0][1]-corners[-1][1]), 0.0)
    apo = (0.1 - cdrift - 0.003) * math.cos(math.pi/16)
    allpts = np.concatenate(polys)
    lo2 = allpts.min(0) - float(FR114) - 0.05; hi2 = allpts.max(0) + float(FR114) + 0.05
    i0 = math.floor(lo2[0]/float(H0)); i1 = math.ceil(hi2[0]/float(H0))
    j0 = math.floor(lo2[1]/float(H0)); j1 = math.ceil(hi2[1]/float(H0))
    cells = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            x0, x1 = i*H0, (i+1)*H0; y0, y1 = j*H0, (j+1)*H0
            cc = np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]], float)
            hd = float(H0)*0.7072
            if min(SF.dist_to_poly(cc, P).min() for P in polysO) > float(FR114) + hd + 0.02:
                continue
            dropped = False
            for ci in range(2):
                ctr_mid = corners[1][ci]
                if max(np.linalg.norm(cc - ctr_mid, axis=1)) < apo:
                    dropped = True
            if dropped: continue
            cells.append((x0, x1, y0, y1))
    vs_meta = max(vsets, key=len)
    meta = dict(pairs=[[int(a),int(b)] for _,(a,b) in vs_meta],
                verts=[[float(v[0]),float(v[1])] for v,_ in vs_meta],
                cdrift=float(cdrift))
    return cells, meta

def gen_w3_cells():
    """(corner, half, w-interval, band) list; sectors tile both halves exactly.
    LEMMA FT (r67): the far-corner disk (cd_far <= TLAST) is discharged by the exact
    sheet transport N R1 m* = exp([-N eta]x) W(u2): the containment system there is
    IDENTICAL (Thm 10.3 T2) to one at relative angle ||eta|| <= delta*sqrt(TLAST^2+GAM^2)
    from the diagonal, whose rescaled tilt (norm <= 0.10405) lands in the already-
    certified near region (cart cells + near W3 + SB-box).  Sheet position is EXACT:
    zeta = 2*delta*(-sin phi, cos phi, 0)  [sym-verified].  So only 'near' W3 is needed."""
    out = []
    for corner in ('near',):
        for half in (1,-1):
            for k in range(NW0):
                w0 = Fr(-1) + Fr(2*k, NW0); w1 = Fr(-1) + Fr(2*(k+1), NW0)
                for b0, b1 in zip(BANDS[:-1], BANDS[1:]):
                    out.append((corner, half, w0, w1, b0, b1))
    return out

# ---------------------------------------------------------------- float finders
def find_wa(cfg, dats, box):
    x0,x1,y0,y1 = [float(v) for v in box]
    cc = np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]])
    best = None
    for dat in dats:
        supf = dat['drift'] + (cc@dat['G'][:,:2].T).max(0) + np.abs(dat['G'][:,2:]).sum(1)*float(BT)
        i = int(np.argmin(supf))
        if best is None or supf[i] > best[1]: best = (i, supf[i])  # worst sample, best row
    return best

def find_wb(cfg, dat, box):
    pool = X.stress_pool(cfg)
    ctr = np.array([(float(box[0])+float(box[1]))/2,(float(box[2])+float(box[3]))/2,0,0,0])
    gvc = dat['g0'][pool] + ctr@dat['gl'][pool].T + np.einsum('i,kij,j->k',ctr,dat['gq'][pool],ctr)
    Ge = dat['G'][pool]; ne = len(pool)
    Aeq = np.vstack([np.ones((1,ne)),Ge.T]); beq = np.zeros(6); beq[0]=1
    r = linprog(gvc,A_eq=Aeq,b_eq=beq,bounds=[(0,None)]*ne,method='highs')
    if r.status != 0: return None
    return tuple(int(pool[k]) for k in range(ne) if r.x[k] > 1e-9)

def find_w2(cfg, dat, obj_pt, sigma, c0req, radial=None):
    """orthant stress LP; objective = g at obj_pt, or radial slope if radial=(corner,e)."""
    pool = X.stress_pool(cfg) if not (radial and radial[0]=='far') else X.z_rows(cfg)
    Ge = dat['G'][pool]; ne = len(pool)
    if radial is None:
        ctr = np.array([obj_pt[0],obj_pt[1],0,0,0])
        obj = dat['g0'][pool] + ctr@dat['gl'][pool].T + np.einsum('i,kij,j->k',ctr,dat['gq'][pool],ctr)
    else:
        corner, e, P0, tmid = radial
        p = np.array([P0[0]+tmid*e[0], P0[1]+tmid*e[1], 0,0,0])
        obj = (dat['g0'][pool] + p@dat['gl'][pool].T
               + np.einsum('i,kij,j->k',p,dat['gq'][pool],p))
        if corner == 'near':
            obj = obj - dat['g0'][pool]   # CS-1: constant term is 0 anyway
    Aeq = np.zeros((3,ne)); Aeq[0]=1; Aeq[1]=Ge[:,0]; Aeq[2]=Ge[:,1]
    Aub = np.zeros((3,ne))
    for k in range(3): Aub[k] = Ge[:,2+k]*sigma[k]
    r = linprog(obj,A_ub=Aub,b_ub=-float(c0req)*np.ones(3),A_eq=Aeq,b_eq=[1,0,0],
                bounds=[(0,None)]*ne,method='highs')
    if r.status != 0: return None
    supp = tuple(int(pool[k]) for k in range(ne) if r.x[k] > 1e-9)
    v = -r.x@Ge[:,2:]
    acts = tuple(int(k) for k in range(3) if abs(v[k]*sigma[k]-float(c0req)) < 1e-8)
    return supp, acts


def find_w1(cfg, dats, box_xbox_corners, sub_sigma):
    """float combo-f LP over the corners of (in-plane box x tau sub-box).
    r68 patch: ROBUST across the whole phi-cell -- constraints from ALL phi samples
    (dats may be a list of eval_at dicts, or a single dict).  The r67 version used
    only the phi-midpoint; witnesses could genuinely fail at the phi endpoints
    (shard 46/47 near-corner fails: exact interval was TIGHT, witness was wrong)."""
    if isinstance(dats, dict): dats = [dats]
    n = len(dats[0]['drift'])
    (x0,x1,y0,y1) = box_xbox_corners
    tp = [np.array([a,b,cc]) for a in ((0,float(BT)*sub_sigma[0]) if True else ())
          for b in ((0,float(BT)*sub_sigma[1]))
          for cc in ((0,float(BT)*sub_sigma[2]))]
    XC = np.array([[px,py,*t_] for px in (float(x0),float(x1)) for py in (float(y0),float(y1))
                   for t_ in tp])
    FV = np.vstack([dat['drift'][None,:] + XC@dat['G'].T for dat in dats])
    nv = n+1
    Aub = np.hstack([FV, -np.ones((len(FV),1))])
    A1 = np.zeros((1,nv)); A1[0,:n] = 1
    r = linprog(np.eye(nv)[-1], A_ub=Aub, b_ub=np.zeros(len(FV)),
                A_eq=A1, b_eq=[1], bounds=[(0,None)]*n+[(None,None)], method='highs')
    if r.status != 0 or r.x[-1] > -1e-6: return None
    lam = r.x[:n]
    keep = [k for k in range(n) if lam[k] > 1e-9]
    tot = sum(lam[k] for k in keep)
    fr = [Fr(lam[k]/tot).limit_denominator(10**6) for k in keep]
    fr[-1] = 1 - sum(fr[:-1])
    if any(l < 0 for l in fr): return None
    return fr, keep



def find_w4(cfg, dats, sigma, w0, w1, b0, b1, c0req, conem=1e-4):
    """float LP for the W4 sector-cone mixed kill (r73, wall-edge sectors).
    Variables lam over stress_pool (constant exact G, zero drift) + epigraph z:
      cone:        Sum lam G_Pi . e(w)  <= -conem   at 5 w-samples (exact recheck in w42)
      transverse:  Sum lam G_{2+k} sigma_k <= -c0req
      ray:         z >= Sum lam g(t e(w), tau=0)    (all phi samples x ray grid)
      min z ; accept z < -2e-4.  Returns (lam_fr rational sum=1, rows)."""
    pool = X.stress_pool(cfg)
    G = dats[0]['G'][pool]; ne = len(pool)
    th0, th1 = 2*math.atan(float(w0)), 2*math.atan(float(w1))
    cons = []; rhs = []
    for th in np.linspace(th0, th1, 5):
        cons.append(np.append(G[:,0]*math.cos(th) + G[:,1]*math.sin(th), 0.0)); rhs.append(-conem)
    for k in range(3):
        cons.append(np.append(G[:,2+k]*sigma[k], 0.0)); rhs.append(-float(c0req))
    for dat in dats:
        g0 = dat['g0'][pool]; gl = dat['gl'][pool]; gq = dat['gq'][pool]
        for th in np.linspace(th0, th1, 3):
            e = np.array([math.cos(th), math.sin(th)])
            for tv in (float(b0), (float(b0)+float(b1))/2, float(b1)):
                p = np.array([tv*e[0], tv*e[1], 0., 0., 0.])
                gv = g0 + gl@p + np.einsum('kij,i,j->k', gq, p, p)
                cons.append(np.append(gv, -1.0)); rhs.append(0.0)
    Aeq = np.zeros((1, ne+1)); Aeq[0,:ne] = 1
    cobj = np.zeros(ne+1); cobj[-1] = 1.0
    r = linprog(cobj, A_ub=np.array(cons), b_ub=np.array(rhs), A_eq=Aeq, b_eq=[1],
                bounds=[(0,None)]*ne + [(None,None)], method='highs')
    if r.status != 0 or r.x[-1] > -2e-4: return None
    lam = r.x[:ne]
    keep = [k for k in range(ne) if lam[k] > 1e-9]
    tot = sum(lam[k] for k in keep)
    fr = [Fr(lam[k]/tot).limit_denominator(10**6) for k in keep]
    fr[-1] = 1 - sum(fr[:-1])
    if any(l < 0 for l in fr): return None
    return fr, [int(pool[k]) for k in keep]



def find_h(cfg, dats, sigma, w0, w1, b0, b1):
    """float LP for the Lemma-H homogeneous transverse Farkas kill (r76/77).
    Variables lam >= 0 over R_H = h_rows(cfg) (pure-transverse rows), Sum lam = 1.
      H1 (per transverse k):  Sum lam [ max_phi G[i,2+k]*sigma_k + RHO0*L_ik ] <= -1e-6
      H2 (objective):         min  Sum lam [ ghat_i + RHO0*h3sup_i + RHO0^2*C4 ]
    ghat_i = max over (phi, w, t) samples of g_i(t e(w), tau=0); L_ik = float bound of
    sup |d g_i / d tau_k| over the sector AABB x BT tau-box.  Accept if the objective
    is < -8e-5; exact h2 is the authority.  Returns (lam_fr rational sum=1, rows)."""
    RH = X.h_rows(cfg)
    n = len(RH)
    if n == 0: return None
    th0, th1 = 2*math.atan(float(w0)), 2*math.atan(float(w1))
    ths = np.linspace(th0, th1, 5)
    tf0, tf1 = float(b0), float(b1)
    bmax = np.array([tf1, tf1, float(X.BT), float(X.BT), float(X.BT)])
    RHO = float(X.RHO0); C4f = float(X.C4)
    a = np.zeros((3, n)); obj = np.zeros(n)
    for col, i in enumerate(RH):
        gh = -1e18
        for dat in dats:
            for k in range(3):
                L = abs(dat['gl'][i, 2+k]) + 2.0*np.abs(dat['gq'][i, 2+k, :])@bmax
                a[k, col] = max(a[k, col] if dat is not dats[0] else -1e18,
                                dat['G'][i, 2+k]*sigma[k] + RHO*L)
            for th in ths:
                e = np.array([math.cos(th), math.sin(th)])
                for tv in (tf0, (tf0+tf1)/2, tf1):
                    p = np.array([tv*e[0], tv*e[1], 0., 0., 0.])
                    gh = max(gh, dat['g0'][i] + p@dat['gl'][i] + p@dat['gq'][i]@p)
        h3s = max(sum(abs(v)*np.prod([bmax[q]**m[q] for q in range(5)])
                      for m, v in d['h3'][i].items()) for d in dats)
        obj[col] = gh + RHO*h3s + RHO*RHO*C4f
    Aub = a; bub = -5e-6*np.ones(3)
    Aeq = np.ones((1, n))
    r = linprog(obj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1],
                bounds=[(0, None)]*n, method='highs')
    if r.status != 0 or r.fun > -8e-5: return None
    lam = r.x
    keep = [k for k in range(n) if lam[k] > 1e-9]
    tot = sum(lam[k] for k in keep)
    fr = [Fr(lam[k]/tot).limit_denominator(10**6) for k in keep]
    fr[-1] = 1 - sum(fr[:-1])
    if any(l < 0 for l in fr): return None
    return fr, [int(RH[k]) for k in keep]

def find_wc(cfg, dat, box, sigma, c0req):
    """float LP for the W-C cart cone-relaxed stress kill (hex1 near-corner cart cells).
    In-plane one-signed over the AABB corners (non-strict cone) + transverse inward
    (>= c0req) + strictly-negative 2nd-order combo over stress_pool.  Exact recheck in
    X.wc2.  Returns (lam_fr rational sum=1, rows) or None."""
    pool = X.stress_pool(cfg)
    G = dat['G'][pool]; ne = len(pool); bt = float(BT)
    x0, x1, y0, y1 = (float(b) for b in box)
    corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    cons = []; rhs = []
    for xc, yc in corners:                                   # cone: in-plane <= 0
        cons.append(np.append(G[:, 0] * xc + G[:, 1] * yc, 0.0)); rhs.append(0.0)
    for k in range(3):                                       # transverse inward >= c0req
        cons.append(np.append(G[:, 2 + k] * sigma[k], 0.0)); rhs.append(-float(c0req))
    g0 = dat['g0'][pool]; gl = dat['gl'][pool]; gq = dat['gq'][pool]
    for xc, yc in corners:                                   # 2nd-order over box x orthant
        for tg in (0.0, sigma[0] * bt):
            for tt1 in (0.0, sigma[1] * bt):
                for tt2 in (0.0, sigma[2] * bt):
                    p = np.array([xc, yc, tg, tt1, tt2])
                    gv = g0 + gl @ p + np.einsum('kij,i,j->k', gq, p, p)
                    cons.append(np.append(gv, -1.0)); rhs.append(0.0)
    Aeq = np.zeros((1, ne + 1)); Aeq[0, :ne] = 1
    cobj = np.zeros(ne + 1); cobj[-1] = 1.0
    r = linprog(cobj, A_ub=np.array(cons), b_ub=np.array(rhs), A_eq=Aeq, b_eq=[1],
                bounds=[(0, None)] * ne + [(None, None)], method='highs')
    if r.status != 0 or r.x[-1] > -2e-4: return None
    lam = r.x[:ne]
    keep = [k for k in range(ne) if lam[k] > 1e-9]
    tot = sum(lam[k] for k in keep)
    fr = [Fr(lam[k] / tot).limit_denominator(10**6) for k in keep]
    fr[-1] = 1 - sum(fr[:-1])
    if any(l < 0 for l in fr): return None
    return fr, [int(pool[k]) for k in keep]


def find_w1t(cfg, dats, box, tbox):
    """find_w1 with an ARBITRARY tau box (r78, M2 leaves): float combo-f LP over the
    corners of (in-plane box x tbox), constraints from all phi samples."""
    if isinstance(dats, dict): dats = [dats]
    n = len(dats[0]['drift'])
    (x0,x1,y0,y1) = box
    tp = [(a,b,cc) for a in (float(tbox[X.g][0]), float(tbox[X.g][1]))
          for b in (float(tbox[X.t1][0]), float(tbox[X.t1][1]))
          for cc in (float(tbox[X.t2][0]), float(tbox[X.t2][1]))]
    XC = np.array([[px,py,*t_] for px in (float(x0),float(x1)) for py in (float(y0),float(y1))
                   for t_ in tp])
    FV = np.vstack([dat['drift'][None,:] + XC@dat['G'].T for dat in dats])
    nv = n+1
    Aub = np.hstack([FV, -np.ones((len(FV),1))])
    A1 = np.zeros((1,nv)); A1[0,:n] = 1
    r = linprog(np.eye(nv)[-1], A_ub=Aub, b_ub=np.zeros(len(FV)),
                A_eq=A1, b_eq=[1], bounds=[(0,None)]*n+[(None,None)], method='highs')
    if r.status != 0 or r.x[-1] > -1e-4: return None
    lam = r.x[:n]
    keep = [k for k in range(n) if lam[k] > 1e-9]
    tot = sum(lam[k] for k in keep)
    fr = [Fr(lam[k]/tot).limit_denominator(10**6) for k in keep]
    fr[-1] = 1 - sum(fr[:-1])
    if any(l < 0 for l in fr): return None
    return fr, keep

def _w1_rec(cfg, dats, csbox, box, tbox, leaves, depth=0, maxdepth=4, gmax=None):
    """(r78, wit=M2) recursive transverse decomposition of one orthant tau-box; each
    leaf killed by wa2 (single row) or w12 (rational combo) -- both are the existing
    exact witnesses on (in-plane box) x (leaf tau box).  SOUND: leaves tile the orthant
    box (checked again independently by verify_sf), and each leaf cert proves the kill
    on its closed sub-box for all delta in (0,RHO0].  Fringe cart cells fail single-shot
    certs only because the exclusion varies over the TRANSVERSE; margins are healthy."""
    xbox = {X.xp1:(box[0],box[1]), X.xp2:(box[2],box[3]), **tbox}
    aabbf = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    cand = _wa_screen(cfg, csbox, aabbf, tbox)
    for i in cand[:4]:
        ok, info = X.wa2(cfg, i, csbox, xbox)
        if ok:
            leaves.append(dict(tbox=[[str(tbox[t][0]), str(tbox[t][1])] for t in _TAUS],
                               wit='A', row=int(i)))
            return True
    fw = find_w1t(cfg, dats, box, tbox)
    if fw:
        ok, info = X.w12(cfg, fw[0], fw[1], csbox, xbox)
        if ok:
            leaves.append(dict(tbox=[[str(tbox[t][0]), str(tbox[t][1])] for t in _TAUS],
                               wit='W1', lam=[str(l) for l in fw[0]],
                               rows=[int(k) for k in fw[1]]))
            return True
    if depth >= maxdepth: return False
    if gmax is None:
        gmax = [max(float(np.abs(dat['G'][:,2+k]).max()) for dat in dats) for k in range(3)]
    k = max(range(3), key=lambda q: (float(tbox[_TAUS[q]][1]) - float(tbox[_TAUS[q]][0])) * gmax[q])
    lo, hi = tbox[_TAUS[k]]
    if hi - lo <= 0: return False
    mid = (lo + hi) / 2
    for a, b in ((lo, mid), (mid, hi)):
        sub = dict(tbox); sub[_TAUS[k]] = (a, b)
        if not _w1_rec(cfg, dats, csbox, box, sub, leaves, depth+1, maxdepth, gmax):
            return False
    return True

# ---------------------------------------------------------------- per-cell certify
ORTH = list(itertools.product([-1,1],repeat=3))

def certify_cart(cfg, csbox, dats, box, dep=0):
    """returns (mode, data) or None."""
    xbox = {X.xp1:(box[0],box[1]), X.xp2:(box[2],box[3]),
            X.g:(-BT,BT), X.t1:(-BT,BT), X.t2:(-BT,BT)}
    fa = find_wa(cfg, dats, box)
    if fa and fa[1] < -1e-4:
        ok, info = X.wa2(cfg, fa[0], csbox, xbox)
        if ok: return 'A', dict(row=int(fa[0]), **info)
    supp = find_wb(cfg, dats[1], box)
    if supp:
        ok, info = X.wb2(cfg, supp, csbox, xbox)
        if ok: return 'B', dict(supp=list(supp), **{'br':info['br']})
    # W2 orthants (stress witness, W1 combo-f fallback per orthant)
    per = {}
    for sigma in ORTH:
        key = ''.join('+' if s>0 else '-' for s in sigma)
        got = False
        for c0req in C0REQS:
            f = find_w2(cfg, dats[1], ((float(box[0])+float(box[1]))/2,(float(box[2])+float(box[3]))/2),
                        sigma, c0req)
            if f is None: continue
            supp, acts = f
            ok, info = X.ww22(cfg, supp, sigma, acts, c0req, csbox, xbox)
            if ok:
                per[key] = dict(wit='S', supp=list(supp), acts=list(acts),
                                c0req=str(c0req), B0=info['B0'], B1=info['B1'])
                got = True; break
        if not got:
            sub = X.orth_subbox(xbox, sigma)
            fw = find_w1(cfg, dats, box, sigma)
            if fw:
                fr, keep = fw
                ok, info = X.w12(cfg, fr, keep, csbox, sub)
                if ok:
                    per[key] = dict(wit='W1', lam=[str(l) for l in fr],
                                    rows=[int(k) for k in keep], **info)
                    got = True
        if not got and dep >= 1:
            # r78 M2: transverse-recursive wa2/w12 leaves (fringe cart cells fail the
            # single-shot certs from TRANSVERSE variation; margins are healthy).
            tbox = {t: ((Fr(0), BT) if s > 0 else (-BT, Fr(0))) for t, s in zip(_TAUS, sigma)}
            leaves = []
            if _w1_rec(cfg, dats, csbox, box, tbox, leaves):
                per[key] = dict(wit='M2', leaves=leaves)
                got = True
        if not got:   # W-C cart cone-relaxed stress (hex1 near-corner cart cells; fast single-shot exact)
            for c0req in C0REQS:
                fc = find_wc(cfg, dats[1], box, sigma, c0req)
                if fc is None: continue
                lam_fr, rows = fc
                ok, info = X.wc2(cfg, lam_fr, rows, sigma, csbox, box)
                if ok:
                    per[key] = dict(wit='WC', lam=[str(l) for l in lam_fr],
                                    rows=[int(r) for r in rows], **{k2: v2 for k2, v2 in info.items()})
                    got = True; break
        if not got: return None
    return 'W2', dict(orthants=per)

_TAUS = (X.g, X.t1, X.t2)
_FLAM = {}
def _flam(cfg):
    """lambdified first-order forms f_i(c,s,xp1,xp2,g,t1,t2) + their transverse gradients (cached)."""
    if cfg not in _FLAM:
        import sympy as sp
        syms = (X.c, X.s, X.xp1, X.xp2, X.g, X.t1, X.t2)
        rows = X.SYM[cfg]['tight']
        fs = [sp.lambdify(syms, r['f'], 'math') for r in rows]
        gr = [[float(sp.diff(r['f'], t)) if sp.diff(r['f'], t).is_number else 1.0 for t in _TAUS] for r in rows]
        _FLAM[cfg] = (fs, gr)
    return _FLAM[cfg]

def _wa_screen(cfg, csbox, aabb, tbox):
    """float over-estimate of sup f_i over the box (c,s rectangle corners over-cover the arc — safe
    for a SCREEN, exact wa2 is the authority).  Returns rows sorted by sup, only those below a loose
    threshold (candidates for the exact single-row kill)."""
    fs, _ = _flam(cfg)
    cs = [(csbox[X.c][a], csbox[X.s][b]) for a in (0, 1) for b in (0, 1)]
    pts = [(float(cc), float(ss), float(x1), float(y1), float(gg), float(tt1), float(tt2))
           for (cc, ss) in cs for x1 in aabb[:2] for y1 in aabb[2:]
           for gg in tbox[X.g] for tt1 in tbox[X.t1] for tt2 in tbox[X.t2]]
    sup = [max(f(*p) for p in pts) for f in fs]
    return sorted([i for i in range(len(sup)) if sup[i] < 0.03], key=lambda i: sup[i])

def _mode1_kill(cfg, dat, csbox, aabb, tbox, depth=0, maxdepth=5):
    """(SF) MODE-1 two-mode transverse kill for one orthant sub-box (the piece certify_w3 was missing).
    Subdivide the transverse; each leaf killed by mode 1 = wa2 (single-row first-order kill, OUTER
    fringe where f_i<0 uniformly) or mode 2 = wb2/ww22 stress (tube INTERIOR).  In-plane (w,tau)
    subdivision comes from the sweep loop.  Float-pre-screened; splits the fringe coord."""
    xbox = {X.xp1:(aabb[0],aabb[1]), X.xp2:(aabb[2],aabb[3]), **tbox}
    cand = _wa_screen(cfg, csbox, aabb, tbox)        # mode 1 (cheap): exact-verify negative candidates
    for i in cand:
        if X.wa2(cfg, i, csbox, xbox)[0]: return True
    # mode 2 (EXPENSIVE stress) only on small INTERIOR sub-boxes near the transverse origin — the tube.
    # Skip it on the depth-0 orthant (w32 already failed it) and on outer-fringe boxes (mode-1's job).
    interior = all(min(abs(tbox[t][0]), abs(tbox[t][1])) < BT * Fr(1, 6) for t in _TAUS)
    if depth >= 1 and interior:
        supp = find_wb(cfg, dat, (aabb[0],aabb[1],aabb[2],aabb[3]))          # mode 2a: full-stress (Psi)
        if supp and X.wb2(cfg, supp, csbox, xbox)[0]: return True
        sigma = tuple(1 if tbox[t][0] >= 0 else -1 for t in _TAUS)           # mode 2b: orthant stress
        cx = (float(aabb[0])+float(aabb[1]))/2; cy = (float(aabb[2])+float(aabb[3]))/2
        for c0req in C0REQS:
            f = find_w2(cfg, dat, (cx, cy), sigma, c0req)
            if f is None: continue
            supp2, acts = f
            if X.ww22(cfg, supp2, sigma, acts, c0req, csbox, xbox)[0]: return True
    if depth >= maxdepth: return False
    # split the transverse coord the best mode-1 candidate is steepest in (the fringe direction)
    _, grad = _flam(cfg)
    best = cand[0] if cand else None
    if best is not None:
        tw = _TAUS[max(range(3), key=lambda k: abs(grad[best][k]) * (tbox[_TAUS[k]][1]-tbox[_TAUS[k]][0]))]
    else:
        tw = max(_TAUS, key=lambda t: tbox[t][1]-tbox[t][0])
    lo, hi = tbox[tw]; mid = (lo+hi)/2
    for a, b in ((lo, mid), (mid, hi)):
        sub = dict(tbox); sub[tw] = (a, b)
        if not _mode1_kill(cfg, dat, csbox, aabb, sub, depth+1, maxdepth): return False
    return True

def certify_w3(cfg, csbox, dats, corner, half, w0, w1, b0, b1):
    wm = float(w0+w1)/2
    th = 2*math.atan(wm)
    e = half*np.array([math.cos(th), math.sin(th)])
    per = {}
    # sector AABB (floats for the W1 finder; exact AABB rebuilt in w12 via xbox)
    import qinterval as QI
    ex, ey = X._ecomp(half)
    exi = QI.eval_iv(ex, {X.w: (w0, w1)}); eyi = QI.eval_iv(ey, {X.w: (w0, w1)})
    p0x = (Fr(0),Fr(0)); p0y = (Fr(0),Fr(0))
    if corner == 'far':
        p0x = QI.mul((Fr(2),Fr(2)), csbox[X.c]); p0y = QI.mul((Fr(2),Fr(2)), csbox[X.s])
    xb1 = QI.add(p0x, QI.mul((b0,b1), exi)); xb2 = QI.add(p0y, QI.mul((b0,b1), eyi))
    aabb = (xb1[0], xb1[1], xb2[0], xb2[1])
    for sigma in ORTH:
        key = ''.join('+' if s>0 else '-' for s in sigma)
        got = False
        for c0req in C0REQS:
            dat = dats[1]
            P0 = np.array([0.,0.])
            tmid = float(b0+b1)/2
            f = find_w2(cfg, dat, None, sigma, c0req,
                        radial=(corner, e, P0 if corner=='near' else dats[1]['P0'], tmid))
            if f is None: continue
            supp, acts = f
            ok, info = X.w32(cfg, supp, sigma, acts, c0req, csbox, corner, half,
                             (w0,w1), (b0,b1))
            if ok:
                per[key] = dict(wit='S', supp=list(supp), acts=list(acts), c0req=str(c0req),
                                B0=info['B0'], B1=info['B1'], A_hi=info['A_hi'])
                got = True; break
        if not got:
            xbox = {X.xp1:(aabb[0],aabb[1]), X.xp2:(aabb[2],aabb[3]),
                    X.g:(-BT,BT), X.t1:(-BT,BT), X.t2:(-BT,BT)}
            sub = X.orth_subbox(xbox, sigma)
            fw = find_w1(cfg, dats, aabb, sigma)
            if fw:
                fr, keep = fw
                ok, info = X.w12(cfg, fr, keep, csbox, sub)
                if ok:
                    per[key] = dict(wit='W1', lam=[str(l) for l in fr],
                                    rows=[int(k) for k in keep], **info)
                    got = True
        if not got:   # (SF) MODE-1: two-mode transverse kill (first-order fringe + stress interior)
            tbox = {t: ((Fr(0), BT) if s > 0 else (-BT, Fr(0))) for t, s in zip(_TAUS, sigma)}
            if _mode1_kill(cfg, dats[1], csbox, aabb, tbox):
                per[key] = dict(wit='M1'); got = True
        if not got and corner == 'near':   # W4 sector-cone mixed kill (r73, wall-edge)
            for c0req in C0REQS:
                f4 = find_w4(cfg, dats, sigma, w0, w1, b0, b1, c0req)
                if f4 is None: continue
                lam_fr, rws = f4
                ok, info = X.w42(cfg, lam_fr, rws, sigma, csbox, half, (w0, w1), (b0, b1))
                if ok:
                    per[key] = dict(wit='W4', lam=[str(l) for l in lam_fr],
                                    rows=[int(k) for k in rws], **{k2: v2 for k2, v2 in info.items()})
                    got = True; break
        if not got and corner == 'near':   # Lemma H homogeneous transverse Farkas (r76/77, marginal boundary)
            fh = find_h(cfg, dats, sigma, w0, w1, b0, b1)
            if fh:
                lam_fr, rws = fh
                ok, info = X.h2(cfg, lam_fr, rws, sigma, csbox, half, (w0, w1), (b0, b1))
                if ok:
                    per[key] = dict(wit='H', lam=[str(l) for l in lam_fr],
                                    rows=[int(k) for k in rws], **{k2: v2 for k2, v2 in info.items()})
                    got = True
        if not got:
            return None
    return 'W3', dict(orthants=per)


def _box_in_sector(box, half, w0, w1, b0, b1):
    """EXACT check: closed box (x0,x1)x(y0,y1) is contained in the sector
    {t*half*e(w): t in [b0,b1], w in [w0,w1]},  e(w) = ((1-w^2)/(1+w^2), 2w/(1+w^2)).
    (i) every corner u = half*(x,y) lies in the closed convex cone spanned by
        d(w0), d(w1), d(w) = (1-w^2, 2w)  [theta(w)=2 atan(w) is increasing; cone
        width < pi enforced via u.(d0+d1) > 0];
    (ii) max corner |.|^2 <= b1^2;  (iii) coordinate-clamp dist^2 from 0 >= b0^2."""
    x0, x1, y0, y1 = box
    d0 = (1 - w0*w0, 2*w0); d1 = (1 - w1*w1, 2*w1)
    for cx in (x0, x1):
        for cy in (y0, y1):
            ux, uy = half*cx, half*cy
            if d0[0]*uy - d0[1]*ux < 0: return False
            if d1[0]*uy - d1[1]*ux > 0: return False
            if ux*(d0[0]+d1[0]) + uy*(d0[1]+d1[1]) <= 0: return False
            if cx*cx + cy*cy > b1*b1: return False
    dx = max(x0, -x1, Fr(0)); dy = max(y0, -y1, Fr(0))
    if dx*dx + dy*dy < b0*b0: return False
    return True

def _enclose_sector(box):
    """rational sector (half,w0,w1,b0,b1) enclosing the box, verified by the exact
    predicate _box_in_sector; None if degenerate (origin too close / cone too wide)."""
    x0, x1, y0, y1 = box
    cx = (float(x0)+float(x1))/2; cy = (float(y0)+float(y1))/2
    th = math.atan2(cy, cx)
    half = 1 if abs(th) <= math.pi/2 else -1
    ws = []
    for ax in (float(x0), float(x1)):
        for ay in (float(y0), float(y1)):
            ux, uy = half*ax, half*ay
            ws.append(math.tan(math.atan2(uy, ux)/2))
    w0 = Fr(min(ws)).limit_denominator(10**5) - Fr(1, 10**5)
    w1 = Fr(max(ws)).limit_denominator(10**5) + Fr(1, 10**5)
    if max(abs(w0), abs(w1)) > Fr(13,10): return None
    dmax2 = max(a*a + b*b for a in (x0, x1) for b in (y0, y1))
    b1 = QI.sqrt_iv(dmax2, dmax2)[1]
    b1 = Fr(math.ceil(b1 * 10**6), 10**6)          # outward round (keep fractions small)
    dx = max(x0, -x1, Fr(0)); dy = max(y0, -y1, Fr(0))
    dmin2 = dx*dx + dy*dy
    if dmin2 < Fr(1, 10000): return None      # too close to the corner (exempt region)
    b0 = QI.sqrt_iv(dmin2, dmin2)[0]
    b0 = Fr(math.floor(b0 * 10**6), 10**6)         # outward round
    if not _box_in_sector((x0, x1, y0, y1), half, w0, w1, b0, b1): return None
    return half, w0, w1, b0, b1

# ---------------------------------------------------------------- phi-cell runner
def run_phicell(idx, cfg, pkind, lo, hi, out):
    t00 = time.time()
    w3e_stat = dict(fails=0)
    csbox = csbox_of(cfg, pkind, lo, hi)
    phis = phis_of(cfg, pkind, lo, hi)
    dats = [SF.EV[cfg].eval_at(p) for p in phis]
    for d, p in zip(dats, phis):
        d['P0'] = np.array([2*math.cos(p), 2*math.sin(p)])
    pc = [cfg, pkind, str(lo), str(hi)]
    cart, meta = gen_cart_cells(cfg, pkind, lo, hi)
    out.write(json.dumps(dict(kind='meta', phicell=pc, h0=str(H0), fr=str(FR114),
                              rb=str(RB), tlast=str(TLAST), C4='60', ft='LemmaFT-far', rf_far=str(RF_FAR), **meta)) + '\n')
    npass = nfail = 0
    # 16-gon exempt test (any-depth): cell fully inside an exempt corner 16-gon
    corners_mid = np.array([[0.,0.],[2*math.cos(phis[1]),2*math.sin(phis[1])]])
    cdrift = meta['cdrift']
    # r69: per-corner exemption radii.  Near corner (0): W3-near [RB,TLAST] + SB-box
    # cover disk(0, TLAST).  Far corner (2e(phi)): Lemma FT covers disk(2e(phi), RF_FAR)
    # at every phi -- exempt iff the cell stays within RF_FAR of the TRUE corner
    # (mid-corner distance + cdrift + slack), quantified by the apodized radius.
    apo_near = (float(TLAST) - 0.003) * math.cos(math.pi/16) - 0.002
    apo_far  = (float(RF_FAR) - cdrift - 0.003) * math.cos(math.pi/16) - 0.002
    apos = [apo_near, apo_far]
    def exempt(x0,x1,y0,y1):
        cc = np.array([[float(x0),float(y0)],[float(x0),float(y1)],
                       [float(x1),float(y0)],[float(x1),float(y1)]])
        for ci in range(2):
            if np.linalg.norm(cc - corners_mid[ci], axis=1).max() < apos[ci]:
                return True
        return False
    # Cartesian with quadtree
    stack = [(Fr(a),Fr(b),Fr(cc),Fr(dd),0) for (a,b,cc,dd) in
             [(x0,x1,y0,y1) for (x0,x1,y0,y1) in cart]]
    while stack:
        x0,x1,y0,y1,dep = stack.pop()
        if exempt(x0,x1,y0,y1): continue
        r = certify_cart(cfg, csbox, dats, (x0,x1,y0,y1), dep)
        if not r and dep >= 2 and (w3e_stat['fails'] < 25 or dep == MAXDEPTH):
            # r78 W3E: in-tube / near-corner-seam cart cells (first-order feasible points
            # inside => no cart cert can close them).  Enclose the box in an exact sector
            # and certify with the radial W3 chain (S/W1/M1/W4/H) -- box containment is
            # exact and re-checked independently at replay.
            se = _enclose_sector((x0, x1, y0, y1))
            if se:
                halfE, we0, we1, be0, be1 = se
                r3 = certify_w3(cfg, csbox, dats, 'near', halfE, we0, we1, be0, be1)
                if r3:
                    r = ('W3E', dict(half=int(halfE), w=[str(we0), str(we1)],
                                     t=[str(be0), str(be1)], **r3[1]))
                else:
                    w3e_stat['fails'] += 1
        if r:
            mode, data = r
            out.write(json.dumps(dict(kind='cell', phicell=pc,
                cell=dict(type='cart', box=[str(x0),str(x1),str(y0),str(y1)]),
                mode=mode, data=data)) + '\n')
            npass += 1
        elif dep < MAXDEPTH:
            xm, ym = (x0+x1)/2, (y0+y1)/2
            stack += [(x0,xm,y0,ym,dep+1),(xm,x1,y0,ym,dep+1),
                      (x0,xm,ym,y1,dep+1),(xm,x1,ym,y1,dep+1)]
        else:
            out.write(json.dumps(dict(kind='fail', phicell=pc,
                cell=dict(type='cart', box=[str(x0),str(x1),str(y0),str(y1)]))) + '\n')
            nfail += 1
    # W3 sectors with w-subdivision
    stack = [(c_,h_,w0,w1,b0,b1,0) for (c_,h_,w0,w1,b0,b1) in gen_w3_cells()]
    while stack:
        corner,half,w0,w1,b0,b1,dep = stack.pop()
        r = certify_w3(cfg, csbox, dats, corner, half, w0, w1, b0, b1)
        if r:
            mode, data = r
            out.write(json.dumps(dict(kind='cell', phicell=pc,
                cell=dict(type='w3', corner=corner, half=half,
                          w=[str(w0),str(w1)], t=[str(b0),str(b1)]),
                mode=mode, data=data)) + '\n')
            npass += 1
        elif dep < W3_DEPTH:
            # subdivide BOTH w AND the tau-band (4-way, adaptive).  The far-corner wall-straddle
            # sectors fail from interval decorrelation over tau, not a charge gap (triage: 72/72
            # resolve) -- the old w-only split left the tau-band un-refined.
            wm = (w0+w1)/2; bm = (b0+b1)/2
            stack += [(corner,half,a0,a1,c0,c1,dep+1)
                      for (a0,a1) in ((w0,wm),(wm,w1)) for (c0,c1) in ((b0,bm),(bm,b1))]
        else:
            out.write(json.dumps(dict(kind='fail', phicell=pc,
                cell=dict(type='w3', corner=corner, half=half,
                          w=[str(w0),str(w1)], t=[str(b0),str(b1)]))) + '\n')
            nfail += 1
    return npass, nfail, time.time()-t00

def run_shard(idx):
    cells = phi_cells()
    cfg, pkind, lo, hi = cells[idx]
    os.makedirs(SHARD_DIR, exist_ok=True)
    shard = os.path.join(SHARD_DIR, f'shard_{idx}.jsonl')
    done = shard.replace('.jsonl', '.done')
    if os.path.exists(done):
        return f'shard {idx}: already done'
    with open(shard, 'w') as out:
        npass, nfail, dt = run_phicell(idx, cfg, pkind, lo, hi, out)
    with open(done, 'w') as f:
        f.write(json.dumps(dict(npass=npass, nfail=nfail, secs=round(dt,1))))
    return f'shard {idx} ({cfg} [{lo},{hi}]): pass={npass} fail={nfail} {dt:.1f}s'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--cells', type=str, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--procs', type=int, default=1)
    args = ap.parse_args()
    cells = phi_cells()
    if args.list:
        for i, pc in enumerate(cells): print(i, pc)
        print(len(cells), 'phi-cells'); return
    if args.validate:
        idxs = [10, 45, 48, 62, 126]   # phi ~ 0.20, 1.03, wall-straddle, 2.90(pentB), 1.60(hex)
    elif args.cells:
        idxs = [int(x) for x in args.cells.split(',')]
    elif args.all:
        idxs = [i for i in range(len(cells))
                if not os.path.exists(os.path.join(SHARD_DIR, f'shard_{i}.done'))]
    else:
        ap.print_help(); return
    if args.procs > 1:
        import multiprocessing as mp
        with mp.Pool(args.procs) as pool:
            for msg in pool.imap_unordered(run_shard, idxs):
                print(msg, flush=True)
    else:
        for i in idxs:
            print(run_shard(i), flush=True)

if __name__ == '__main__':
    main()
