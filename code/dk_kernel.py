"""dk_kernel.py -- Lemma DK (det-dual, translation-exact) interval cell certifier.

Cell: (delta, phi, p1h, p2h, gh) = center +- hw, 5 shared AA symbols.
Chart (r86): u2 = exp(delta [w(phi)]) u*,  w = -sin(phi) r1 + cos(phi) r2,
  W* rows r1=(0,0,1), r2=-(1,1,0)/sqrt2, r3=u*=(1,-1,0)/sqrt2;  W(u2) = W* Rw^T.
Inner rotation R1 = exp([xi]) W(u2), xi = delta*(-p2h, p1h, gh)  (tilt delta*ph, spin delta*gh).

CERTIFICATE (soundness = Lemma DK + hedging, r86 notes; independent of hull combinatorics):
  For triples T_k = [(a_i,b_i,j_i)]_{i=1..3} of (outer edge endpoints, inner vertex) and
  fixed weights theta_k >= 0, if over the WHOLE cell (AA enclosures):
    (V) validity:   n_i.(q_m - q_{a_i}) <= 0  for all outer vertices m  (n_i = outward normal
        of the segment q_{a_i} q_{b_i}; exact equality at m=a_i,b_i is analytic, skipped),
    (L) lam_1 = det(n_2,n_3) > 0, lam_2 = det(n_3,n_1) > 0, lam_3 = det(n_1,n_2) > 0,
    (K) H := sum_k theta_k G_k < 0,  G_k := sum_i lam_i * n_i.(q_{a_i} - y_{j_i}),
  then for EVERY parameter in the cell and EVERY t in R^2, weak containment
  pi(R1 V) + t subset K(u2) FAILS.
  Proof: containment => n_i.(y+t) <= c_i (validity) => c_i - n_i.y_j - n_i.t >= 0; multiply by
  lam_i >= 0, sum: G_k - (sum lam_i n_i).t >= 0; Cramer identity sum lam_i n_i = 0 EXACTLY
  (identically in the parameters, since lam are the 2x2 dets of the other two normals);
  so G_k >= 0 for each k, hence H >= 0 -- contradicting (K).  No translation bound enters.
"""
import math, itertools
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scipy.linalg import expm
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aan as _aan
import tm2 as _tm2
import fast_interval as F
from fast_interval import FI, rd, ru

SQ2 = math.sqrt(2.0)
US  = np.array([1.0, -1.0, 0.0])/SQ2
R1B = np.array([0.0, 0.0, 1.0])
R2B = np.array([-1.0, -1.0, 0.0])/SQ2
WSTAR = np.array([R1B, R2B, US])          # rows
A = 0.55
V4 = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
VERTS = np.vstack([V4, -A*V4])            # 8 x 3

# ----------------------------------------------------------------- float center geometry
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def geom_f(par):
    d, ph, p1, p2, g = par
    w = -math.sin(ph)*R1B + math.cos(ph)*R2B
    Rw = expm(d*skew(w))
    W = WSTAR @ Rw.T                       # frame at u2 (third row = u2)
    X = (W @ VERTS.T).T                    # 8x3 outer frame coords
    q = X[:, :2]
    xi = np.array([-d*p2, d*p1, d*g])
    Y = (expm(skew(xi)) @ X.T).T
    y = Y[:, :2]
    return q, y

def hull_edges(q):
    h = ConvexHull(q)
    vs = list(h.vertices)                  # ccw
    return [(vs[i], vs[(i+1) % len(vs)]) for i in range(len(vs))]

def n_c_of_edge(q, a, b):
    e = q[b] - q[a]
    n = np.array([e[1], -e[0]])            # outward for ccw polygon
    return n, n @ q[a]

def margin_lp(par):
    """true exclusion margin m = min_t max_{ij} (n_i.(y_j+t) - c_i); m>0 <=> excluded for all t.
    returns m, support list [(a,b,j,slackrate)] of pairs active at optimum."""
    q, y = geom_f(par)
    ed = hull_edges(q)
    rows, meta = [], []
    for (a, b) in ed:
        n, c = n_c_of_edge(q, a, b)
        for j in range(8):
            rows.append((n, n @ y[j] - c)); meta.append((a, b, j))
    # LP: min z  s.t. z >= v_ij + n_i.t  -> vars (z,t1,t2)
    Aub = np.array([[-1.0, n[0], n[1]] for n, v in rows])
    bub = np.array([-v for n, v in rows])
    res = linprog(c=[1.0, 0, 0], A_ub=Aub, b_ub=bub, bounds=[(None, None)]*3, method='highs')
    m = res.fun
    sup = []
    marg = res.ineqlin.marginals            # dual (<=0 convention)
    for k, mu in enumerate(marg):
        if abs(mu) > 1e-9:
            sup.append((meta[k], -mu))
    return m, sup, res.x[1:]

def G_of_triple_f(par, T):
    """float G_k at par for triple T=[(a,b,j)]*3; None if lambda sign fails."""
    q, y = geom_f(par)
    ns, cs = [], []
    for (a, b, j) in T:
        n, c = n_c_of_edge(q, a, b); ns.append(n); cs.append(c)
    lam = [ns[1][0]*ns[2][1]-ns[1][1]*ns[2][0],
           ns[2][0]*ns[0][1]-ns[2][1]*ns[0][0],
           ns[0][0]*ns[1][1]-ns[0][1]*ns[1][0]]
    if min(lam) <= 0: return None
    return sum(l*(c - n @ y[j]) for l, (n, c, (aa, bb, j)) in zip(lam, [(n, c, t) for n, c, t in zip(ns, cs, T)]))

# ----------------------------------------------------------------- AA cell geometry
def geom_aa(cen, hw, K=_tm2):
    """Taylor-model enclosures of q (8x2) and y (8x2) over the cell cen +- hw.
    K = kernel module (aan: degree-1; tm2: degree-2)."""
    AAN = K.TM2 if hasattr(K, 'TM2') else K.AAN
    rot_from_vec = K.rot_from_vec; matvec = K.matvec
    d  = AAN.sym(cen[0], 0, hw[0])
    ph = AAN.sym(cen[1], 1, hw[1])
    p1 = AAN.sym(cen[2], 2, hw[2])
    p2 = AAN.sym(cen[3], 3, hw[3])
    g  = AAN.sym(cen[4], 4, hw[4])
    s2 = AAN.from_FI(FI(rd(1.0/SQ2), ru(1.0/SQ2)))
    sph, cph = ph.sin(), ph.cos()
    # w = -sin(ph)*(0,0,1) + cos(ph)*(-s2,-s2,0)
    eta = [ -(cph*s2)*d, -(cph*s2)*d, -(sph)*d ]     # eta = d*w
    Rw = rot_from_vec(eta)
    # W = WSTAR @ Rw^T ; row i of W: W[i][j] = sum_k WSTAR[i][k]*Rw[j][k]
    ws = [[AAN.const(0.0), AAN.const(0.0), AAN.const(1.0)],
          [-s2, -s2, AAN.const(0.0)],
          [ s2, -s2, AAN.const(0.0)]]
    W = [[ws[i][0]*Rw[j][0] + ws[i][1]*Rw[j][1] + ws[i][2]*Rw[j][2] for j in range(3)]
         for i in range(3)]
    aF = AAN.from_FI(FI(rd(11.0/20.0), ru(11.0/20.0)))
    X = []
    for vi in range(8):
        if vi < 4: comp = [AAN.const(V4[vi][k]) for k in range(3)]
        else:      comp = [-(aF*V4[vi-4][k]) if V4[vi-4][k] >= 0 else (aF*(-V4[vi-4][k])) for k in range(3)]
        X.append(matvec(W, comp))
    q = [[X[j][0], X[j][1]] for j in range(8)]
    xi = [-(d*p2), d*p1, d*g]
    Rx = rot_from_vec(xi)
    y = []
    for j in range(8):
        Yj = matvec(Rx, X[j])
        y.append([Yj[0], Yj[1]])
    return q, y

def certify_cell(cen, hw, triples, theta, verbose=False, K=_tm2):
    """certify (V),(L),(K) over the cell.  Returns dict(ok, Hhi, lam_lo, val_hi)."""
    AAN = K.TM2 if hasattr(K, 'TM2') else K.AAN
    q, y = geom_aa(cen, hw, K=K)
    # cache halfplanes by edge
    hp = {}
    def halfplane(a, b):
        if (a, b) in hp: return hp[(a, b)]
        ex = q[b][0] - q[a][0]; ey = q[b][1] - q[a][1]
        n = [ey, -ex]                         # outward for ccw (a,b)
        # validity: n.(q_m - q_a) <= 0 for m != a,b
        vhi = -1e30
        for m in range(8):
            if m == a or m == b: continue
            s = n[0]*(q[m][0]-q[a][0]) + n[1]*(q[m][1]-q[a][1])
            vhi = max(vhi, s.hi())
        hp[(a, b)] = (n, vhi)
        return hp[(a, b)]
    H = AAN.const(0.0)
    lam_lo = 1e30; val_hi = -1e30; charge = 0.0
    for T, th in zip(triples, theta):
        if th == 0.0: continue
        ns, etas = [], []
        for (a, b, j) in T:
            n, vhi = halfplane(a, b)
            val_hi = max(val_hi, vhi)
            ns.append(n); etas.append(max(0.0, vhi))
        lam = [ns[1][0]*ns[2][1]-ns[1][1]*ns[2][0],
               ns[2][0]*ns[0][1]-ns[2][1]*ns[0][0],
               ns[0][0]*ns[1][1]-ns[0][1]*ns[1][0]]
        for L in lam: lam_lo = min(lam_lo, L.lo())
        Gk = AAN.const(0.0)
        for L, n, (a, b, j) in zip(lam, ns, T):
            diff = [q[a][0]-y[j][0], q[a][1]-y[j][1]]
            Gk = Gk + L*(n[0]*diff[0] + n[1]*diff[1])
        H = H + Gk*th
        # Lemma DK' (r90): offset charge lam_i*eta_i replaces the validity requirement.
        # eta_i >= max(0, sup_cell max_m n_i.(q_m - q_{a_i})) is a nonneg float constant;
        # lam_i >= 0 over the cell (lam_lo gate), so sup(lam_i*eta_i) <= lam_i.hi()*eta_i.
        for L, eta in zip(lam, etas):
            if eta > 0.0: charge += th * max(L.hi(), 0.0) * eta
    Hadj = H.hi() + charge          # float adds; charge inflated below for rounding
    Hadj += 1e-15*abs(Hadj) + 1e-300
    ok = (Hadj < 0.0) and (lam_lo > 0.0)
    out = dict(ok=ok, Hhi=H.hi(), Hadj=Hadj, charge=charge, Hlo=H.lo(),
               lam_lo=lam_lo, val_hi=val_hi)
    if verbose: print(out)
    return out

# ----------------------------------------------------------------- selection heuristics
def corners(cen, hw):
    cs = []
    for mask in range(32):
        cs.append([c + (1 if (mask >> k) & 1 else -1)*h for k, (c, h) in enumerate(zip(cen, hw))])
    return cs

def select_triples(cen, hw, maxtriples=10, nrand=150, seed=0):
    """pool = union of LP support pairs over center+corners; theta by maximin over
    corners + random interior points."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    pts = [list(cen)] + corners(cen, hw)
    pool = {}
    m0 = None
    for p in pts:
        try:
            m, sup, _ = margin_lp(p)
        except Exception:
            continue
        if m0 is None: m0 = m
        for (t, w) in sup:
            key = (int(t[0]), int(t[1]), int(t[2]))
            pool[key] = pool.get(key, 0.0) + w
    pairs = sorted(pool, key=lambda k: -pool[k])
    cands = []
    for T0 in itertools.combinations(pairs, 3):
        edges = {(a, b) for (a, b, j) in T0}
        if len(edges) < 3: continue
        for T in (list(T0), list(T0)[::-1]):     # both cyclic orientations
            gval = G_of_triple_f(cen, T)
            if gval is not None and gval < 0:
                cands.append((gval, T))
                break
    cands.sort()
    cands = cands[:maxtriples]
    if not cands: return m0, [], []
    Ts = [T for _, T in cands]
    cs = pts + [[c + (2*rng.random()-1)*h for c, h in zip(cen, hw)] for _ in range(nrand)]
    Gmat = _np.zeros((len(cs), len(Ts)))
    for ci, c in enumerate(cs):
        for ti, T in enumerate(Ts):
            gv = G_of_triple_f(c, T)
            Gmat[ci, ti] = -gv if gv is not None else -1e6
    nt = len(Ts)
    res = linprog(c=[0.0]*nt + [-1.0],
                  A_ub=np.hstack([-Gmat, np.ones((len(cs), 1))]), b_ub=np.zeros(len(cs)),
                  A_eq=[[1.0]*nt + [0.0]], b_eq=[1.0],
                  bounds=[(0, None)]*nt + [(None, None)], method='highs')
    theta = res.x[:nt]
    keep = [(T, th) for T, th in zip(Ts, theta) if th > 1e-12]
    return m0, [T for T, _ in keep], [th for _, th in keep]

def try_cell(cen, hw, verbose=False, K=_tm2):
    m, Ts, th = select_triples(cen, hw)
    if not Ts:
        return dict(ok=False, why='no-triple', m=m)
    out = certify_cell(cen, hw, Ts, th, verbose=verbose, K=K)
    out['m'] = m; out['ntriples'] = len(Ts)
    out['triples'] = Ts; out['theta'] = list(th)
    return out
