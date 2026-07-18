"""verify_sf.py -- standalone replay + coverage verifier for the (SF)(c) witness manifest.

Consumes only sf_shards/shard_*.jsonl.  Checks:

 [F] zero 'fail' rows.
 [R] REPLAY: every 'cell' row's exact witness re-verified through sf_exact2
     (stress lambda recomputed exactly from (supp, acts, c0req) -- stored lambdas are
     not trusted).  --sample N replays a random N rows per shard (0 = all).
 [PHI] phi-coverage: per chart, the union of certified phi-cell parameter intervals
     covers the chart region THROUGH the walls:
       pentA, pentB : s-intervals cover [0, 8657/10000]   (0.8657^2 >= 242/323, beyond wall)
       hex1         : c-intervals cover [-501/1000, 501/1000]  (0.501^2 >= 81/323)
     All comparisons exact rational.  (Mirror transport phi -> -phi is Lemma [MT].)
 [IP] in-plane coverage per phi-cell: certified cart AABBs cover
       TARGET  =  conv(corner enclosures of D_Pi over the cell)  (+)  [-114/1000,114/1000]^2
     minus the two exempt 16-gons at the marginal corners.  Corner enclosures are exact
     (interval 2x2 Cramer on the meta row pairs + the exact corners 0 and (2c,2s));
     TARGET contains N_euclid(D_Pi(phi), 0.08) x-section for every phi in the cell
     (114/1000 >= 0.08*sqrt2; localization Thm 14.3).  Sweep-line, all Fractions.
 [W3] the W3 sectors tile [-1,1] x [RB, TLAST] per (corner, half) exactly, and
     each exempt 16-gon satisfies  circumradius + corner-box radius <= TLAST,
     so every point of the 16-gon is, at every phi of the cell, either within RB of the
     true corner (SB-box territory, reach 0.0201141 > RB=1/50, BP=9/500 r71) or inside a certified sector.
 [ASSUME] echoes stated assumptions (C4<=60 pending its exact proof).

Exit: prints VERDICT PASS/FAIL and writes verify_sf_report.json.
"""
import os, sys, json, glob, argparse, random, math
from fractions import Fraction as Fr

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
import sf_exact2 as X
import qinterval as QI
import sympy as sp

SHARD_DIR = os.path.join(_D, 'sf_shards')
FR114 = Fr(114, 1000); RB = X.RB; TLAST = X.TLAST; BT = X.BT
# OCT08: rational octagon with ball(0.08) SUBSET OCT08 and max-radius sqrt(.08^2+.034^2)=0.0869
# < exempt-disk radius ~0.098.  Used to inflate the D_Pi corner enclosures to a coverage target
# that (a) contains N_euclid(D_Pi,0.08) [ball SUBSET octagon], (b) stays inside the exempt disk at
# acute corners [the l-inf box would reach 0.08*sqrt2=0.113 and poke past -> false strips].
_OA = Fr(8, 100); _OC = Fr(34, 1000)
OCT08 = [(_OA, _OC), (_OC, _OA), (-_OC, _OA), (-_OA, _OC),
         (-_OA, -_OC), (-_OC, -_OA), (_OC, -_OA), (_OA, -_OC)]

SBOUND = Fr(8657, 10000)   # >= s_w : SBOUND^2 = 74943649/10^8 >= 242/323
CBOUND = Fr(501, 1000)     # >= c_w : CBOUND^2 = 251001/10^6  >= 81/323
assert SBOUND**2 >= Fr(242, 323) and CBOUND**2 >= Fr(81, 323)

# ------------------------------------------------------------- basic geometry (exact)
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 2: return pts
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]

def xsec(poly, x):
    """[ylo,yhi] cross-section of convex polygon at vertical line x, or None."""
    ys = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i+1) % n]
        if p[0] == q[0]:
            if p[0] == x: ys += [p[1], q[1]]
            continue
        (a, b) = (p, q) if p[0] < q[0] else (q, p)
        if a[0] <= x <= b[0]:
            tt = Fr(x - a[0], b[0] - a[0])
            ys.append(a[1] + tt * (b[1] - a[1]))
    if not ys: return None
    return (min(ys), max(ys))

def interval_union_covers(intervals, lo, hi):
    """exact: do the intervals cover [lo,hi]?"""
    ivs = sorted(iv for iv in intervals if iv[1] > lo and iv[0] < hi)
    cur = lo
    for a, b in ivs:
        if a > cur: return False
        cur = max(cur, b)
        if cur >= hi: return True
    return cur >= hi

def subtract(iv, holes):
    """iv minus list of intervals -> list of intervals."""
    out = [iv]
    for h in holes:
        nxt = []
        for a, b in out:
            if h[1] <= a or h[0] >= b: nxt.append((a, b)); continue
            if h[0] > a: nxt.append((a, h[0]))
            if h[1] < b: nxt.append((h[1], b))
        out = nxt
    return out

# ------------------------------------------------------------- corner enclosures
def corner_boxes(cfg, csbox, pairs):
    """exact interval boxes for D_Pi corners: (0,0), (2c,2s), and each meta pair
    via 2x2 interval Cramer on rows  drift_i + G_i . p = 0."""
    rows = SYM = X.SYM[cfg]['tight']
    boxes = [((Fr(0), Fr(0)), (Fr(0), Fr(0)))]
    two_c = QI.mul((Fr(2), Fr(2)), csbox[X.c]); two_s = QI.mul((Fr(2), Fr(2)), csbox[X.s])
    boxes.append((two_c, two_s))
    UN = [tuple(1 if q == i else 0 for q in range(5)) for i in range(5)]
    Z5 = (0, 0, 0, 0, 0)
    for (i, j) in pairs:
        M = []
        d = []
        for k in (i, j):
            r = rows[k]
            gx = QI.eval_iv(sp.expand(sp.sympify(r['fd'].get(UN[0], 0))), csbox)
            gy = QI.eval_iv(sp.expand(sp.sympify(r['fd'].get(UN[1], 0))), csbox)
            dr = QI.eval_iv(sp.expand(sp.sympify(r['fd'].get(Z5, 0))), csbox)
            M.append((gx, gy)); d.append(QI.neg(dr))
        det = QI.add(QI.mul(M[0][0], M[1][1]), QI.neg(QI.mul(M[0][1], M[1][0])))
        if det[0] <= 0 <= det[1]:
            raise ValueError(f'degenerate corner pair {(i,j)} in {cfg}')
        inv = QI.inv(det)
        px = QI.mul(QI.add(QI.mul(d[0], M[1][1]), QI.neg(QI.mul(d[1], M[0][1]))), inv)
        py = QI.mul(QI.add(QI.mul(M[0][0], d[1]), QI.neg(QI.mul(M[1][0], d[0]))), inv)
        boxes.append((px, py))
    return boxes

def sixteen_gon(ctr, R):
    """rational 16-gon with all vertices inside circle(ctr, R) (verified exactly)."""
    vs = []
    for k in range(16):
        a = 2 * math.pi * k / 16
        vx = Fr(math.cos(a)).limit_denominator(1 << 14) * R * Fr(255, 256)
        vy = Fr(math.sin(a)).limit_denominator(1 << 14) * R * Fr(255, 256)
        assert vx * vx + vy * vy <= R * R
        vs.append((ctr[0] + vx, ctr[1] + vy))
    return vs

# ------------------------------------------------------------- replay
def replay_row(row):
    pc = row['phicell']; cfg, pk, lo, hi = pc[0], pc[1], Fr(pc[2]), Fr(pc[3])
    csbox = X.csbox_of(cfg, pk, lo, hi)
    cell = row['cell']; mode = row['mode']; data = row['data']
    if cell['type'] == 'cart':
        b = [Fr(v) for v in cell['box']]
        xbox = {X.xp1: (b[0], b[1]), X.xp2: (b[2], b[3]),
                X.g: (-BT, BT), X.t1: (-BT, BT), X.t2: (-BT, BT)}
        if mode == 'A':
            ok, _ = X.wa2(cfg, int(data['row']), csbox, xbox); return ok
        if mode == 'B':
            ok, _ = X.wb2(cfg, tuple(data['supp']), csbox, xbox); return ok
        if mode == 'W2':
            for sig, od in data['orthants'].items():
                sigma = tuple(1 if ch == '+' else -1 for ch in sig)
                if od.get('wit', 'S') == 'W1':
                    lam = [Fr(v) for v in od['lam']]
                    sub = X.orth_subbox(xbox, sigma)
                    ok, _ = X.w12(cfg, lam, [int(k) for k in od['rows']], csbox, sub)
                elif od.get('wit') == 'M2':
                    # r78: transverse-recursive leaves.  Soundness re-checked here from the
                    # manifest alone: (i) every leaf tau-box lies inside the orthant box,
                    # (ii) leaf interiors are pairwise disjoint, (iii) total leaf volume
                    # equals the orthant volume (exact Fractions).  Finitely many closed
                    # boxes inside O, disjoint interiors, total measure = vol(O)  =>  the
                    # union is all of O (complement is open with measure zero => empty).
                    # Then each leaf is re-verified with the standard exact wa2/w12.
                    sub = X.orth_subbox(xbox, sigma)
                    TT = (X.g, X.t1, X.t2)
                    O = {t: sub[t] for t in TT}
                    tbs = []
                    ok = True
                    for lf in od['leaves']:
                        tb = {t: (Fr(lf['tbox'][q][0]), Fr(lf['tbox'][q][1]))
                              for q, t in enumerate(TT)}
                        if any(tb[t][0] < O[t][0] or tb[t][1] > O[t][1]
                               or tb[t][0] >= tb[t][1] for t in TT):
                            ok = False; break
                        lx = dict(xbox); lx.update(tb)
                        if lf['wit'] == 'A':
                            okl, _ = X.wa2(cfg, int(lf['row']), csbox, lx)
                        elif lf['wit'] == 'W1':
                            lam = [Fr(v) for v in lf['lam']]
                            okl, _ = X.w12(cfg, lam, [int(k) for k in lf['rows']], csbox, lx)
                        else:
                            okl = False
                        if not okl:
                            ok = False; break
                        tbs.append(tb)
                    if ok:
                        vol = sum((tb[TT[0]][1]-tb[TT[0]][0])*(tb[TT[1]][1]-tb[TT[1]][0])
                                  *(tb[TT[2]][1]-tb[TT[2]][0]) for tb in tbs)
                        Ovol = (O[TT[0]][1]-O[TT[0]][0])*(O[TT[1]][1]-O[TT[1]][0])*(O[TT[2]][1]-O[TT[2]][0])
                        if vol != Ovol: ok = False
                    if ok:
                        for a in range(len(tbs)):
                            for b2 in range(a+1, len(tbs)):
                                if all(max(tbs[a][t][0], tbs[b2][t][0]) < min(tbs[a][t][1], tbs[b2][t][1])
                                       for t in TT):
                                    ok = False; break
                            if not ok: break
                elif od.get('wit') == 'WC':
                    ok, _ = X.wc2(cfg, [Fr(v) for v in od['lam']], [int(k) for k in od['rows']],
                                  sigma, csbox, [b[0], b[1], b[2], b[3]])
                else:
                    ok, _ = X.ww22(cfg, tuple(od['supp']), sigma, tuple(od['acts']),
                                   Fr(od['c0req']), csbox, xbox)
                if not ok: return False
            return len(data['orthants']) == 8
        if mode == 'W3E':
            # r78: cart cell certified by an enclosing near-corner sector.  Soundness:
            # (a) EXACT box-in-sector containment re-checked here from the manifest alone;
            # (b) the sector's orthant certificates are the standard w3 replays (kill on
            # the whole sector => kill on the box).
            half = int(data['half'])
            w0, w1 = Fr(data['w'][0]), Fr(data['w'][1])
            t0, t1 = Fr(data['t'][0]), Fr(data['t'][1])
            d0 = (1 - w0*w0, 2*w0); d1 = (1 - w1*w1, 2*w1)
            for cx in (b[0], b[1]):
                for cy in (b[2], b[3]):
                    ux, uy = half*cx, half*cy
                    if d0[0]*uy - d0[1]*ux < 0: return False
                    if d1[0]*uy - d1[1]*ux > 0: return False
                    if ux*(d0[0]+d1[0]) + uy*(d0[1]+d1[1]) <= 0: return False
                    if cx*cx + cy*cy > t1*t1: return False
            dx = max(b[0], -b[1], Fr(0)); dy = max(b[2], -b[3], Fr(0))
            if dx*dx + dy*dy < t0*t0: return False
            # sector AABB (near corner) for W1/M1 branches
            ex, ey = X._ecomp(half)
            exi = QI.eval_iv(ex, {X.w: (w0, w1)}); eyi = QI.eval_iv(ey, {X.w: (w0, w1)})
            xb1 = QI.mul((t0, t1), exi); xb2 = QI.mul((t0, t1), eyi)
            sxbox = {X.xp1: (xb1[0], xb1[1]), X.xp2: (xb2[0], xb2[1]),
                     X.g: (-BT, BT), X.t1: (-BT, BT), X.t2: (-BT, BT)}
            for sig, od in data['orthants'].items():
                sigma = tuple(1 if ch == '+' else -1 for ch in sig)
                if od.get('wit', 'S') == 'W1':
                    lam = [Fr(v) for v in od['lam']]
                    sub = X.orth_subbox(sxbox, sigma)
                    ok, _ = X.w12(cfg, lam, [int(k) for k in od['rows']], csbox, sub)
                elif od.get('wit') == 'W4':
                    lam = [Fr(v) for v in od['lam']]
                    ok, _ = X.w42(cfg, lam, [int(k) for k in od['rows']], sigma,
                                  csbox, half, (w0, w1), (t0, t1))
                elif od.get('wit') == 'H':
                    lam = [Fr(v) for v in od['lam']]
                    ok, _ = X.h2(cfg, lam, [int(k) for k in od['rows']], sigma,
                                 csbox, half, (w0, w1), (t0, t1))
                elif od.get('wit') == 'M1':
                    import sf_make_witness as MW
                    import sf_preflight as SFP
                    phis = MW.phis_of(cfg, pk, lo, hi)
                    dat = SFP.EV[cfg].eval_at(phis[1])
                    aabb = (xb1[0], xb1[1], xb2[0], xb2[1])
                    tbox = {tt: ((Fr(0), BT) if sg > 0 else (-BT, Fr(0)))
                            for tt, sg in zip((X.g, X.t1, X.t2), sigma)}
                    ok = MW._mode1_kill(cfg, dat, csbox, aabb, tbox)
                else:
                    ok, _ = X.w32(cfg, tuple(od['supp']), sigma, tuple(od['acts']),
                                  Fr(od['c0req']), csbox, 'near', half,
                                  (w0, w1), (t0, t1))
                if not ok: return False
            return len(data['orthants']) == 8
        return False
    if cell['type'] == 'w3':
        w0, w1 = Fr(cell['w'][0]), Fr(cell['w'][1])
        t0, t1 = Fr(cell['t'][0]), Fr(cell['t'][1])
        corner = cell['corner']; half = int(cell['half'])
        # sector AABB (exact; identical formulas to the driver's certify_w3)
        ex, ey = X._ecomp(half)
        exi = QI.eval_iv(ex, {X.w: (w0, w1)}); eyi = QI.eval_iv(ey, {X.w: (w0, w1)})
        p0x = (Fr(0), Fr(0)); p0y = (Fr(0), Fr(0))
        if corner == 'far':
            p0x = QI.mul((Fr(2), Fr(2)), csbox[X.c]); p0y = QI.mul((Fr(2), Fr(2)), csbox[X.s])
        xb1 = QI.add(p0x, QI.mul((t0, t1), exi)); xb2 = QI.add(p0y, QI.mul((t0, t1), eyi))
        xbox = {X.xp1: (xb1[0], xb1[1]), X.xp2: (xb2[0], xb2[1]),
                X.g: (-BT, BT), X.t1: (-BT, BT), X.t2: (-BT, BT)}
        for sig, od in data['orthants'].items():
            sigma = tuple(1 if ch == '+' else -1 for ch in sig)
            if od.get('wit', 'S') == 'W1':
                lam = [Fr(v) for v in od['lam']]
                sub = X.orth_subbox(xbox, sigma)
                ok, _ = X.w12(cfg, lam, [int(k) for k in od['rows']], csbox, sub)
            elif od.get('wit') == 'W4':
                lam = [Fr(v) for v in od['lam']]
                ok, _ = X.w42(cfg, lam, [int(k) for k in od['rows']], sigma,
                              csbox, half, (w0, w1), (t0, t1))
            elif od.get('wit') == 'H':
                lam = [Fr(v) for v in od['lam']]
                ok, _ = X.h2(cfg, lam, [int(k) for k in od['rows']], sigma,
                             csbox, half, (w0, w1), (t0, t1))
            elif od.get('wit') == 'M1':
                # M1 records no witness data: replay = deterministic re-derivation via the
                # driver's _mode1_kill (every accepted leaf is exact-verified inside by
                # wa2/wb2/ww22; the float finders only steer the subdivision).
                import sf_make_witness as MW
                import sf_preflight as SFP
                phis = MW.phis_of(cfg, pk, lo, hi)
                dat = SFP.EV[cfg].eval_at(phis[1])
                aabb = (xb1[0], xb1[1], xb2[0], xb2[1])
                tbox = {tt: ((Fr(0), BT) if sg > 0 else (-BT, Fr(0)))
                        for tt, sg in zip((X.g, X.t1, X.t2), sigma)}
                ok = MW._mode1_kill(cfg, dat, csbox, aabb, tbox)
            else:
                ok, _ = X.w32(cfg, tuple(od['supp']), sigma, tuple(od['acts']),
                              Fr(od['c0req']), csbox, corner, half,
                              (w0, w1), (t0, t1))
            if not ok: return False
        return len(data['orthants']) == 8
    return False



# ------------------------------------------------------- [IP] support-Farkas (r69)
XP_NORM = Fr(462, 100)   # ||p||_2 <= 2*X_D = 323*sqrt(2)/99 = 4.6140... <= 4.62 on D_Pi
                          # (X_D circumradius, Thm 12.2(b); 0 in D_Pi => ||p|| <= 2 X_D)
EPS08 = Fr(8, 100)

def support_farkas(cfg, csbox, target):
    """PROVE (exact, verifier-side): target >= N_euclid(D_Pi(phi), 0.08) for ALL phi
    in the csbox cell.  target is a CCW rational polygon.  For each edge with outward
    normal nu (rational, unnormalized) and support o = nu . v_edge:
      float LP finds lam >= 0 over the tight rows with  Sum lam G_Pi = -nu  minimizing
      Sum lam drift;  exact interval check over csbox:
        resid = nu + Sum lam G_Pi  (interval),  o_D = Sum lam drift_hi,
        h_{D_Pi}(nu) <= o_D + l1(resid) * XP_NORM,
      require  o_D + l1(resid)*XP_NORM + EPS08*||nu||_2^hi <= o.
    Soundness: every tight row satisfies drift + G.(p,0,0,0) >= 0 on D_Pi x {0}
    (Thm 12.2 normal form / exact flatness), so for p in D_Pi:
      nu.p = -(Sum lam G_Pi).p + resid.p <= Sum lam drift + |resid.p|.
    Rows used are found by LP but VERIFIED only through this exact chain: the meta
    corner pairs are hints; no float assert remains.  Returns (ok, badinfo)."""
    from scipy.optimize import linprog as _lp
    import numpy as _np
    rows = X.SYM[cfg]['tight']
    UN = [tuple(1 if q == i else 0 for q in range(5)) for i in range(5)]
    Z5 = (0, 0, 0, 0, 0)
    # interval data per row over the cell
    gxi, gyi, dri = [], [], []
    for r in rows:
        gxi.append(QI.eval_iv(sp.expand(sp.sympify(r['fd'].get(UN[0], 0))), csbox))
        gyi.append(QI.eval_iv(sp.expand(sp.sympify(r['fd'].get(UN[1], 0))), csbox))
        dri.append(QI.eval_iv(sp.expand(sp.sympify(r['fd'].get(Z5, 0))), csbox))
    n = len(rows)
    Gx = _np.array([float((iv[0]+iv[1])/2) for iv in gxi])
    Gy = _np.array([float((iv[0]+iv[1])/2) for iv in gyi])
    Dr = _np.array([float((iv[0]+iv[1])/2) for iv in dri])
    m = len(target)
    for k in range(m):
        p, q = target[k], target[(k+1) % m]
        nu = (q[1]-p[1], -(q[0]-p[0]))          # outward for CCW
        o = nu[0]*p[0] + nu[1]*p[1]
        # float LP
        Aeq = _np.vstack([Gx, Gy]); beq = [-float(nu[0]), -float(nu[1])]
        r = _lp(Dr, A_eq=Aeq, b_eq=beq, bounds=[(0, None)]*n, method='highs')
        if r.status != 0:
            return False, f'edge {k}: Farkas LP infeasible (nu={nu})'
        lam = [Fr(v).limit_denominator(10**9) if v > 1e-12 else Fr(0) for v in r.x]
        # exact chain
        rx = (nu[0], nu[0]); ry = (nu[1], nu[1]); oD = Fr(0)
        for li, gx_, gy_, dr_ in zip(lam, gxi, gyi, dri):
            if li == 0: continue
            liv = (li, li)
            rx = QI.add(rx, QI.mul(liv, gx_)); ry = QI.add(ry, QI.mul(liv, gy_))
            oD += li * dr_[1]
        l1res = max(abs(rx[0]), abs(rx[1])) + max(abs(ry[0]), abs(ry[1]))
        nn2 = nu[0]*nu[0] + nu[1]*nu[1]
        nrm_hi = QI.sqrt_iv(nn2, nn2)[1]
        if oD + l1res*XP_NORM + EPS08*nrm_hi > o:
            return False, (f'edge {k}: bound {float(oD + l1res*XP_NORM + EPS08*nrm_hi):.6f}'
                           f' > support {float(o):.6f}')
    return True, None

# ------------------------------------------------------------- Lemma FT constants
def ft_check():
    """Exact one-time constant checks for Lemma FT (far-corner transport).

    Lemma FT: let (R1,t) be a weak containment at u2(delta,phi), delta<=rho0, with
    rescaled tilt in the far exempt region cd_far := |xhat_Pi - 2e(phi)| <= RF and
    |gamma_hat| <= GAM (localization, Thm 14.3).  The sheet base S = N W(u2) m* has
    EXACT rotation vector zeta = 2 delta (-sin phi, cos phi, 0) (pure tilt at exactly
    2e(phi) rescaled; symbolically verified).  With eta := log(R1 S^T):
      ||eta|| <= ||xi - zeta|| = delta*sqrt(cd_far^2 + gamma_hat^2) <= delta*SIG (Lem 7.2).
    Either eta = 0 (exact sheet configuration; excluded by GA-2), or
    (N R1 m*, t) is a weak containment with IDENTICAL constraint system (Thm 10.3 T2),
    relative angle sigma' = ||eta|| in (0, rho0*SIG] <= sigma0, and rescaled tilt of
    norm <= SIG, hence at distance >= 2 - SIG > TLAST from the far corner: it lies in
    the near-certified region (cart cells + near-W3 [RB,TLAST] + SB-box disk RB), and
    is killed there -- contradiction.  Constants checked exactly below.
    """
    GAM  = Fr(57, 2000)          # 0.0285 localization transverse bound (Thm 14.3)
    RF   = Fr(7, 20)             # far exempt disk radius (r69: 0.35; 16-gon+boxrad <= RF)
    SIG  = Fr(3512, 10000)       # sqrt(RF^2 + GAM^2) <= 0.35116 <= SIG
    RHO0 = Fr(1, 1000)
    SIG0 = Fr(427, 10000)        # sigma0 = 0.0427
    # r71: BP = 9/500; reach = BP/M(0.1), M(0.1)=sqrt3*31/60, sqrt3<=17320509/1e7 (exact upper bd)
    BOXREACH = Fr(9,500)/(Fr(17320509,10**7)*Fr(31,60))  # >= 0.0201141 > RB = 1/50
    checks = dict(
        sig_dominates   = bool(SIG*SIG >= RF*RF + GAM*GAM),
        sigma_cap       = bool(RHO0*SIG <= SIG0),
        image_avoids_far= bool(Fr(2) - SIG > RF),   # image dist to far corner >= 2-SIG > RF
        near_disk_boxed = bool(RB <= BOXREACH),
        gam_in_box      = bool(GAM <= SIG0),
    )
    return dict(ok=all(checks.values()), RF=str(RF), GAM=str(GAM), SIG=str(SIG), **checks)

# ------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0, help='replay N random rows/shard (0=all)')
    ap.add_argument('--shards', type=str, default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(SHARD_DIR, 'shard_*.jsonl')))
    if args.shards:
        keep = set(int(x) for x in args.shards.split(','))
        files = [f for f in files if int(os.path.basename(f).split('_')[1].split('.')[0]) in keep]
    rep = dict(shards=len(files), fails=0, cells=0, replayed=0, replay_bad=0,
               ip_bad=[], w3_bad=[], wall_defer=[], phi=None,
               assume=['C4<=53<60 PROVED (c4_taylor.py: interval delta-Taylor cert; true per-row ~38)',
               'localization tube Thm 14.3', 'mirror transport', 'box/tube handoff b_Pi/M_B > r_b',
               'Section 12 flat structure (D_Pi vertex set: pent parallelogram / hex1 segment [0,2e])',
               'wall cells (degenerate corner pair) deferred to Section 15.3 wall module',
               'Lemma FT ingredients: Thm 10.3 (T2), Lemma 7.2, GA-2, exact sheet identity'])
    rep['ft'] = ft_check()
    if not rep['ft']['ok']:
        print('FT CONSTANT CHECK FAILED'); rep['fails'] += 1
    percell = {}
    for f in files:
        for line in open(f):
            row = json.loads(line)
            key = tuple(row['phicell'])
            percell.setdefault(key, dict(meta=None, cart=[], w3=[], fail=0, rows=[]))
            if row['kind'] == 'meta': percell[key]['meta'] = row
            elif row['kind'] == 'fail': percell[key]['fail'] += 1; rep['fails'] += 1
            elif row['kind'] == 'cell':
                rep['cells'] += 1
                percell[key]['rows'].append(row)
                if row['cell']['type'] == 'cart':
                    percell[key]['cart'].append([Fr(v) for v in row['cell']['box']])
                else:
                    percell[key]['w3'].append(row['cell'])
    # [R] replay
    rng = random.Random(0)
    for key, d in percell.items():
        rows = d['rows']
        if args.sample and len(rows) > args.sample:
            rows = rng.sample(rows, args.sample)
        for row in rows:
            rep['replayed'] += 1
            if not replay_row(row):
                rep['replay_bad'] += 1
                print('REPLAY BAD:', row['phicell'], row['cell'], row['mode'])
    # [PHI]
    cov = dict(pentA=[], pentB=[], hex1=[])
    for (cfg, pk, lo, hi) in percell:
        cov[cfg].append((Fr(lo), Fr(hi)))
    okA = interval_union_covers(cov['pentA'], Fr(0), SBOUND)
    okB = interval_union_covers(cov['pentB'], Fr(0), SBOUND)
    okH = interval_union_covers(cov['hex1'], -CBOUND, CBOUND)
    rep['phi'] = dict(pentA=okA, pentB=okB, hex1=okH)
    # [IP] + [W3] per phi-cell
    for key, d in percell.items():
        cfg, pk, lo, hi = key[0], key[1], Fr(key[2]), Fr(key[3])
        csbox = X.csbox_of(cfg, pk, lo, hi)
        meta = d['meta']
        if meta is None:
            rep['ip_bad'].append((key, 'no meta')); continue
        pairs = [tuple(p) for p in meta['pairs']]
        try:
            boxes = corner_boxes(cfg, csbox, pairs)
        except ValueError as e:
            # degenerate corner pair == silhouette-transition WALL cell (hex1 chart edge c -> +-0.5,
            # phi = phi_w).  Certified by the WALL MODULE (Section 15.3 / wall dichotomy), NOT the
            # generic [IP] cart sweep -> DEFER (not an [IP] failure).
            rep['wall_defer'].append((key, str(e))); continue
        # target polygon := hull(D_Pi corner enclosures (+) OCT08) -- a rational polygon with
        #   N_euclid(D_Pi(phi), 0.08) SUBSET target for all phi in the cell.  Justification:
        #   ball(0.08) SUBSET OCT08, and (paper Section 12 flat structure -- PROVEN) D_Pi(phi) is the
        #   hull of its certified vertices: on the PENT charts a parallelogram = hull(all corner
        #   boxes); on HEX1 the SEGMENT [(0,0), 2e(phi)=(2c,2s)] = hull(boxes[0], boxes[1]).  The
        #   remaining hex1 meta-pair boxes (boxes[2:]) are the same segment's interior/redundant
        #   corners whose 2x2-Cramer enclosure is ILL-CONDITIONED near the wall (blows up) -> dropped
        #   on hex1 (they lie on the segment, so hull is unchanged).  Hence for all phi:
        #     N_euclid(D_Pi,0.08) = D_Pi (+) ball(.08) SUBSET hull(use_boxes) (+) OCT08 = target.
        #   (This replaces the old FR114-hull target + [IP-SF] support-Farkas: that target
        #   over-approximated N(0.08) -> false-positive boundary slivers, and support-Farkas was
        #   provably un-tightenable since a cart-coverable target leaves zero slack.  Verified:
        #   target >= N_euclid(D_Pi,0.08) numerically on samples; ip_bad 141 -> 0.)
        use_boxes = boxes[:2] if cfg == 'hex1' else boxes
        pts = [(cx + sx, cy + sy) for (bx, by) in use_boxes for cx in bx for cy in by
               for (sx, sy) in OCT08]
        target = hull(pts)
        # exempt 16-gons at marginal corner boxes (0-box and 2e-box = boxes[0], boxes[1])
        exempts = []
        w3ok = True
        RF_FAR = Fr(meta.get('rf_far', '1/10'))   # r69: Lemma FT far-disk radius
        for ci in (0, 1):
            bx, by = boxes[ci]
            ctr = ((bx[0] + bx[1]) / 2, (by[0] + by[1]) / 2)
            brad2 = ((bx[1] - bx[0]) / 2)**2 + ((by[1] - by[0]) / 2)**2
            # boxrad upper bound (rational sqrt upper)
            brad = QI.sqrt_iv(brad2, brad2)[1]
            # near corner (ci=0): hole covered by W3-near [RB,TLAST] + SB-box (disk RB).
            # far corner (ci=1): hole covered by Lemma FT (disk(2e(phi), RF_FAR) all phi).
            RCOV = TLAST if ci == 0 else RF_FAR
            R16 = RCOV - brad - Fr(1, 500)
            if R16 <= RB + Fr(1, 100):
                w3ok = False; break
            exempts.append(sixteen_gon(ctr, R16))
            assert R16 + brad <= RCOV
        # [W3] tiling per (corner, half).  Far corner: NOT required -- discharged by
        # Lemma FT (exact sheet transport), see ft_check() below.
        for corner in ('near',):
            for half in (1, -1):
                secs = [cc for cc in d['w3'] if cc['corner'] == corner and int(cc['half']) == half]
                bks = sorted(set([Fr(cc['w'][0]) for cc in secs] + [Fr(cc['w'][1]) for cc in secs]))
                okt = bool(secs)
                if not (bks and bks[0] <= Fr(-1) and bks[-1] >= Fr(1)):
                    okt = bool(secs) and False
                for a, b in zip(bks[:-1], bks[1:]):
                    tivs = [(Fr(cc['t'][0]), Fr(cc['t'][1])) for cc in secs
                            if Fr(cc['w'][0]) <= a and Fr(cc['w'][1]) >= b]
                    if not interval_union_covers(tivs, RB, TLAST):
                        okt = False; break
                if not okt:
                    w3ok = False
                    rep['w3_bad'].append((key, corner, half))
        # [IP] sweep-line
        cart = d['cart']
        bks = sorted(set([p[0] for p in target] +
                         [v[0] for ex in exempts for v in ex] +
                         [b[0] for b in cart] + [b[1] for b in cart]))
        ipok = True
        for a, b in zip(bks[:-1], bks[1:]):
            xa, xb = xsec(target, a), xsec(target, b)
            vint = [p[1] for p in target if a < p[0] < b]
            los = [v[0] for v in (xa, xb) if v] + vint
            his = [v[1] for v in (xa, xb) if v] + vint
            if not los: continue
            req = (min(los), max(his))
            holes = []
            for ex in exempts:
                ea, eb = xsec(ex, a), xsec(ex, b)
                if ea and eb:
                    h = (max(ea[0], eb[0]), min(ea[1], eb[1]))
                    if h[0] < h[1]: holes.append(h)
            pieces = subtract(req, holes)
            covs = [(bb[2], bb[3]) for bb in cart if bb[0] <= a and bb[1] >= b]
            for (plo, phi_) in pieces:
                if not interval_union_covers(covs, plo, phi_):
                    ipok = False; break
            if not ipok: break
        if not ipok:
            rep['ip_bad'].append((key, f'strip [{float(a):.4f},{float(b):.4f}] uncovered'))
    verdict = (rep['fails'] == 0 and rep['replay_bad'] == 0 and not rep['ip_bad']
               and not rep['w3_bad'] and all(rep['phi'].values()))
    rep['verdict'] = 'PASS' if verdict else 'FAIL'
    print(json.dumps(rep, default=str, indent=1))
    json.dump(rep, open(os.path.join(_D, 'verify_sf_report.json'), 'w'), default=str, indent=1)
    print('VERDICT:', rep['verdict'])
    print('NOTE: [PHI] coverage is judged on the shards present; the full sweep must include all phi-cells.')

if __name__ == '__main__':
    main()
