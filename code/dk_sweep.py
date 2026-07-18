"""dk_sweep.py -- adaptive DK det-dual sweep driver (rule-3 tube domain).  r90.

Disposition of a cell C = cen +- hw in (delta, phi, p1h, p2h, gh):
  1. CORNER-SKIP (sound): C is discarded iff for EVERY parameter of C the rescaled
     inner-rotation point (p1h,p2h,gh) lies in the certified kill ball of radius RC
     around a marginal corner:  diagonal corner (0,0,0), or sheet corner
     (2cos phi, 2sin phi, 0) (Lemma ST, proved r90: sheet marginal = exp([2 delta e(phi)])
     exactly; Cor 10.4 transports the diagonal cylinder ball to it exactly).
     Test: sup over C of dist-to-corner <= RC, with the phi-motion of the sheet corner
     bounded by 2*hw_phi (|d/dphi 2e(phi)| = 2), all outward-rounded.
     RC must be justified per phi-cell: RC <= d_lo(u2)/(M0*delta_hi) for all (delta,phi)
     in C (cited from the IN-2 depth grid; M0=1 for angles <= 0.4641).
  2. DK certify: dk_kernel.try_cell (Lemma DK + hedging, TM2 degree-2, outward-rounded).
  3. SPLIT along the axis with largest hw/minw ratio (> 1); if no axis splittable,
     record STUCK (= FAIL: needs operator attention; sweep is only complete when
     stuck list is empty).

Manifest rows (jsonl): {kind: pass|skip, cen, hw, and for pass: triples, theta, Hhi,
lam_lo, val_hi}.  A standalone verifier re-checks pass rows via dk_kernel.certify_cell
and skip rows via the corner test + the cited RC, and re-checks the exact dyadic tiling.
"""
import math, json, time, heapq
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dk_kernel as DK

def _lp_cert(cen, hw, K):
    """Fallback certifier for the wall-dip flat-margin band: read the triple off the true LP margin's
    support (margin_lp) and rigorously re-certify at this cell. Sound — certify_cell re-checks (V),(L),(K);
    the LP only PROPOSES the (non-trusted) triple. Returns the pass-dict or None."""
    if K is None: return None
    try:
        m, sup, _ = DK.margin_lp(cen)
    except Exception:
        return None
    if m <= 0 or len(sup) < 3: return None
    T = [(int(a), int(b), int(j)) for ((a, b, j), _r) in sup[:3]]
    for th in ([1.,1,1],[1.,-1,1],[1,1,-1.],[-1.,1,1],[1.,-1,-1],[-1.,-1,1],[1.,1,1.],[-1.,-1,-1.]):
        try:
            o = DK.certify_cell(cen, hw, [T], th, K=K)
            if o.get('ok'):
                o['triples'] = [T]; o['theta'] = th; return o
        except Exception:
            pass
    return None

# ---------------------------------------------------------------- corner skip (interval)
def corner_dist_hi(cen, hw):
    """rigorous upper bound of max over the cell of min(dist to diagonal corner,
    dist to sheet corner (2cos phi, 2sin phi, 0)).  Conservative float+eps arithmetic."""
    EPS = 1e-12
    p1c, p2c, gc = cen[2], cen[3], cen[4]
    h1, h2, hg = hw[2], hw[3], hw[4]
    # diagonal corner: per-axis max |coord|
    d2 = (abs(p1c)+h1)**2 + (abs(p2c)+h2)**2 + (abs(gc)+hg)**2
    dd = math.sqrt(d2)*(1+EPS)
    # sheet corner: center at phi center, moves by <= 2*hw_phi over the cell
    cphi, sphi = math.cos(cen[1]), math.sin(cen[1])
    s2 = (abs(p1c-2*cphi)+h1)**2 + (abs(p2c-2*sphi)+h2)**2 + (abs(gc)+hg)**2
    ds = math.sqrt(s2)*(1+EPS) + 2*hw[1]
    return min(dd, ds)

# ---------------------------------------------------------------- adaptive sweep
def sweep(reg_lo, reg_hi, RC, minw, out=None, tmax=None, K=None, verbose=False):
    """reg_lo/hi: 5-vectors; RC: corner-ball radius (must be cited/certified for the
    whole region's phi,delta range); minw: per-axis minimum half-widths.
    Returns (npass, nskip, stuck list).  Writes manifest rows to `out` (file handle)."""
    t0 = time.time()
    # RC may be a float (global corner-skip radius) OR a callable RC(cen,hw) returning a
    # certified per-cell radius (r91: RC(C) = d_lo(C's delta,phi window)/(M0*delta_hi),
    # cited from rc_dlo's proof-grade two-case depth certifier -- strictly sounder than a
    # single global min, which is dominated by the razor-thin wall-dip).
    rc_fn = RC if callable(RC) else (lambda cen, hw: RC)
    cen0 = [(a+b)/2 for a, b in zip(reg_lo, reg_hi)]
    hw0  = [(b-a)/2 for a, b in zip(reg_lo, reg_hi)]
    if out: out.write(json.dumps(dict(kind='header', reg_lo=list(reg_lo),
                 reg_hi=list(reg_hi), RC=('percell' if callable(RC) else RC), minw=list(minw)))+"\n")
    # stack items: (cen, hw, parent_triples, parent_theta). Triple-pool caching: a child first tries
    # its parent's selected triples (re-verified rigorously by certify_cell on the child's smaller cell,
    # where the enclosures are tighter, so they often now certify) BEFORE paying for a full re-selection
    # (33 LPs). Sound: triples are a non-trusted witness; certify_cell always re-checks (V),(L),(K).
    kw = dict(K=K) if K else {}
    stack = [(cen0, hw0, None, None)]
    npass = nskip = ncell = nreuse = 0
    stuck = []
    while stack:
        if tmax and time.time()-t0 > tmax:
            stuck += [(c, h, None) for c, h, _, _ in stack]   # unprocessed = stuck (time guard, demo only)
            break
        cen, hw, ptri, pth = stack.pop()
        ncell += 1
        rc = rc_fn(cen, hw)
        if corner_dist_hi(cen, hw) <= rc:
            nskip += 1
            if out: out.write(json.dumps(dict(kind='skip', cen=cen, hw=hw, RC=rc))+"\n")
            continue
        r = None
        # 1. cheap path: re-certify with the parent's triples (no selection LPs)
        if ptri:
            try:
                o = DK.certify_cell(cen, hw, ptri, pth, **kw)
                if o.get('ok'):
                    o['triples'] = ptri; o['theta'] = pth; r = o; nreuse += 1
            except Exception:
                r = None
        # 2. full selection (only if reuse missed)
        if r is None:
            try:
                r = DK.try_cell(cen, hw, **kw)
            except Exception as e:        # domain guard / LP failure = NOT certified -> split
                r = dict(ok=False, why='exc:%s' % type(e).__name__)
        if r.get('ok'):
            npass += 1
            if out: out.write(json.dumps(dict(kind='pass', cen=cen, hw=hw,
                     triples=r.get('triples'), theta=r.get('theta'),
                     Hadj=r.get('Hadj'), lam_lo=r['lam_lo'], val_hi=r['val_hi']))+"\n")
            continue
        # split along most-splittable axis; pass this cell's selected triples down to the children
        ratios = [h/m for h, m in zip(hw, minw)]
        ax = int(np.argmax(ratios))
        if ratios[ax] <= 1.0:
            # LAST-RESORT LP-triple fallback (wall-dip flat-margin band): the geometric selector can
            # miss the certifying triple where the true exclusion margin is ~1e-6. Read the triple off
            # the true LP margin's support and re-certify at this (now minw-fine) cell. Sound: certify_cell
            # rigorously re-checks (V),(L),(K); the LP only proposes the (non-trusted) triple.
            lo = _lp_cert(cen, hw, kw.get('K'))
            if lo is not None:
                npass += 1
                if out: out.write(json.dumps(dict(kind='pass', cen=cen, hw=hw,
                         triples=lo['triples'], theta=lo['theta'], Hadj=lo.get('Hadj'),
                         lam_lo=lo['lam_lo'], val_hi=lo['val_hi']))+"\n")
                continue
            stuck.append((cen, hw, r))
            if out: out.write(json.dumps(dict(kind='stuck', cen=cen, hw=hw,
                     why=r.get('why','cert')))+"\n")
            continue
        ctri, cth = r.get('triples'), r.get('theta')   # candidate triples for children (may certify smaller cells)
        h2 = list(hw); h2[ax] = hw[ax]/2
        cA = list(cen); cA[ax] = cen[ax]-h2[ax]
        cB = list(cen); cB[ax] = cen[ax]+h2[ax]
        stack.append((cA, h2, ctri, cth)); stack.append((cB, h2, ctri, cth))
    return npass, nskip, stuck, ncell
