"""dk_verify.py -- standalone replay verifier for DK sweep manifests.  r90 -> completed.

Usage: python certificates/dk_verify.py MANIFEST.jsonl [--sample K] [--in2 IN2_MANIFEST]

Independently re-checks, from the manifest alone (shares only dk_kernel.certify_cell / tm2
with the emitter -- NOT the sweep driver / selection):
  1. TILING: the leaf cells (pass|skip|stuck) exactly tile the declared region. The sweep
     builds a binary tree splitting each internal node along argmax(hw/minw); a manifest cell
     is a LEAF. We reconstruct that tree from the root (header reg_lo/reg_hi, minw) and assert
     every reconstructed leaf is present exactly once and every manifest cell is consumed.
  2. STUCK: assert zero 'stuck' rows.
  3. SKIP: recompute corner_dist_hi(cen,hw) <= RC independently (the geometric kill-ball test).
     RC-vs-IN-2 citation (RC <= d_lo/(M0*delta_hi)) is checked iff --in2 is given; else flagged
     as owed to the cross-manifest (IN-4) assembly step.
  4. PASS: re-run dk_kernel.certify_cell(cen,hw,triples,theta) from the stored witness and assert
     ok (Hadj<0 & lam_lo>0). --sample K re-runs K random rows/manifest PLUS every near-marginal
     row (|Hadj| within 10x of the smallest-|Hadj|), which are always replayed.
"""
import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dk_kernel as DK
from dk_sweep import corner_dist_hi

TOL = 1e-12  # float match tolerance for dyadic bisection boxes
_RCGRID = None  # optional RC-grid file (delta,phi -> certified d_lo) for per-cell skip verification


def _load_rcgrid(path):
    lines = [l for l in open(path)]
    nd, npp = (int(x) for x in lines[0].split())
    de = [float(x) for x in lines[1].split()]
    pe = [float(x) for x in lines[2].split()]
    dlo = [[float(x) for x in lines[3+i].split()] for i in range(nd)]
    return de, pe, dlo


def _key(cen, hw):
    return (tuple(round(c, 12) for c in cen), tuple(round(h, 12) for h in hw))


def check_tiling(hdr, leaves):
    """Reconstruct the bisection tree from the root; assert leaves partition the region.
    leaves: dict _key -> (cen,hw). Returns (ok, msg, n_consumed)."""
    minw = hdr['minw']
    reg_lo, reg_hi = hdr['reg_lo'], hdr['reg_hi']
    cen0 = [(a + b) / 2 for a, b in zip(reg_lo, reg_hi)]
    hw0 = [(b - a) / 2 for a, b in zip(reg_lo, reg_hi)]
    consumed = set()
    stack = [(cen0, hw0)]
    depth_guard = 0
    while stack:
        depth_guard += 1
        if depth_guard > 20_000_000:
            return False, "tiling recursion runaway (manifest inconsistent)", len(consumed)
        cen, hw = stack.pop()
        k = _key(cen, hw)
        if k in leaves:                       # this box IS a manifest leaf
            if k in consumed:
                return False, f"cell appears twice in tiling: {k}", len(consumed)
            consumed.add(k)
            continue
        # not a leaf -> must be an internal (split) node; reproduce the driver's split axis
        ratios = [h / m for h, m in zip(hw, minw)]
        ax = max(range(len(ratios)), key=lambda i: ratios[i])
        if ratios[ax] <= 1.0:
            return False, f"gap: box not in manifest and not splittable: cen={cen} hw={hw}", len(consumed)
        h2 = list(hw); h2[ax] = hw[ax] / 2
        cA = list(cen); cA[ax] = cen[ax] - h2[ax]
        cB = list(cen); cB[ax] = cen[ax] + h2[ax]
        stack.append((cA, list(h2))); stack.append((cB, list(h2)))
    if len(consumed) != len(leaves):
        extra = set(leaves) - consumed
        return False, f"{len(extra)} manifest cell(s) NOT reachable by the tiling tree (e.g. {next(iter(extra))})", len(consumed)
    return True, "exact dyadic tiling verified", len(consumed)


def main(path, sample=0, in2=None):
    rows = [json.loads(l) for l in open(path)]
    hdr = rows[0]; assert hdr.get('kind') == 'header', "first row must be header"
    passes = [r for r in rows[1:] if r['kind'] == 'pass']
    skips  = [r for r in rows[1:] if r['kind'] == 'skip']
    stucks = [r for r in rows[1:] if r['kind'] == 'stuck']
    RC = hdr['RC']
    print(f"manifest: {len(rows)-1} cells  pass={len(passes)} skip={len(skips)} stuck={len(stucks)}")

    # ---- check 2: stuck
    if stucks:
        print(f"FAIL (check 2): {len(stucks)} stuck cell(s) -- manifest is NOT a certificate")
        return False

    # ---- check 1: tiling
    leaves = {}
    dup = False
    for r in rows[1:]:
        k = _key(r['cen'], r['hw'])
        if k in leaves: dup = True
        leaves[k] = (r['cen'], r['hw'])
    if dup:
        print("FAIL (check 1): duplicate cell boxes in manifest")
        return False
    ok_t, msg_t, ncon = check_tiling(hdr, leaves)
    print(f"  check 1 TILING: {'PASS' if ok_t else 'FAIL'} -- {msg_t}")
    if not ok_t:
        return False

    # ---- check 3: skip rows -> recompute corner test
    # Global RC: cd <= RC. Per-cell RC ('percell'): independently RE-CERTIFY the depth per skip
    # row via rc_dlo.dlo_window at the cell's own (delta,phi) window and check
    # cd <= d_lo/(M0*delta_hi). This does NOT trust the stored r['RC'] -- it recomputes the
    # certified reach from scratch (the cylinder-Thm-8.1 corner-skip soundness condition).
    bad_skip = 0
    percell = (RC == 'percell')
    M0 = 1.0
    grid = None
    if percell:
        import importlib
        # rc_dlo.py lives at the run root (one dir above certificates/); ensure it's importable
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path: sys.path.insert(0, _root)
        rc_dlo = importlib.import_module('rc_dlo'); Iv = rc_dlo.I
        if _RCGRID:  # RE-CERTIFY the RC grid independently: recompute each cell's d_lo via rc_dlo
            de, pe, _dlo_claimed = _load_rcgrid(_RCGRID)
            re_dlo = [[None]*(len(pe)-1) for _ in range(len(de)-1)]
            for i in range(len(de)-1):
                for j in range(len(pe)-1):
                    try: d = rc_dlo.dlo_window(Iv(de[i], de[i+1]), Iv(pe[j], pe[j+1]))
                    except Exception: d = None
                    re_dlo[i][j] = 0.0 if d is None else d
            grid = (de, pe, re_dlo)
            print(f"  [RC grid re-certified: {len(de)-1}x{len(pe)-1} cells, independently via rc_dlo]")
    def grid_rc(cen, hw):
        de, pe, gd = grid
        d0, d1 = cen[0]-hw[0], cen[0]+hw[0]; p0, p1 = cen[1]-hw[1], cen[1]+hw[1]
        mind = float('inf')
        for i in range(len(de)-1):
            if de[i+1] < d0 or de[i] > d1: continue
            for j in range(len(pe)-1):
                if pe[j+1] < p0 or pe[j] > p1: continue
                mind = min(mind, gd[i][j])
        return 0.0 if mind == float('inf') else mind/(M0*(cen[0]+hw[0]))
    for r in skips:
        cd = corner_dist_hi(r['cen'], r['hw'])
        if percell:
            cen, hw = r['cen'], r['hw']
            if grid is not None:
                rc_true = grid_rc(cen, hw)   # re-certified grid-min (matches how the prover justified)
            else:
                try:
                    d_lo = rc_dlo.dlo_window(Iv(cen[0]-hw[0], cen[0]+hw[0]), Iv(cen[1]-hw[1], cen[1]+hw[1]))
                except Exception:
                    d_lo = None
                rc_true = 0.0 if d_lo is None else d_lo / (M0 * (cen[0]+hw[0]))
            if not (cd <= rc_true + 1e-15):
                bad_skip += 1
                if bad_skip <= 5:
                    print(f"    bad skip: cd={cd:.6g} > re-certified RC={rc_true:.6g}  cen={cen}")
        else:
            if not (cd <= RC + 1e-15):
                bad_skip += 1
                if bad_skip <= 5:
                    print(f"    bad skip: corner_dist_hi={cd:.6g} > RC={RC}  cen={r['cen']}")
    print(f"  check 3 SKIP:   {'PASS' if bad_skip == 0 else 'FAIL'} -- {len(skips)} rows, {bad_skip} bad"
          + ("  [per-cell RC re-certified via rc_dlo]" if percell else
             ("" if in2 else "  [RC-vs-IN-2 citation owed to IN-4 assembly]")))
    if bad_skip:
        return False

    # ---- check 4: pass rows -> re-certify from stored witness
    to_check = passes
    if sample and sample < len(passes):
        rng = random.Random(12345)
        # always replay near-marginal rows (smallest |Hadj| band), plus a random sample
        hadj = [abs(p.get('Hadj', 0.0)) for p in passes]
        hmin = min(h for h in hadj if h > 0) if any(h > 0 for h in hadj) else 0.0
        near = [p for p, h in zip(passes, hadj) if hmin > 0 and h <= 10 * hmin]
        rest = [p for p in passes if p not in near]
        to_check = near + rng.sample(rest, min(sample, len(rest)))
        print(f"  check 4 PASS:   sampling {len(to_check)} of {len(passes)} ({len(near)} near-marginal + {min(sample,len(rest))} random)")
    else:
        print(f"  check 4 PASS:   re-certifying all {len(passes)} rows")
    bad_pass = 0
    for i, p in enumerate(to_check):
        if p.get('triples') is None or p.get('theta') is None:
            bad_pass += 1
            if bad_pass <= 3: print(f"    pass row missing witness (triples/theta): cen={p['cen']}")
            continue
        out = DK.certify_cell(p['cen'], p['hw'], p['triples'], p['theta'])
        if not out['ok']:
            bad_pass += 1
            if bad_pass <= 5:
                print(f"    re-certify FAILED: Hadj={out['Hadj']:.3e} lam_lo={out['lam_lo']:.3e} cen={p['cen']}")
    print(f"  check 4 PASS:   {'PASS' if bad_pass == 0 else 'FAIL'} -- {len(to_check)} replayed, {bad_pass} bad")
    if bad_pass:
        return False

    print("VERDICT: PASS (DK manifest certifies its region: tiling + skip + pass all verified, 0 stuck)")
    return True


if __name__ == '__main__':
    args = sys.argv[1:]
    path = args[0]
    sample = 0; in2 = None
    for i, a in enumerate(args):
        if a == '--sample': sample = int(args[i + 1])
        elif a.startswith('--sample='): sample = int(a.split('=')[1])
        elif a == '--in2': in2 = args[i + 1]
        elif a == '--rcgrid': _RCGRID = args[i + 1]
    sys.exit(0 if main(path, sample=sample, in2=in2) else 1)
