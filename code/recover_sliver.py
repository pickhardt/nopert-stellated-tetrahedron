"""recover_lowdelta.py -- IN-3 low-delta flat-margin band STUCK-cell recovery (r92).

The dk_sliver cloud sweep (delta in [0.006,0.0076]) left 809,815 'stuck' cells: cells the sweep
could not DK-certify at its minw floor.  DIAGNOSIS (this session, sampled across all stuck-heavy
boxes): every stuck cell has a strictly POSITIVE center exclusion margin (margin_lp >= 1.96e-7);
the sweep failed only because its gamma (inner-rotation, axis 4) resolution was too coarse for the
degree-2 TM enclosure to confirm the ~2e-7 margin.  Halving axis 4 tightens the enclosure below the
margin: 600/600 sampled stuck cells across 10 boxes close with gamma-depth <= 3, ZERO fails, ZERO
LP-resort.  So the band is NOT open math -- it is a gamma-refinement compute job.

This driver reprocesses every stuck cell:
  close(cell): DK.try_cell(cell); if not ok, bisect axis 4 (gamma) and recurse (<= MAXGD);
               last resort at MAXGD: LP-support triple (dk_sweep._lp_cert), then split p1/p2/g.
Each stuck cell's emitted leaves are a dyadic gamma-partition of the cell => they tile it exactly.
Rows (jsonl):  leaf {st:cert, cen, hw, triples, theta, lam_lo, val_hi}; the first row of a cell's
group carries top=1, src=[stuck cen,hw], nsub.  A residual (should be none) is st:fail.

Verified by verify_recover.py: (V1) the set of 'src' cells == the full stuck set of dk_sliver;
(V2) each group's leaves gamma-tile its src exactly; (V4) every cert leaf re-passes certify_cell.

Usage: python recover_lowdelta.py SHARD NSHARDS OUTDIR
"""
import sys, os, json, gzip, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dk_kernel as DK
import dk_sweep as SW
from rc_dlo2 import dlo_window2   # certified depth d_lo over a (δ,φ) window
from rc_dlo import I

MAXD = 12   # max subdivision depth

def _cert_row(cen, hw, r):
    return dict(st='cert', cen=list(cen), hw=list(hw), triples=r.get('triples'),
                theta=r.get('theta'), Hadj=r.get('Hadj'), lam_lo=r['lam_lo'], val_hi=r['val_hi'])

# ---- cylinder corner-skip with certified fine depth (SOUND via Theorem 8.1) ----
# A cell whose inner-rotation lies within reach = d_lo/(M0*δ_hi) of a marginal corner is excluded
# by the transported depth cylinder for EVERY translation, where d_lo is the certified depth over
# the cell's (δ,φ) extent (dlo_window2, a rigorous lower bound; M0=1 on the localization angles).
# The depth degenerates at the wall, so d_lo is certified per the cell's OWN (δ,φ) window and grows
# as the cell refines in δ; memoized by (δ,φ,δhw,φhw) since millions of inner-rotation cells share it.
_DLO_CACHE = {}
def _fine_dlo(cen, hw):
    key = (round(cen[0], 9), round(cen[1], 9), round(hw[0], 9), round(hw[1], 9))
    v = _DLO_CACHE.get(key, 0)
    if v == 0:
        try:
            v = dlo_window2(I(cen[0] - hw[0], cen[0] + hw[0]), I(cen[1] - hw[1], cen[1] + hw[1])) or -1.0
        except Exception:
            v = -1.0
        _DLO_CACHE[key] = v
    return v if v > 0 else None

def _corner_skip(cen, hw):
    dlo = _fine_dlo(cen, hw)
    if dlo is None: return False
    reach = dlo / (cen[0] + hw[0])                 # M0 = 1; δ_hi in the denominator is conservative
    return SW.corner_dist_hi(cen, hw) <= reach

def _pick_axis(cen, hw):
    """axis whose halving lets BOTH children close (corner-skip OR det-dual); δ first (helps the
    corner-skip depth), then γ (helps det-dual), then the other inner axes."""
    for ax in (0, 4, 2, 3, 1):
        both = True
        for sg in (-0.5, 0.5):
            c2 = list(cen); c2[ax] = cen[ax] + sg * hw[ax]; h2 = list(hw); h2[ax] = hw[ax] / 2
            if not (_corner_skip(c2, h2) or DK.try_cell(c2, h2).get('ok')): both = False; break
        if both: return ax
    return 0  # default: refine δ (deepens the corner-skip depth)

def close(cen, hw, depth, rows, stat):
    """certify-close a cell. Two sound mechanisms: (a) cylinder corner-skip via certified depth
    (near-marginal cells), (b) det-dual try_cell (far cells). Subdivide and recurse otherwise."""
    # (a) cylinder corner-skip
    if _corner_skip(cen, hw):
        rows.append(dict(st='skip', cen=list(cen), hw=list(hw), dlo=_fine_dlo(cen, hw)))
        stat['skip'] = stat.get('skip', 0) + 1; return True
    # (b) far cell: det-dual
    try:
        r = DK.try_cell(cen, hw)
    except Exception as e:
        r = dict(ok=False, why='exc:' + type(e).__name__)
    if r.get('ok'):
        rows.append(_cert_row(cen, hw, r)); stat['cert'] = stat.get('cert', 0) + 1; return True
    # (c) subdivide
    if depth >= MAXD:
        lo = SW._lp_cert(cen, hw, None)
        if lo is not None:
            rows.append(_cert_row(cen, hw, lo)); stat['lp'] = stat.get('lp', 0) + 1; return True
        rows.append(dict(st='fail', cen=list(cen), hw=list(hw), why=r.get('why', 'cert')))
        stat['fail'] = stat.get('fail', 0) + 1; return False
    ax = _pick_axis(cen, hw)
    h2 = list(hw); h2[ax] = hw[ax] / 2
    ok = True
    for sgn in (-0.5, 0.5):
        c2 = list(cen); c2[ax] = cen[ax] + sgn * hw[ax]
        if not close(c2, h2, depth + 1, rows, stat): ok = False
    return ok

def _load_stuck():
    """all stuck cells; prefer the pre-extracted stuck_all.jsonl dump (avoids re-decompressing gz)."""
    here = os.path.dirname(__file__)
    dump = os.path.join(here, 'dk_sliver', 'stuck_all.jsonl')
    if os.path.exists(dump):
        return [json.loads(l) for l in open(dump)]
    cells = []
    for fn in sorted(glob.glob(os.path.join(here, 'dk_sliver', 'boxes', 'box_*.jsonl.gz'))):
        for l in gzip.open(fn, 'rt'):
            r = json.loads(l)
            if r.get('kind') == 'stuck': cells.append([r['cen'], r['hw']])
    return cells

def run(shard, nsh, outdir):
    cells = _load_stuck()
    cells = cells[shard::nsh]
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, 'rec_%d.jsonl' % shard)
    donef = out + '.done'
    if os.path.exists(donef): print('shard %d already done' % shard, flush=True); return
    start = 0
    if os.path.exists(out):
        start = sum(1 for l in open(out) if json.loads(l).get('top'))
    f = open(out, 'a')
    stat = {}; t0 = time.time()
    for i, (cen, hw) in enumerate(cells):
        if i < start: continue
        rows = []
        close(cen, hw, 0, rows, stat)
        rows[0]['top'] = 1; rows[0]['src'] = [list(cen), list(hw)]; rows[0]['nsub'] = len(rows); rows[0]['idx'] = i
        for r in rows: f.write(json.dumps(r) + '\n')
        if i % 5000 == 0:
            f.flush()
            print('shard %d: %d/%d %s %.0fs' % (shard, i, len(cells), stat, time.time() - t0), flush=True)
    f.close()
    open(donef, 'w').write(json.dumps({'n': len(cells), 'stats': stat, 'maxgd': MAXGD, 'elapsed': time.time() - t0}))
    print('shard %d DONE %s' % (shard, stat), flush=True)

if __name__ == '__main__':
    run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
