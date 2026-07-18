"""recover_lowdelta.py -- IN-3 low-delta flat-margin band STUCK-cell recovery (r92).

The dk_full cloud sweep (delta in [0.006,0.0076]) left 809,815 'stuck' cells: cells the sweep
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

Verified by verify_recover.py: (V1) the set of 'src' cells == the full stuck set of dk_full;
(V2) each group's leaves gamma-tile its src exactly; (V4) every cert leaf re-passes certify_cell.

Usage: python recover_lowdelta.py SHARD NSHARDS OUTDIR
"""
import sys, os, json, gzip, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dk_kernel as DK
import dk_sweep as SW

MAXGD = 6   # max gamma-bisection depth (sampled worst case was 3; headroom)

def _cert_row(cen, hw, r):
    return dict(st='cert', cen=list(cen), hw=list(hw), triples=r.get('triples'),
                theta=r.get('theta'), Hadj=r.get('Hadj'), lam_lo=r['lam_lo'], val_hi=r['val_hi'])

_SCALE = [2e-2, 3e-3, 1e-2, 1e-2, 1e-2]  # per-axis nominal widths for widest-relative fallback

def _best_axis(cen, hw):
    """axis whose single halving immediately certifies BOTH children (enclosure-tightest direction);
    else the widest-relative axis. gamma (4) is tried first since it closes the vast majority."""
    for ax in (4, 2, 3, 1, 0):
        both = True
        for sg in (-0.5, 0.5):
            c2 = list(cen); c2[ax] = cen[ax] + sg * hw[ax]; h2 = list(hw); h2[ax] = hw[ax] / 2
            if not DK.try_cell(c2, h2).get('ok'): both = False; break
        if both: return ax
    return max(range(5), key=lambda i: hw[i] / _SCALE[i])

def close(cen, hw, depth, rows, stat):
    """certify-close a cell; append cert (or fail) leaf rows; return True iff fully closed.
    gamma (axis 4) is the enclosure-tight direction for the bulk (closes at depth<=3); the deep
    wall phi_w core (~phi 1.039) needs a p1/phi split instead, so after gamma-exhaustion we fall
    back to the ADAPTIVE best-axis split (the axis that certifies both children), then LP."""
    try:
        r = DK.try_cell(cen, hw)
    except Exception as e:
        r = dict(ok=False, why='exc:' + type(e).__name__)
    if r.get('ok'):
        rows.append(_cert_row(cen, hw, r)); stat['cert'] = stat.get('cert', 0) + 1; return True
    if depth < MAXGD:
        ax = 4  # gamma-first (cheap; closes the bulk)
    elif depth < MAXGD + 8:
        ax = _best_axis(cen, hw)  # wall-core: adaptive best axis (p1/phi) after gamma is exhausted
    else:
        lo = SW._lp_cert(cen, hw, None)
        if lo is not None:
            rows.append(_cert_row(cen, hw, lo)); stat['lp'] = stat.get('lp', 0) + 1; return True
        rows.append(dict(st='fail', cen=list(cen), hw=list(hw), why=r.get('why', 'cert')))
        stat['fail'] = stat.get('fail', 0) + 1; return False
    h2 = list(hw); h2[ax] = hw[ax] / 2
    ok = True
    for sgn in (-0.5, 0.5):
        c2 = list(cen); c2[ax] = cen[ax] + sgn * hw[ax]
        if not close(c2, h2, depth + 1, rows, stat): ok = False
    return ok

def _load_stuck():
    """all stuck cells; prefer the pre-extracted stuck_all.jsonl dump (avoids re-decompressing gz)."""
    here = os.path.dirname(__file__)
    dump = os.path.join(here, 'dk_full', 'stuck_all.jsonl')
    if os.path.exists(dump):
        return [json.loads(l) for l in open(dump)]
    cells = []
    for fn in sorted(glob.glob(os.path.join(here, 'dk_full', 'boxes', 'box_*.jsonl.gz'))):
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
