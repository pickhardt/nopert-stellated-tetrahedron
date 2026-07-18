"""recover_fix.py -- mop-up pass for the IN-3 recovery residual 'fail' leaves.

The main gamma-only recovery (recover_lowdelta.py, running job) leaves a handful of 'fail' leaves at
the deep wall phi_w core (phi~1.039, delta~0.0062): cells that gamma-subdivision alone can't close
because there the enclosure-tight direction is p1/phi, not gamma.  DIAGNOSIS: those cells still have
margin_lp>0 (~4e-7) and each closes with ONE adaptive best-axis split.  This pass reprocesses every
'fail' leaf across recover_out/rec_[0-9]*.jsonl with the adaptive closer and writes rec_fix.jsonl:
one top-group per fail box (src = the fail box), leaves adaptively tiling it, 0 fail.  verify_recover.py
admits a fail leaf iff rec_fix has a matching (V2-tiled, 0-fail) group for it.

Usage: python recover_fix.py
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dk_kernel as DK
import dk_sweep as SW

MAXD = 14
_SCALE = [2e-2, 3e-3, 1e-2, 1e-2, 1e-2]

def _cert_row(cen, hw, r):
    return dict(st='cert', cen=list(cen), hw=list(hw), triples=r.get('triples'),
                theta=r.get('theta'), Hadj=r.get('Hadj'), lam_lo=r['lam_lo'], val_hi=r['val_hi'])

def _best_axis(cen, hw):
    for ax in (2, 3, 1, 4, 0):   # wall-core: p1/p2 first (the enclosure-tight dirs here)
        both = True
        for sg in (-0.5, 0.5):
            c2 = list(cen); c2[ax] = cen[ax] + sg * hw[ax]; h2 = list(hw); h2[ax] = hw[ax] / 2
            if not DK.try_cell(c2, h2).get('ok'): both = False; break
        if both: return ax
    return max(range(5), key=lambda i: hw[i] / _SCALE[i])

def close(cen, hw, depth, rows, stat):
    try:
        r = DK.try_cell(cen, hw)
    except Exception as e:
        r = dict(ok=False, why='exc:' + type(e).__name__)
    if r.get('ok'):
        rows.append(_cert_row(cen, hw, r)); stat['cert'] = stat.get('cert', 0) + 1; return True
    if depth >= MAXD:
        lo = SW._lp_cert(cen, hw, None)
        if lo is not None:
            rows.append(_cert_row(cen, hw, lo)); stat['lp'] = stat.get('lp', 0) + 1; return True
        rows.append(dict(st='fail', cen=list(cen), hw=list(hw), why=r.get('why', 'cert')))
        stat['fail'] = stat.get('fail', 0) + 1; return False
    ax = _best_axis(cen, hw)
    h2 = list(hw); h2[ax] = hw[ax] / 2
    ok = True
    for sgn in (-0.5, 0.5):
        c2 = list(cen); c2[ax] = cen[ax] + sgn * hw[ax]
        if not close(c2, h2, depth + 1, rows, stat): ok = False
    return ok

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'recover_out')
    fails = []
    for fn in sorted(glob.glob(os.path.join(outdir, 'rec_[0-9]*.jsonl'))):
        for l in open(fn):
            r = json.loads(l)
            if r.get('st') == 'fail': fails.append((r['cen'], r['hw']))
    print('residual fail leaves to re-close:', len(fails))
    if not fails:
        print('no fails -- nothing to do'); return 0
    out = open(os.path.join(outdir, 'rec_fix.jsonl'), 'w')
    stat = {}; t0 = time.time(); nbad = 0
    for i, (cen, hw) in enumerate(fails):
        rows = []
        ok = close(cen, hw, 0, rows, stat)
        rows[0]['top'] = 1; rows[0]['src'] = [list(cen), list(hw)]; rows[0]['nsub'] = len(rows)
        for r in rows: out.write(json.dumps(r) + '\n')
        if not ok: nbad += 1
    out.close()
    print('DONE  stats=%s  unclosed=%d  (%.0fs)' % (stat, nbad, time.time() - t0))
    return 0 if nbad == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
