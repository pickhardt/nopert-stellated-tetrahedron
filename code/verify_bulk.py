"""verify_bulk.py -- STANDALONE replay verifier for the (SB-box) bulk manifests.

Consumes ONLY the manifest files manifest_sbbox_bulk_*.json (rows
[dlo,dhi,flo,fhi,mode,minslack,eta_rel]) and re-establishes, independently of
the adaptive driver and its state:

  (T)  the rectangles tile [1e-5, 1.28e-3] x [0, pi] EXACTLY (no gap, no
       overlap), verified by an exact-Fraction sweep-line over delta-strips;
  (F)  no FAIL rows;
  (C)  every row's certificate re-verifies through the interval kernel
       (cell_checks + verify_ppoint of flagcalc2/wall_sweep2), with the
       recorded mode; recorded eta_rel <= ETA_BUDGET re-derived here;
  (H)  any row with positivity slack < SLACK_FLOOR is re-certified on its
       four half-width children (hairline-cell hardening).

Soundness note: the replay shares the interval kernel (flagcalc2) with the
emitter but NOT the adaptive driver, its splitting logic, or its state; the
float candidate-weight construction is heuristic and irrelevant to soundness
(Lemma WS-2), so a fresh candidate solve here is a genuine re-verification.

Usage: python verify_bulk.py [chunk_index] [num_chunks]
Writes verify_bulk_report_<chunk>.json.
"""
import sys, os, json, glob, time
from fractions import Fraction as Fr
sys.path.insert(0, 'certificates')
sys.path.insert(0, '.')
import numpy as np
from wall_sweep2 import try_cell, ETA_BUDGET, modes_for, PHI_W

D1, D2 = 1e-5, 1.28e-3
SLACK_FLOOR = 1e-7

def load_rows():
    rows = []
    for fn in sorted(glob.glob('certificates/manifest_sbbox_bulk_*.json')):
        rows += [tuple(r) for r in json.load(open(fn))]
    return rows

def check_tiling(rows):
    """Exact no-gap no-overlap tiling of [D1,D2]x[0,pi] by the rectangles."""
    rects = [(Fr(r[0]), Fr(r[1]), Fr(r[2]), Fr(r[3])) for r in rows]
    lo_d, hi_d = Fr(D1), Fr(D2)
    PI = Fr(np.pi)   # exact value of the float pi used by the driver
    cuts = sorted({d for r in rects for d in (r[0], r[1])})
    assert cuts[0] == lo_d and cuts[-1] == hi_d, "delta range mismatch"
    for a, b in zip(cuts, cuts[1:]):
        strip = [r for r in rects if r[0] <= a and r[1] >= b]
        # every rect intersecting the open strip must span it (binary splits)
        for r in rects:
            if r[0] < b and r[1] > a:
                assert r[0] <= a and r[1] >= b, ("straddling rect", r, a, b)
        ivs = sorted((r[2], r[3]) for r in strip)
        cover = Fr(0)
        for flo, fhi in ivs:
            assert flo == cover, ("phi gap/overlap", float(a), float(flo), float(cover))
            cover = fhi
        assert cover == PI, ("phi cover ends short", float(cover))
    return len(cuts) - 1

def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nchunk = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rows = load_rows()
    report = {'n_rows': len(rows), 'chunk': chunk, 'nchunk': nchunk}
    # (F) no FAIL rows + eta budget
    fails = [r for r in rows if r[4] == 'FAIL']
    assert not fails, f"{len(fails)} FAIL rows"
    bad_eta = [r for r in rows if not (r[6] <= ETA_BUDGET)]
    assert not bad_eta, f"eta over budget: {bad_eta[:3]}"
    report['eta_budget'] = float(ETA_BUDGET)
    report['max_eta_recorded'] = max(r[6] for r in rows)
    report['min_slack_recorded'] = min(r[5] for r in rows)
    if chunk == 0:
        report['n_strips'] = check_tiling(rows)
        report['tiling'] = 'EXACT-PASS'
    # (C) replay
    t0 = time.time(); nbad = 0; minsl = np.inf; maxeta = 0.0; nhair = 0
    mine_rows = [r for i, r in enumerate(rows) if i % nchunk == chunk]
    for (dlo, dhi, flo, fhi, mode, sl0, eta0) in mine_rows:
        ok, sl, eta = try_cell(dlo, dhi, flo, fhi, mode)
        if not ok:
            nbad += 1
            print('REPLAY FAIL', dlo, dhi, flo, fhi, mode)
            continue
        minsl = min(minsl, sl); maxeta = max(maxeta, eta)
        if sl < SLACK_FLOOR:   # (H) hairline hardening: 4 half-width children
            nhair += 1
            dm, fm = 0.5*(dlo+dhi), 0.5*(flo+fhi)
            for (a, b) in [(dlo, dm), (dm, dhi)]:
                for (c, d) in [(flo, fm), (fm, fhi)]:
                    okc = False
                    for m2 in ([mode] + modes_for(c, d)):
                        okc, slc, etac = try_cell(a, b, c, d, m2)
                        if okc: break
                    assert okc, ('hairline child failed', a, b, c, d)
    report.update(n_replayed=len(mine_rows), n_bad=nbad, n_hairline=nhair,
                  min_slack_replay=float(minsl), max_eta_replay=float(maxeta),
                  seconds=time.time()-t0)
    assert nbad == 0
    json.dump(report, open(f'certificates/verify_bulk_report_{chunk}.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))

if __name__ == '__main__':
    main()
