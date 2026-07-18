"""verify_recover.py -- standalone replay verifier for the IN-3 low-delta STUCK recovery layer
(recover_out/, produced by recover_lowdelta.py).  Checks, from disk only:

 (V1) COVERAGE: the set of 'src' top-cell boxes in rec_[0-9]*.jsonl equals EXACTLY the set of
      'stuck' cells of the dk_full sweep (dk_full/stuck_all.jsonl, or re-extracted from boxes/*.gz).
 (V2) DYADIC TILING: within each top-cell group the leaf rows tile the src box exactly.  Verified by
      iterative sibling-merge: two boxes with equal hw that agree on all axes but one k, whose k-centers
      differ by 2*hw[k], merge to their parent; a valid partition merges back to the src box.  Exact in
      floats (all cuts are midpoints; hw halved).  Works for gamma-only AND adaptive multi-axis splits.
 (V3) STATUS: every leaf st in {cert,fail}.  A 'fail' leaf is admissible IFF a rec_fix.jsonl group has
      src == that fail box, is itself (V2)-tiled, and has ZERO fail leaves (fail re-closure, as wf_fix).
      Any fail leaf not so re-closed => FAIL.
 (V4) CERT ROWS: re-run dk_kernel.certify_cell(cen,hw,triples,theta) from the stored witness and assert
      ok (Hadj<0 & lam_lo>0).  --sample K re-runs K pseudorandom cert leaves per shard; --full does all.
      (Shares dk_kernel/tm2 with the emitter -- re-execution replay, not an independent reimplementation.)

Usage: python verify_recover.py [--outdir recover_out] [--sample 400] [--full]
"""
import sys, os, json, gzip, glob, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dk_kernel as DK

TOL = 1e-9   # relative-safe float match for dyadic boxes (hw down to ~1e-7)

def _bkey(cen, hw):
    return (tuple(round(c, 11) for c in cen), tuple(round(h, 11) for h in hw))

def load_stuck_set():
    here = os.path.dirname(os.path.abspath(__file__))
    dump = os.path.join(here, 'dk_full', 'stuck_all.jsonl')
    S = set()
    if os.path.exists(dump):
        for l in open(dump):
            cen, hw = json.loads(l); S.add(_bkey(cen, hw))
        return S
    for fn in glob.glob(os.path.join(here, 'dk_full', 'boxes', 'box_*.jsonl.gz')):
        for l in gzip.open(fn, 'rt'):
            r = json.loads(l)
            if r.get('kind') == 'stuck': S.add(_bkey(r['cen'], r['hw']))
    return S

def load_groups(fn):
    cur = None; src = None
    for l in open(fn):
        r = json.loads(l)
        if r.get('top'):
            if cur is not None: yield src[0], src[1], cur
            src = r['src']; cur = [r]
        else:
            if cur is None: continue
            cur.append(r)
    if cur is not None: yield src[0], src[1], cur

def check_tiling(src_cen, src_hw, leaves):
    """iterative sibling-merge; True iff leaf boxes exactly dyadically partition the src box."""
    boxes = [(list(r['cen']), list(r['hw'])) for r in leaves]
    # dedup guard: no exact duplicate leaf
    keys = [_bkey(c, h) for c, h in boxes]
    if len(set(keys)) != len(keys): return False, 'dup-leaf'
    changed = True
    while len(boxes) > 1 and changed:
        changed = False
        n = len(boxes)
        used = [False] * n
        out = []
        # index by hw for cheaper pairing
        for i in range(n):
            if used[i]: continue
            ci, hi = boxes[i]
            merged = False
            for j in range(i + 1, n):
                if used[j]: continue
                cj, hj = boxes[j]
                if any(abs(hi[a] - hj[a]) > TOL for a in range(5)): continue
                diff = [a for a in range(5) if abs(ci[a] - cj[a]) > TOL]
                if len(diff) != 1: continue
                k = diff[0]
                if abs(abs(ci[k] - cj[k]) - 2 * hi[k]) > TOL: continue
                pc = list(ci); pc[k] = (ci[k] + cj[k]) / 2
                ph = list(hi); ph[k] = hi[k] * 2
                out.append((pc, ph)); used[i] = used[j] = True; merged = True; changed = True
                break
            if not merged:
                out.append((ci, hi)); used[i] = True
        boxes = out
    if len(boxes) == 1:
        c, h = boxes[0]
        if all(abs(c[a] - src_cen[a]) <= TOL and abs(h[a] - src_hw[a]) <= TOL for a in range(5)):
            return True, 'ok'
        return False, 'merged-to-wrong-box'
    return False, 'no-full-merge(%d)' % len(boxes)

def _v4_chunk(rows):
    """re-execute certify_cell on a chunk of cert rows; return count that fail to reproduce ok."""
    import dk_kernel as _DK
    bad = 0
    for r in rows:
        try:
            o = _DK.certify_cell(r['cen'], r['hw'], r['triples'], r['theta'])
            if not o.get('ok'): bad += 1
        except Exception:
            bad += 1
    return bad

def main():
    outdir = 'recover_out'; sample = 400; full = False; jobs = 1
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x == '--outdir': outdir = a[i + 1]
        elif x == '--sample': sample = int(a[i + 1])
        elif x == '--full': full = True
        elif x == '--jobs': jobs = int(a[i + 1])
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, outdir)
    files = sorted(glob.glob(os.path.join(outdir, 'rec_[0-9]*.jsonl')))
    fixfn = os.path.join(outdir, 'rec_fix.jsonl')
    print('recover shards:', len(files), '| fix file:', os.path.exists(fixfn))

    # fix groups: src_key -> (tiling_ok, zero_fail)
    fixmap = {}
    if os.path.exists(fixfn):
        for scen, shw, leaves in load_groups(fixfn):
            tok, _ = check_tiling(scen, shw, leaves)
            zf = all(r['st'] == 'cert' for r in leaves if not r.get('top') or r['st'] in ('cert', 'fail'))
            zf = not any(r['st'] == 'fail' for r in leaves)
            fixmap[_bkey(scen, shw)] = (tok, zf, leaves)

    stuck = load_stuck_set()
    srcset = set(); n_top = 0; n_leaf = 0
    tiling_bad = []; unclosed_fail = []; cert_rows = []
    for fn in files:
        for scen, shw, leaves in load_groups(fn):
            n_top += 1
            srcset.add(_bkey(scen, shw))
            ok, why = check_tiling(scen, shw, leaves)
            if not ok: tiling_bad.append((os.path.basename(fn), why))
            for r in leaves:
                n_leaf += 1
                if r['st'] == 'cert':
                    cert_rows.append(r)
                elif r['st'] == 'fail':
                    fk = _bkey(r['cen'], r['hw'])
                    fx = fixmap.get(fk)
                    if not fx or not fx[0] or not fx[1]:
                        unclosed_fail.append(fk)
                    else:
                        cert_rows.extend(x for x in fx[2] if x['st'] == 'cert')

    missing = stuck - srcset; extra = srcset - stuck
    v1 = (len(missing) == 0 and len(extra) == 0 and len(srcset) == len(stuck))
    print('V1 coverage: stuck=%d src=%d missing=%d extra=%d  %s'
          % (len(stuck), len(srcset), len(missing), len(extra), 'PASS' if v1 else 'FAIL'))
    v2 = (len(tiling_bad) == 0)
    print('V2 dyadic-tiling: top-cells=%d bad=%d  %s' % (n_top, len(tiling_bad), 'PASS' if v2 else 'FAIL'))
    if tiling_bad[:5]: print('   examples:', tiling_bad[:5])
    v3 = (len(unclosed_fail) == 0)
    print('V3 status: leaves=%d unclosed-fail=%d (fix-closed ok)  %s'
          % (n_leaf, len(unclosed_fail), 'PASS' if v3 else 'FAIL'))

    rng = random.Random(12345)
    pick = cert_rows if full else rng.sample(cert_rows, min(sample * max(1, len(files)), len(cert_rows)))
    if jobs > 1 and len(pick) > jobs:
        import multiprocessing as mp
        try: mp.set_start_method('fork')
        except RuntimeError: pass
        nchunks = jobs * 8  # over-shard for load balance
        chunks = [pick[i::nchunks] for i in range(nchunks)]
        with mp.Pool(jobs) as pool:
            bad = sum(pool.map(_v4_chunk, chunks))
    else:
        bad = _v4_chunk(pick)
    v4 = (bad == 0)
    print('V4 re-exec (%s %d of %d cert leaves, jobs=%d): bad=%d  %s'
          % ('FULL' if full else 'sample', len(pick), len(cert_rows), jobs, bad, 'PASS' if v4 else 'FAIL'))

    verdict = 'PASS' if (v1 and v2 and v3 and v4) else 'FAIL'
    print('-' * 60); print('RECOVER-LAYER VERDICT:', verdict)
    json.dump(dict(v1=v1, v2=v2, v3=v3, v4=v4, stuck=len(stuck), src=len(srcset),
                   leaves=n_leaf, unclosed_fail=len(unclosed_fail), verdict=verdict),
              open(os.path.join(outdir, 'verify_recover_report.json'), 'w'), indent=1)
    return 0 if verdict == 'PASS' else 1

if __name__ == '__main__':
    sys.exit(main())
