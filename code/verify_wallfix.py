"""r84 standalone replay verifier for the IN-2 wall layer (wallfix_out).
Checks, from disk only:
 (V1) COVERAGE: the set of top-cell boxes in wf_[0-9]*.jsonl equals EXACTLY the set of
      'wall' cells in dg_full/*.jsonl (float-exact box equality, count and set).
 (V2) TILING: within each top-cell group, the leaf rows (st in {cert,fail}) tile the src
      box exactly: verified by recursive dyadic reconstruction (each leaf must be reachable
      by halving; union area equals box area; no overlaps) -- exact in floats since all
      cuts are midpoints of floats.
 (V3) STATUS: every leaf has st in {cert,fail}; a group with any fail leaf must be re-closed
      in wf_fix.jsonl by a group with identical src and zero fail leaves (wf_fix groups are
      themselves (V2)-tiled and (V4)-checked).
 (V4) CERT ROWS: m='sbbox': independent recheck dlo==sqrt(2*(1-dot.hi))/500 with
      dot_us_bounds recomputed and dot.lo>=cos_upper(rc), rc backed by the sbbox manifest
      file on disk. m='wall2d..'/'w5(4)'/'wall2b': re-execute the certificate (fresh run;
      NOTE: shares depth_twocase.py with the emitter -- re-execution replay, not an
      independent reimplementation; the exact-tail layer has the independent verifier).
      --sample N re-executes a deterministic pseudorandom subset of N rows per shard.
Exit: prints PASS/FAIL per check + summary. Usage:
  python verify_wallfix.py [--outdir wallfix_out] [--sample 500] [--full]"""
import sys, os, json, math, glob, random
sys.path.insert(0,'.')
import fast_interval as F
from fast_interval import FI
from dotus_tight import dot_us_bounds

def cos_upper(x):
    xi=FI(x); return (FI(1.0)-xi.sqr()*FI(0.5)+xi.sqr().sqr()*FI(1.0/24.0)).hi

def load_groups(fn):
    groups=[]; cur=None
    for ln in open(fn):
        row=json.loads(ln)
        if row.get('top'):
            if cur is not None: groups.append(cur)
            cur={'src':row.get('src'),'rows':[row],'file':fn}
        elif cur is not None:
            cur['rows'].append(row)
    if cur is not None: groups.append(cur)
    return groups

def check_tiling(src,leaves):
    """exact dyadic tiling check: recursively verify the leaf multiset tiles src."""
    th,thw,ph,phw=src
    key={}
    for r in leaves:
        k=(r['th'],r['thw'],r['ph'],r['phw'])
        if k in key: return False,'dup-leaf'
        key[k]=False
    def cover(t,tw,p,pw,depth=0):
        if (t,tw,p,pw) in key:
            if key[(t,tw,p,pw)]: return False
            key[(t,tw,p,pw)]=True; return True
        if depth>16: return False
        for sa in (-0.5,0.5):
            for sb in (-0.5,0.5):
                if not cover(t+sa*tw,tw/2,p+sb*pw,pw/2,depth+1): return False
        return True
    ok=cover(th,thw,ph,phw)
    return (ok and all(key.values())), ('ok' if ok else 'gap/extra')

def _v4_chunk(rows):
    """re-execute the wall2d/w5 depth certificate on a chunk; return count that fail to reproduce."""
    import depth_twocase as _TC
    bad=0
    for r in rows:
        m=r['m']
        try:
            if m.startswith('w5'): d,ok,why=_TC.w5_collision_cert(r['th'],r['thw'],r['ph'],r['phw'])
            else: d,ok,why=_TC.dlo_cert_wall2d(r['th'],r['thw'],r['ph'],r['phw'])
        except Exception: ok=False; d=0
        if not ok or d<=0: bad+=1
    return bad

def main():
    outdir='wallfix_out'; sample=200; full=False; jobs=1
    args=sys.argv[1:]
    while args:
        a=args.pop(0)
        if a=='--outdir': outdir=args.pop(0)
        elif a=='--sample': sample=int(args.pop(0))
        elif a=='--full': full=True
        elif a=='--jobs': jobs=int(args.pop(0))
    # (V1) coverage vs dg_full
    dgwall=set()
    for fn in sorted(glob.glob('dg_full/dg_*.jsonl')):
        for ln in open(fn):
            r=json.loads(ln)
            if r.get('st')=='wall': dgwall.add((r['th'],r['thw'],r['ph'],r['phw']))
    tops=set(); groups_all=[]
    for fn in sorted(glob.glob(os.path.join(outdir,'wf_[0-9]*.jsonl'))):
        gs=load_groups(fn); groups_all+=gs
        for g in gs:
            if g['src']: tops.add(tuple(g['src']))
    missing=dgwall-tops; extra=tops-dgwall
    complete = not missing
    print("V1 coverage: dg wall cells=%d, wf top cells=%d, missing=%d extra=%d %s"%(
        len(dgwall),len(tops),len(missing),len(extra),
        "PASS" if (complete and not extra) else ("INCOMPLETE" if not extra else "FAIL")))
    # wf_fix groups
    fixfn=os.path.join(outdir,'wf_fix.jsonl')
    fixgroups=load_groups(fixfn) if os.path.exists(fixfn) else []
    # (V2)+(V3) per-src rule: a src is CLOSED iff SOME group (wf_* or wf_fix) with that src
    # has leaves that exactly tile the src box with zero fail leaves. Partial groups from
    # kill/restart are superseded by wf_fix groups; live-shard tail groups are in flight.
    bysrc={}
    for g in groups_all+fixgroups:
        if g['src'] is None: continue
        bysrc.setdefault(tuple(g['src']),[]).append(g)
    closed=set(); certrows=[]; RCseen=set(); npartial=0
    for k,gs in bysrc.items():
        best=None
        for g in gs:
            leaves=[r for r in g['rows'] if r.get('st') in ('cert','fail')]
            okt,_=check_tiling(g['src'],leaves)
            if not okt: continue
            if any(r['st']=='fail' for r in leaves): continue
            best=leaves; break
        if best is None: npartial+=1; continue
        closed.add(k)
        for r in best:
            certrows.append(r)
            if r.get('m')=='sbbox': RCseen.add(r.get('rc'))
    unclosed=[k for k in bysrc if k not in closed]
    missing_dg=dgwall-set(bysrc.keys())
    print("V2/V3 per-src: srcs=%d closed=%d unclosed=%d (live/partial), dg-wall not yet seen=%d %s"%(
        len(bysrc),len(closed),len(unclosed),len(missing_dg),
        "PASS" if (not unclosed and not missing_dg) else "INCOMPLETE"))
    # (V4) cert rows
    # rc validity
    for rc in RCseen:
        need='manifest_sbbox_bulk_2.6e-03.json' if rc==5.0e-3 else ('manifest_sbbox_bulk_1.3e-03.json' if rc==2.5e-3 else None)
        okrc = need is not None and os.path.exists(need)
        print("  rc=%.3g backed by %s: %s"%(rc,need,"PASS" if okrc else "FAIL"))
    cite=[r for r in certrows if r['m']=='sbbox']
    other=[r for r in certrows if r['m']!='sbbox']
    badc=0
    for r in cite:
        dlo_,dhi_=dot_us_bounds(r['th'],r['thw'],r['ph'],r['phw'])
        rc=r['rc']; CU=cos_upper(rc)
        okc = dlo_>=CU and dhi_<1.0
        dv=((FI(1.0)-FI(dhi_))*FI(2.0)).sqrt().lo/500.0
        if not okc or not (0<r['d']<=dv): badc+=1
    print("V4 citations: %d rows, bad=%d %s"%(len(cite),badc,"PASS" if badc==0 else "FAIL"))
    # re-execution of wall2d/w5 rows
    rng=random.Random(20260705)
    sel=other if full else rng.sample(other,min(sample,len(other)))
    if jobs>1 and len(sel)>jobs:
        import multiprocessing as mp
        try: mp.set_start_method('fork')
        except RuntimeError: pass
        nchunks=jobs*8
        chunks=[sel[i::nchunks] for i in range(nchunks)]
        with mp.Pool(jobs) as pool:
            bad2=sum(pool.map(_v4_chunk,chunks))
    else:
        bad2=_v4_chunk(sel)
    print("V4 re-exec (%s %d of %d, jobs=%d): bad=%d %s"%("FULL" if full else "sample",len(sel),len(other),jobs,bad2,
          "PASS" if bad2==0 else "FAIL"))

if __name__=='__main__':
    main()
