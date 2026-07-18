"""r84 manifest emitter for IN-2 wall layer. Runs the full per-src verification (same rules
as verify_wallfix.py) and, ONLY if every dg_full wall cell is closed, emits
manifest_depthgrid_r84.json consumed by assemble.py. The manifest records: coverage counts,
method histogram, min certified depth, citation radius rc + backing sbbox manifest, the
verification protocol actually executed (citation recheck = FULL; wall2d/w5 re-execution =
sample size recorded), and per-file row counts. Usage:
  python manifest_depthgrid.py [--sample 2000] [--full-reexec]"""
import sys, os, json, glob, math, random, time
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
        elif cur is not None: cur['rows'].append(row)
    if cur is not None: groups.append(cur)
    return groups

def tiles(src,leaves):
    th,thw,ph,phw=src
    key={}
    for r in leaves:
        k=(r['th'],r['thw'],r['ph'],r['phw'])
        if k in key: return False
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
    return cover(th,thw,ph,phw) and all(key.values())

def main():
    sample=2000; fullre=False
    args=sys.argv[1:]
    while args:
        a=args.pop(0)
        if a=='--sample': sample=int(args.pop(0))
        elif a=='--full-reexec': fullre=True
    outdir='wallfix_out'
    shards=sorted(glob.glob(os.path.join(outdir,'wf_[0-9]*.jsonl')))
    notdone=[fn for fn in shards if not os.path.exists(fn+'.done')]
    if notdone:
        print("REFUSE: shards not .done:",notdone); return 1
    dgwall=set()
    for fn in sorted(glob.glob('dg_full/dg_*.jsonl')):
        for ln in open(fn):
            r=json.loads(ln)
            if r.get('st')=='wall': dgwall.add((r['th'],r['thw'],r['ph'],r['phw']))
    fixfn=os.path.join(outdir,'wf_fix.jsonl')
    allg=[]
    files={}
    for fn in shards+([fixfn] if os.path.exists(fixfn) else []):
        gs=load_groups(fn); allg+=gs; files[fn]=sum(len(g['rows']) for g in gs)
    bysrc={}
    for g in allg:
        if g['src'] is None: continue
        bysrc.setdefault(tuple(g['src']),[]).append(g)
    missing=dgwall-set(bysrc.keys())
    if missing:
        print("REFUSE: %d dg wall cells with no group"%len(missing)); return 1
    certrows=[]; unclosed=[]
    for k,gs in bysrc.items():
        best=None
        for g in gs:
            leaves=[r for r in g['rows'] if r.get('st') in ('cert','fail')]
            if not tiles(g['src'],leaves): continue
            if any(r['st']=='fail' for r in leaves): continue
            best=leaves; break
        if best is None: unclosed.append(k); continue
        certrows+=best
    if unclosed:
        print("REFUSE: %d unclosed srcs"%len(unclosed))
        for k in unclosed[:10]: print("  ",k)
        return 1
    # citation recheck: FULL
    badc=0; RCs=set(); hist={}; dmin=9.0
    for r in certrows:
        m=r.get('m','?'); hist[m]=hist.get(m,0)+1; dmin=min(dmin,r['d'])
        if m=='sbbox':
            RCs.add(r['rc'])
            dlo_,dhi_=dot_us_bounds(r['th'],r['thw'],r['ph'],r['phw'])
            CU=cos_upper(r['rc'])
            dv=((FI(1.0)-FI(dhi_))*FI(2.0)).sqrt().lo/500.0 if dhi_<1.0 else -1
            if not (dlo_>=CU and 0<r['d']<=dv): badc+=1
    if badc: print("REFUSE: %d bad citations"%badc); return 1
    rcback={}
    for rc in RCs:
        need='manifest_sbbox_bulk_2.6e-03.json' if rc==5.0e-3 else ('manifest_sbbox_bulk_1.3e-03.json' if rc==2.5e-3 else None)
        if need is None or not os.path.exists(need):
            print("REFUSE: rc=%s unbacked"%rc); return 1
        rcback[str(rc)]=need
    # re-execution
    import depth_twocase as TC
    other=[r for r in certrows if r.get('m')!='sbbox']
    rng=random.Random(20260705)
    sel=other if fullre else rng.sample(other,min(sample,len(other)))
    bad2=0
    for r in sel:
        m=r['m']
        try:
            if m.startswith('w5'): d,ok,_=TC.w5_collision_cert(r['th'],r['thw'],r['ph'],r['phw'])
            else: d,ok,_=TC.dlo_cert_wall2d(r['th'],r['thw'],r['ph'],r['phw'])
        except Exception: ok=False
        if not ok or d<=0: bad2+=1
    if bad2: print("REFUSE: %d re-exec failures"%bad2); return 1
    man={'kind':'depthgrid_wall_layer','date':time.strftime('%Y-%m-%dT%H:%M:%SZ'),
         'dg_wall_cells':len(dgwall),'closed_srcs':len(bysrc),'cert_leaf_rows':len(certrows),
         'method_hist':hist,'min_d':dmin,'rc_backing':rcback,
         'checks':{'coverage':'FULL','tiling':'FULL','citations':'FULL',
                   'reexec':'FULL' if fullre else 'sample:%d'%len(sel)},
         'files':files,
         'caveats':['re-execution replay shares depth_twocase.py with the emitter',
                    'libm directed-rounding axiom for interval trig (as in paper 15.5(ii))',
                    'dg_full cert-cell layer (986,870 cells) verified separately; this manifest covers the wall layer']}
    out='manifest_depthgrid_r84.json'
    json.dump(man,open(out,'w'),indent=1)
    print("EMITTED",out); print(json.dumps(man['method_hist'],indent=1)); print("min_d=%.3g"%dmin)
    return 0

if __name__=='__main__':
    sys.exit(main())
