"""adelta_run_all.py -- full overnight A_delta sweep over delta in [1.5e-3,1.5e-2], phi in [0,pi].
Shards delta (finer at low delta where the short-edge mean-value is costly), runs adelta_sweep4.sweep
per shard across a process pool, caches each shard's result (resumable), and aggregates.

PROVES  A_delta <= max(shard maxima) < 16  (ladder-closing) with 0 unresolved cells, off the +-1e-6
silhouette-transition windows (excluded, charged to margin-continuity as in the paper).

Usage: python adelta_run_all.py [NPROC]      (results in adelta_out/, summary adelta_out/SUMMARY.json)
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adelta_out')

def shards():
    out = []
    d = 1.5e-3
    while d < 1.5e-2 - 1e-12:
        w = 1e-5 if d < 3e-3 else 5e-5           # fine low-delta, coarse high-delta
        out.append((d, min(d + w, 1.5e-2)))
        d += w
    return out

def work(slab):
    import adelta_sweep4 as SW
    dlo, dhi = slab
    tag = f'{dlo:.7e}_{dhi:.7e}'
    fp = os.path.join(OUT, f'sh_{tag}.json')
    if os.path.exists(fp):
        try: return json.load(open(fp))
        except Exception: pass
    t0 = time.time()
    st = SW.sweep(dlo, dhi)
    rep = dict(dlo=dlo, dhi=dhi, max=st['max'], loc=st['loc'], acc=st['acc'], excl=st['excl'],
               cells=st['cells'], unres=len(st['unres']), unres_ex=st['unres'][:2], secs=time.time()-t0)
    json.dump(rep, open(fp, 'w'))
    return rep

def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, mp.cpu_count() - 2)
    os.makedirs(OUT, exist_ok=True)
    sl = shards()
    print(f'{len(sl)} shards, {nproc} procs', flush=True)
    t0 = time.time()
    with mp.Pool(nproc) as p:
        res = []
        for i, r in enumerate(p.imap_unordered(work, sl), 1):
            res.append(r)
            if i % 10 == 0 or i == len(sl):
                gm = max(x['max'] for x in res); ur = sum(x['unres'] for x in res)
                print(f'  {i}/{len(sl)} done | running max={gm:.4f} unres={ur} | {time.time()-t0:.0f}s', flush=True)
    gmax = max(x['max'] for x in res); unres = sum(x['unres'] for x in res)
    acc = sum(x['acc'] for x in res); excl = sum(x['excl'] for x in res); cells = sum(x['cells'] for x in res)
    loc = max(res, key=lambda x: x['max'])['loc']
    summ = dict(A_delta_bound=gmax, argmax=loc, acc=acc, excl=excl, cells=cells, unres=unres,
                shards=len(sl), secs=time.time()-t0,
                verdict=('PROVED A_delta<=%.4f < 16 (ladder-closing), 0 unresolved' % gmax)
                        if (gmax < 16 and unres == 0) else 'INCOMPLETE')
    json.dump(summ, open(os.path.join(OUT, 'SUMMARY.json'), 'w'), indent=1)
    print('\n' + json.dumps(summ, indent=1), flush=True)

if __name__ == '__main__':
    main()
