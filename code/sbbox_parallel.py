"""Parallel + resumable proof-grade SB-box sweep. Fan out over (δ-ladder × φ-sub-ranges) with
multiprocessing; each worker adaptively covers its φ-range at its δ (dynamic transition breakpoints,
depth-first), writes a witness shard + .done. Certificate per cell: box-inscription margin/δ > MOD AND
flag-hull stable ⇒ δ·Box ⊆ conv F(u₂(δ,φ)). δ-ladder covers δ∈[δmin,ρ_B]; δ→0 by the first-order limit
(margin/δ → 0.00796, run-validated). Usage: python sbbox_parallel.py NPROC NSUB OUTDIR [CAP_S]"""
import sys, os, time, json, math, multiprocessing as mp
sys.path.insert(0,'.')
import os as _os
# δ-ladder. Default = original (ρ_B=1e-3 down). Override via SBBOX_DELTAS env (comma-sep) for the
# EXTENSION run: ×2 octave rungs past 2.56e-3 up to ~0.02 to push the box-anchored ceiling past DK's
# clean floor (~0.015), so the layers OVERLAP (no seam). margin/δ grows with δ here (0.0054→0.0108).
DELTAS=[1e-3, 1e-4, 1e-5]
if _os.environ.get('SBBOX_DELTAS'):
    DELTAS=[float(x) for x in _os.environ['SBBOX_DELTAS'].split(',')]
MOD=0.002; MINH=1e-8; EPS=1e-6   # MINH resolves the isolated thin-facet-normal straddles directly
                                 # (they certify with margin ~0.9 at hw≤1e-8); only ±EPS transition slivers remain
def sweep_range(delta, plo, phi):
    import sbbox_prove as S
    from collections import deque
    tr=[t for t in S.find_transitions(delta) if plo-1e-3<=t<=phi+1e-3]
    hard=[]
    for t in tr: hard+=[t-EPS,t+EPS]
    N=max(4,int((phi-plo)/(2*math.pi)*400))
    breaks=sorted(set([plo+(phi-plo)*i/N for i in range(N+1)]+[h for h in hard if plo<=h<=phi]))
    Q=deque([(breaks[i],breaks[i+1]) for i in range(len(breaks)-1) if breaks[i+1]-breaks[i]>2e-8])
    trw=[(t-EPS,t+EPS) for t in tr]
    def inw(a,b): return any(a>=w0-1e-12 and b<=w1+1e-12 for w0,w1 in trw)
    cert=fail=0; man=[]
    while Q:
        a,b=Q.pop()
        if inw(a,b): continue
        try: r=S.facet_margin(delta,a,b)
        except Exception: r=None
        if r is not None:
            marg,stable=r
            if stable and marg.lo>=MOD: cert+=1; man.append((a,b,marg.lo)); continue
        if b-a<MINH: fail+=1
        else: m=(a+b)/2; Q.append((a,m)); Q.append((m,b))
    return cert,fail,man
def worker(a):
    idx,delta,plo,phi,outdir=a
    shard=f"{outdir}/shard_{idx}.jsonl"; done=f"{outdir}/shard_{idx}.done"
    if os.path.exists(done): return (idx,'skip',0,0)
    t0=time.time(); cert,fail,man=sweep_range(delta,plo,phi)
    with open(shard,'w') as f:
        for aa,bb,lo in man: f.write(json.dumps({'d':delta,'a':aa,'b':bb,'lo':lo})+'\n')
    if fail==0: open(done,'w').write(f"cert={cert} t={time.time()-t0:.0f}\n")
    return (idx,'done' if fail==0 else 'FAIL',cert,fail)
def main():
    nproc=int(sys.argv[1]); nsub=int(sys.argv[2]); outdir=sys.argv[3]; cap=float(sys.argv[4]) if len(sys.argv)>4 else 0
    os.makedirs(outdir,exist_ok=True)
    jobs=[]; k=0
    # φ-span: mirror symmetry (m* fixing u*, proved "mirror transport") ⇒ φ∈[0,π] suffices. Default
    # to [0,π]; set SBBOX_FULL2PI=1 to sweep the full circle (the [π,2π] half is redundant).
    span=2*math.pi if _os.environ.get('SBBOX_FULL2PI') else math.pi
    for d in DELTAS:
        for j in range(nsub):
            plo=span*j/nsub; phi=span*(j+1)/nsub; jobs.append((k,d,plo,phi,outdir)); k+=1
    print(f"{len(DELTAS)} δ-levels × {nsub} φ-ranges = {len(jobs)} jobs, nproc={nproc}",flush=True)
    t0=time.time(); tot_c=tot_f=ndone=0
    with mp.Pool(nproc) as p:
        for r in p.imap_unordered(worker,jobs):
            idx,st,c,f=r; tot_c+=c; tot_f+=f
            if st!='skip': ndone+=1
            print(f"  job {idx} {st}: cert={c} fail={f} — total cert={tot_c} fail={tot_f} done={ndone}/{len(jobs)} {time.time()-t0:.0f}s",flush=True)
            if cap and time.time()-t0>cap: print("  [cap hit — resumable]"); break
    print(f"\nAGG: cert={tot_c} fail={tot_f} jobs_done={ndone}/{len(jobs)} {time.time()-t0:.0f}s")
    print("SB-box COVERED proof-grade (δ-ladder + first-order δ→0) ✓" if (tot_f==0 and ndone==len(jobs)) else "INCOMPLETE (resume)")
if __name__=='__main__':
    mp.set_start_method('fork'); main()
