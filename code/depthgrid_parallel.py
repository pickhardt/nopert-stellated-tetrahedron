"""Parallel + resumable depth-grid sweep over F. Fan out FBOX into sub-regions (skip those outside F);
each worker adaptively covers its sub-region with dlo_cert2 (point depth - proven Lambda_d*hw, hull-stable),
writes a witness shard + .done. Cells: cert (d_lo>0), defer (inside rho0-disk about u*, Lemma 6' covers),
wall (bottomed out at MINH, hull still unstable -> silhouette-transition two-case TODO), skip (outside F).
Usage: python depthgrid_parallel.py NPROC NSUB OUTDIR [CAP_S]"""
import sys, os, time, json, math, multiprocessing as mp
sys.path.insert(0,'.')
FBOX=(0.95,2.19,0.0,0.786); USTAR=(math.pi/2, math.pi/4); RHO0=1e-3; MINH=1e-6  # r-final: lowered 5e-5->1e-6 so dlo_cert_wall2d closes wall cells via subdivision (99.3%); near-u* residual bottoms out

def _disk(th,thw,ph,phw,inner):
    dth=abs(th-USTAR[0]); dph=abs(ph-USTAR[1])*math.sin(USTAR[0])
    if inner: return math.hypot(dth+thw, dph+phw*math.sin(USTAR[0]))<=RHO0   # WHOLE cell in disk
    return math.hypot(max(0.0,dth-thw), max(0.0,dph-phw*math.sin(USTAR[0])))<RHO0  # cell TOUCHES disk

def sweep_region(th0,th1,ph0,ph1,shard):
    import depth_grid2 as DG2, depth_twocase as TC, ball_cover5 as B5
    from collections import deque
    Q=deque([(th0,th1,ph0,ph1)]); cert=defer=wall=skip=0; dmin=9.0
    f=open(shard,'w')
    while Q:
        a,b,c,d=Q.pop(); th=(a+b)/2; ph=(c+d)/2; thw=(b-a)/2; phw=(d-c)/2
        if not B5.in_F(th,thw,ph,phw): skip+=1; continue
        if _disk(th,thw,ph,phw,True):
            defer+=1; f.write(json.dumps({'th':th,'thw':thw,'ph':ph,'phw':phw,'st':'defer'})+'\n'); continue
        if _disk(th,thw,ph,phw,False) and (thw>MINH or phw>MINH):
            m=(a+b)/2; mm=(c+d)/2; Q.extend([(a,m,c,mm),(m,b,c,mm),(a,m,mm,d),(m,b,mm,d)]); continue
        try: dd,ok,why=DG2.dlo_cert2(th,thw,ph,phw)
        except Exception: ok=False; why='exc'
        sub='plain'
        if not ok:   # hull-unstable transition cell → subset-safe (provably-silhouette stable edges)
            try: dd,ok,why=TC.dlo_cert_wall(th,thw,ph,phw); sub='wall'
            except Exception: ok=False
        if not ok:   # r80/final: coarse-combinatorics + degree inradius cert (93% direct, SOUND;
                     # -> 99.3% with the worker's own subdivision below; near-u* residual -> wall)
            try: dd,ok,why=TC.dlo_cert_wall2d(th,thw,ph,phw); sub='wall2d'
            except Exception: ok=False
        if ok:
            cert+=1; dmin=min(dmin,dd); f.write(json.dumps({'th':th,'thw':thw,'ph':ph,'phw':phw,'st':'cert','d':dd,'m':sub})+'\n')
        elif thw>MINH or phw>MINH:
            m=(a+b)/2; mm=(c+d)/2; Q.extend([(a,m,c,mm),(m,b,c,mm),(a,m,mm,d),(m,b,mm,d)])
        else:
            wall+=1; f.write(json.dumps({'th':th,'thw':thw,'ph':ph,'phw':phw,'st':'wall','why':why})+'\n')
    f.close(); return cert,defer,wall,skip,dmin

def worker(a):
    idx,th0,th1,ph0,ph1,outdir=a
    shard=f"{outdir}/dg_{idx}.jsonl"; done=f"{outdir}/dg_{idx}.done"
    if os.path.exists(done): return (idx,'skip',0,0,0,9.0)
    import ball_cover5 as B5
    if not B5.in_F((th0+th1)/2,(th1-th0)/2,(ph0+ph1)/2,(ph1-ph0)/2):
        open(done,'w').write('outsideF\n'); return (idx,'outF',0,0,0,9.0)
    t0=time.time(); c,de,w,sk,dm=sweep_region(th0,th1,ph0,ph1,shard)
    open(done,'w').write(f"cert={c} defer={de} wall={w} dmin={dm} t={time.time()-t0:.0f}\n")
    return (idx,'done',c,de,w,dm)

def main():
    nproc=int(sys.argv[1]); nsub=int(sys.argv[2]); outdir=sys.argv[3]; cap=float(sys.argv[4]) if len(sys.argv)>4 else 0
    os.makedirs(outdir,exist_ok=True)
    th0,th1,ph0,ph1=FBOX; jobs=[]; k=0
    for i in range(nsub):
        a=th0+(th1-th0)*i/nsub; b=th0+(th1-th0)*(i+1)/nsub
        for j in range(nsub):
            c=ph0+(ph1-ph0)*j/nsub; d=ph0+(ph1-ph0)*(j+1)/nsub
            jobs.append((k,a,b,c,d,outdir)); k+=1
    print(f"F split {nsub}x{nsub} = {len(jobs)} sub-regions, nproc={nproc}",flush=True)
    t0=time.time(); C=D=W=0; dm=9.0; nd=0
    with mp.Pool(nproc) as p:
        for r in p.imap_unordered(worker,jobs):
            idx,st,c,de,w,d=r; C+=c; D+=de; W+=w; dm=min(dm,d); nd+=1
            if st in('done','outF'):
                print(f"  region {idx} {st}: cert={c} defer={de} wall={w} — TOTAL cert={C} defer={D} wall={W} dmin={dm:.4f} done={nd}/{len(jobs)} {time.time()-t0:.0f}s",flush=True)
            if cap and time.time()-t0>cap: print("  [cap hit — resumable]"); break
    print(f"\nAGG: cert={C} defer={D} wall={W}  min d_lo={dm:.5f}  regions_done={nd}/{len(jobs)}  {time.time()-t0:.0f}s")
    print("DEPTH GRID COVERED ✓ (cert + critical-disk defer)" if (nd==len(jobs) and W==0) else f"INCOMPLETE: {W} wall cells need the two-case port")
if __name__=='__main__':
    mp.set_start_method('fork'); main()
