"""Interval-delta continuum certificate for the (SB-box) extension (paper Section 15.6).

Replaces the (broken) point-delta + Lipschitz-interpolation argument with a GENUINE interval-delta
sweep: wall_sweep2.sweep_level certifies delta*K_box subset conv F(u2(delta,phi)) over delta-INTERVAL
x phi-INTERVAL cells (two-symbol Taylor forms in t=delta-dc, s=phi-fc; WS-2 dual-weight linear
residual, which stays tight over intervals -- unlike the facet-margin/determinant route). Adaptive
in both delta and phi. Covers delta in [1.28e-3, 5e-2] (meeting the base sweep's ceiling), phi in
[0,pi] (mirror gives [-pi,0]); 0 FAILS => the continuum is certified with no rungs/interpolation.

Usage: python ivd_extension_sweep.py [NPROC]     -> writes ivd_ext_out/SUMMARY.json
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ivd_ext_out')
DLO, DHI = 1.28e-3, 5e-2          # meets base-sweep ceiling (1.28e-3) up to the extension reach

def chunks():
    # finer strips at low delta (higher flag curvature relative to delta), coarser high up
    out = []; d = DLO
    while d < DHI - 1e-12:
        w = 2e-4 if d < 6e-3 else (5e-4 if d < 1.5e-2 else 1e-3)
        out.append((d, min(d + w, DHI))); d += w
    return out

def work(strip):
    import wall_sweep2 as WS
    d1, d2 = strip
    tag = '%.6e_%.6e' % (d1, d2)
    fp = os.path.join(OUT, 'ch_%s.json' % tag)
    if os.path.exists(fp):
        try: return json.load(open(fp))
        except Exception: pass
    t0 = time.time()
    done, nfail, minsl, maxeta, _ = WS.sweep_level(d1, d2, 0.0, math.pi, init_n=32, maxdepth=48)
    rep = dict(d1=d1, d2=d2, cells=len(done), fails=int(nfail),
               minslack=float(minsl), maxeta=float(maxeta), secs=round(time.time()-t0, 1))
    json.dump(rep, open(fp, 'w'))
    return rep

def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, mp.cpu_count() - 2)
    os.makedirs(OUT, exist_ok=True)
    sl = chunks()
    print('%d delta-strips over [%.3g,%.3g], %d procs' % (len(sl), DLO, DHI, nproc), flush=True)
    t0 = time.time(); res = []
    with mp.Pool(nproc) as p:
        for i, r in enumerate(p.imap_unordered(work, sl), 1):
            res.append(r)
            if i % 10 == 0 or i == len(sl):
                tf = sum(x['fails'] for x in res)
                print('  %d/%d strips | fails=%d | %.0fs' % (i, len(sl), tf, time.time()-t0), flush=True)
    fails = sum(x['fails'] for x in res); cells = sum(x['cells'] for x in res)
    minsl = min(x['minslack'] for x in res); maxeta = max(x['maxeta'] for x in res)
    summ = dict(delta_range=[DLO, DHI], phi_range=[0, math.pi], strips=len(sl), cells=cells,
                fails=fails, min_slack=minsl, max_eta=maxeta, secs=round(time.time()-t0, 1),
                verdict=('CERTIFIED interval-delta: 0 FAILS over the continuum' if fails == 0
                         else 'INCOMPLETE: %d fails' % fails))
    json.dump(summ, open(os.path.join(OUT, 'SUMMARY.json'), 'w'), indent=1)
    print('\n' + json.dumps(summ, indent=1), flush=True)

if __name__ == '__main__':
    main()
