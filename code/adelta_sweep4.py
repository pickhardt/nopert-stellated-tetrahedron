"""adelta_sweep4.py -- CERTIFIED sup A_delta over delta in [1.5e-3,1.5e-2], phi in [0,pi], off the
+-TRW transition windows.  Unified regime, structured recursion (bounded, no runaway):
  * coarse cells (dhw>DMV or phw>PMV): direct interval jet adelta_jet.cell_Adelta (tight in benign);
  * fine cells: BIVARIATE mean-value adelta_mv2.cell_Adelta_mv2 (handles the low-delta pentA/pentB
    short edge AND the transition tubes, where the direct jet's cancellation is unresolvable).
Subdivide toward the mean-value floor (DFLOOR,PFLOOR); accept when bound<=TARGET, or <=CEIL at floor;
the +-TRW window around a transition is excluded (paper's +-1e-6).  A_delta=max over accepted cells.
Edge-set corner-validation makes each cell sound.  Shardable over delta for overnight.
Usage: python adelta_sweep4.py DLO DHI OUT.json
"""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adelta_jet as AJ
import adelta_mv2 as M2
import sbbox_prove as S

TARGET = 8.0; CEIL = 16.0
DMV = 1e-6; PMV = 1e-3          # below this -> bivariate mean-value
DFLOOR = 3e-8; PFLOOR = 3e-5    # subdivision floors
TRW = 1e-6

def _valid(dlo, dhi, plo, phi):
    sm = AJ.flag_struct(0.5*(dlo+dhi), 0.5*(plo+phi))
    if sm is None: return None
    base = frozenset(sm[0])
    for (dd, pp) in ((dlo, plo), (dlo, phi), (dhi, plo), (dhi, phi)):
        sc = AJ.flag_struct(dd, pp)
        if sc is None or frozenset(sc[0]) != base: return None
    return sm[0]

def _near_trans(plo, phi, trs):
    return any(plo-TRW <= t <= phi+TRW for t in trs)

def process(dlo, dhi, plo, phi, trs, st):
    st['cells'] += 1
    struct = _valid(dlo, dhi, plo, phi)
    dhw, phw = 0.5*(dhi-dlo), 0.5*(phi-plo)
    # STRADDLES a transition (edge-set change, incl. short-edge merges find_transitions misses):
    # isolate it in phi down to the +-TRW window, then exclude (paper's +-1e-6 transition window).
    if struct is None:
        if phw <= TRW:
            st['excl'] += 1; return
        pm = 0.5*(plo+phi); process(dlo, dhi, plo, pm, trs, st); process(dlo, dhi, pm, phi, trs, st); return
    if dhw <= DMV and phw <= PMV:
        e = M2.cell_Adelta_mv2(dlo, dhi, plo, phi, struct)
    else:
        e = AJ.cell_Adelta(dlo, dhi, plo, phi, struct)
    if e is not None and e <= TARGET:
        if e > st['max']: st['max'] = e; st['loc'] = (0.5*(dlo+dhi), 0.5*(plo+phi))
        st['acc'] += 1; return
    if dhw <= DFLOOR and phw <= PFLOOR:
        if e is not None and e <= CEIL:
            if e > st['max']: st['max'] = e; st['loc'] = (0.5*(dlo+dhi), 0.5*(plo+phi))
            st['acc'] += 1; return
        st['unres'].append((dlo, dhi, plo, phi, e)); return
    rd = dhw/DFLOOR; rp = phw/PFLOOR
    if rp >= rd and phw > PFLOOR:
        pm = 0.5*(plo+phi); process(dlo, dhi, plo, pm, trs, st); process(dlo, dhi, pm, phi, trs, st)
    elif dhw > DFLOOR:
        dm = 0.5*(dlo+dhi); process(dlo, dm, plo, phi, trs, st); process(dm, dhi, plo, phi, trs, st)
    else:
        pm = 0.5*(plo+phi); process(dlo, dhi, plo, pm, trs, st); process(dlo, dhi, pm, phi, trs, st)

def sweep(dlo, dhi):
    st = dict(max=0.0, loc=None, acc=0, excl=0, cells=0, unres=[])
    trs = sorted(set(round(t, 9) for d in (dlo, dhi) for t in S.find_transitions(d) if 0 <= t <= math.pi))
    edges = sorted({0.0, math.pi} | {t + s*o for t in trs for s in (-1, 1)
                                     for o in (2e-2, 4e-3, 8e-4, 1.6e-4, 3e-5, TRW)} )
    edges = [x for x in edges if 0 <= x <= math.pi]
    edges = sorted(set(edges) | set(i*math.pi/32 for i in range(33)))
    edges = sorted(x for x in edges if 0 <= x <= math.pi)
    for i in range(len(edges)-1):
        if edges[i+1]-edges[i] > 1e-12:
            process(dlo, dhi, edges[i], edges[i+1], trs, st)
    return st

if __name__ == '__main__':
    dlo, dhi = float(sys.argv[1]), float(sys.argv[2])
    outp = sys.argv[3] if len(sys.argv) > 3 else None
    st = sweep(dlo, dhi)
    rep = dict(dlo=dlo, dhi=dhi, max=st['max'], loc=st['loc'], acc=st['acc'], cells=st['cells'],
               excl=st['excl'], unres=len(st['unres']), unres_ex=st['unres'][:3])
    print(json.dumps(rep))
    if outp: json.dump(rep, open(outp, 'w'))
