"""Tighter certified depth d_lo for the DK corner-skip. dlo_window2 = max of:
  (a) the FULL float-hull family with any marginal side-test that matches a WD-1-PROVEN exempt
      (EXEMPT_A/B/A2/B2) exempted -- SOUND: those exempts are proven (Lemma WD-1 antisymmetry),
      and dlo_family strictly validates every OTHER test over the window; if any NON-known test
      fails we drop to None (never exempt an unproven test);
  (b) rc_dlo's conservative two-case (always valid, incl. windows straddling the wall).
Both are valid certified lower bounds; the max is sound and tighter. (a) validates and gives a
~30x tighter d_lo everywhere except a razor-thin |dphi| < ~0.002 short-edge-degenerate strip at
the wall, where it drops to (b). Use in BOTH the grid generator and the verifier re-certification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rc_dlo
from rc_dlo import side_slacks_I, shadow_I
from scipy.spatial import ConvexHull

_KNOWN_EXEMPT = rc_dlo.EXEMPT_A | rc_dlo.EXEMPT_B | rc_dlo.EXEMPT_A2 | rc_dlo.EXEMPT_B2

def dlo_window2(dl, ph):
    dc, pc = 0.5*(dl.lo+dl.hi), 0.5*(ph.lo+ph.hi)
    qf, zf = rc_dlo.shadow_f(dc, pc)
    cyc = list(ConvexHull(qf).vertices)
    edges = [(cyc[k], cyc[(k+1) % len(cyc)]) for k in range(len(cyc))]
    # (a) full hull with only WD-PROVEN exempts applied to its marginal failing tests
    q, z = shadow_I(dl, ph)
    fails = [key for key, s in side_slacks_I(q, edges) if not s.lo > 0]
    ex = set(f for f in fails if f in _KNOWN_EXEMPT)
    unknown = [f for f in fails if f not in _KNOWN_EXEMPT]
    df = rc_dlo.dlo_family(dl, ph, edges, ex) if not unknown else None
    # (b) conservative two-case (valid near/through the wall)
    dw = rc_dlo.dlo_window(dl, ph)
    cands = [x for x in (df, dw) if x is not None]
    return max(cands) if cands else None
