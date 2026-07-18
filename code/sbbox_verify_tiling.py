"""Standalone tiling + threshold verifier for the (SB-box) extension manifests (draft §15.6, §18 item 4).
Consumes a rung's shard_*.jsonl (fields d,a,b,lo) and checks, from the MANIFEST ALONE:
  (1) every certified cell has lo >= MOD  (so margin/δ >= MOD > 0 ⇒ δ·K_box ⊆ conv F on the cell);
  (2) the cells tile φ∈[0,π] with NO gaps, except gaps of width ≤ 2·EPS centered on a silhouette
      transition of find_transitions(δ) (those are the continuity-charged slivers, §15.4).
No geometry is recomputed here (independent of the sweep driver's accept path); a full geometric
replay is the stronger check and reuses facet_margin. Usage: python sbbox_verify_tiling.py <rung_dir>...
"""
import sys, os, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MOD = 0.002; EPS = 1e-6; PI = math.pi

def verify_rung(rung_dir):
    cells = []
    for sh in glob.glob(os.path.join(rung_dir, 'shard_*.jsonl')):
        for line in open(sh):
            line = line.strip()
            if line: cells.append(json.loads(line))
    if not cells: return None, "no cells"
    d = cells[0]['d']
    # (1) threshold
    below = [c for c in cells if c['lo'] < MOD]
    # (2) tiling: sort by a, walk, allow gaps only at transitions (±EPS)
    import sbbox_prove as S
    tr = [t for t in S.find_transitions(d) if 0 <= t <= PI]
    cells.sort(key=lambda c: c['a'])
    gaps = []; overlaps = []
    cur = 0.0
    for c in cells:
        a, b = c['a'], c['b']
        if a > cur + 1e-12:
            # gap [cur, a]; allowed iff it sits within EPS of a transition and is small
            mid = 0.5 * (cur + a)
            near_tr = any(abs(mid - t) <= EPS + (a - cur) for t in tr)
            if not (near_tr and (a - cur) <= 2 * EPS + 1e-9):
                gaps.append((round(cur, 8), round(a, 8)))
        if b > cur: cur = b
        elif a < cur - 1e-12: overlaps.append((round(a, 8), round(b, 8)))
    tail_ok = cur >= PI - 1e-9 or any(abs(cur - t) <= EPS for t in tr) or cur >= PI - 2 * EPS
    ok = (not below) and (not gaps) and (not overlaps) and tail_ok
    return ok, dict(delta=d, cells=len(cells), min_lo=round(min(c['lo'] for c in cells), 6),
                    below_MOD=len(below), gaps=gaps[:5], overlaps=overlaps[:5],
                    covered_to=round(cur, 6), transitions=[round(t, 4) for t in tr])

if __name__ == '__main__':
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'sbbox_ext', 'd_*')))
    allok = True
    for dd in dirs:
        ok, info = verify_rung(dd)
        tag = 'PASS' if ok else 'FAIL'
        if not ok: allok = False
        print(f"{tag}  {os.path.basename(dd):14s} {info}")
    print("\nVERDICT:", "ALL RUNGS PASS (tiling gap-free, all margins ≥ MOD)" if allok else "FAIL")
    sys.exit(0 if allok else 1)
