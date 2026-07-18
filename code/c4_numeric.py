"""Numeric measurement of the order-4 remainder constant C4:
   sup |h_true(delta, xhat) - P3(delta, xhat)| / delta^4
over all tight rows, charts, phi samples, xhat in the V-box, delta in {1e-3, 5e-4}.
h_true: exact constraint (full rotation exponentials, true unit normal).
P3: the exact Taylor tables (sf_rowbounds_sym.pkl) evaluated at (phi, xhat).
"""
import sys, numpy as np, itertools, pickle
sys.path.insert(0,'certificates')
import sf_preflight as SF

a_st = 11/20
vs = np.array([(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)], float)
V = np.vstack([vs, -a_st*vs])
names = ['v1','v2','v3','v4','p1','p2','p3','p4']
s2 = np.sqrt(2)
Wst = np.array([[0,0,1],[-1/s2,-1/s2,0],[1/s2,-1/s2,0]])
wh = (Wst@V.T).T   # frame coords
CONFIGS = {'pentA':['v1','v4','p1','v2','p4'],
           'hex1' :['v1','v4','p1','v3','v2','p4'],
           'pentB':['v1','v4','p1','v3','p4']}
def skew(a):
    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
def rot(K):
    # exact rotation exp(K) for skew K via Rodrigues
    w = np.array([K[2,1], K[0,2], K[1,0]]); th = np.linalg.norm(w)
    if th < 1e-300: return np.eye(3)
    A = K/th
    return np.eye(3) + np.sin(th)*A + (1-np.cos(th))*(A@A)

def h_true_rows(cfg, phi, xh, d):
    """true normalized constraint values for all tight rows of cfg at (phi,xh,delta)."""
    ah = np.array([-np.sin(phi), np.cos(phi), 0.0])
    M = rot(skew(-d*ah))                    # engine M = exp(-d A)
    xi = np.array([-xh[1], xh[0], xh[2]])*d
    R = rot(skew(xi))
    RM = R@M
    outer = (M@wh.T).T[:, :2]
    inner = (RM@wh.T).T[:, :2] + d*np.array([xh[3], xh[4]])
    hull = CONFIGS[cfg]; idx = [names.index(nm) for nm in hull]
    J2 = np.array([[0,-1.],[1,0]])
    out = {}
    for k in range(len(idx)):
        a_i, b_i = idx[k], idx[(k+1)%len(idx)]
        qa, qb = outer[a_i], outer[b_i]
        nt = J2@(qb-qa); nn = np.linalg.norm(nt)
        nh = nt/nn
        for j in range(8):
            out[(f"{hull[k]}-{hull[(k+1)%len(hull)]}", names[j])] = nh@(inner[j]-qa)
    return out

def measure(deltas=(1e-3, 5e-4), nphi=7, seed=0):
    rng = np.random.default_rng(seed)
    PHI_W = float(np.arctan(11*np.sqrt(2)/9))
    ranges = {'pentA': (1e-4, PHI_W-1e-4), 'hex1': (PHI_W+1e-4, np.pi-PHI_W-1e-4),
              'pentB': (np.pi-PHI_W+1e-4, np.pi-1e-4)}
    BX = np.array([2.39, 2.39, 0.0285, 0.0285, 0.0285])
    xh_samples = [np.array(sg)*BX for sg in itertools.product([-1,1],repeat=5)]
    xh_samples += [rng.uniform(-1,1,5)*BX for _ in range(20)] + [np.zeros(5)]
    worst = {}; worst_row = {}
    for cfg in CONFIGS:
        ev = SF.EV[cfg]
        lo, hi = ranges[cfg]
        for phi in np.linspace(lo, hi, nphi):
            dat = ev.eval_at(phi)
            for d in deltas:
                ht = {}
                for xh in xh_samples:
                    hrows = h_true_rows(cfg, phi, xh, d)
                    for k, lab in enumerate(ev.labels):
                        e, j = lab.split('|')
                        v = hrows[(e,j)]
                        f = dat['drift'][k] + dat['G'][k]@xh
                        g = dat['g0'][k] + dat['gl'][k]@xh + xh@dat['gq'][k]@xh
                        h3 = 0.0
                        for m, co in dat['h3'][k].items():
                            t = co
                            for i,p in enumerate(m): t *= xh[i]**p
                            h3 += t
                        P3 = d*f + d*d*g + d**3*h3
                        # resolve sign convention: engine rows have folded sign
                        r1 = abs(v - P3); r2 = abs(-v - P3)
                        rem = min(r1, r2)/d**4
                        key = (cfg,)
                        if rem > worst.get(key, 0):
                            worst[key] = rem; worst_row[key] = (lab, phi, d, float(rem))
    return worst, worst_row

if __name__ == '__main__':
    worst, worst_row = measure()
    for k in worst:
        print(k, 'C4_est = %.2f' % worst[k], ' at', worst_row[k])
