"""dlo_cert2: certified depth over a u2-cell = point-interval depth at center  -  Lambda_d * Delta.
Soundness: (i) hull_stable over the WHOLE cell (constant combinatorics, so flags are in bijection);
(ii) depth-perturbation lemma: |r_P(u') - r_P(u)| <= Lambda_d(C)*Delta in the transported frame
(depth frame-invariant), Lambda_d = sqrt(2*(sqrt3+12/L_lo)^2 + 48/L_lo^2), L_lo = certified lower
bound on every silhouette edge length over the cell; (iii) Delta >= max geodesic angle center->cell
point, bounded by thw + phw*sin_hi(th).  d_cert = d_pt.lo - Lambda_d*Delta."""
import math
import numpy as np
import fast_interval as F
from fast_interval import FI
import ball_cover as BC, ball_cover5 as B5
import depth_grid as DG
from fractions import Fraction as Fr
R3=math.sqrt(3.0)

def dlo_cert2(th,thw,ph,phw):
    u2c=BC.u2center(th,ph); eo=BC.edge_order(u2c)
    if eo is None: return None,False,'no-hull'
    box=(th,thw,ph,phw,0,0,0,0,0,0)
    if not B5.hull_stable(box,eo): return None,False,'hull-unstable'
    # point depth at center (degenerate interval -> tight); reuse full-interval routine at width 0
    d_pt,ok,why=DG.dlo_cert(th,0.0,ph,0.0)
    if not ok: return None,False,'pt:'+why
    # certified L_lo over the cell (true orthonormal-frame shadow edge lengths, interval)
    W=F.frame(F.iu2(th-thw,th+thw,ph-phw,ph+phw)); V=F.verts(Fr(11,20))
    q=[F.proj(F.applyW(W,v)) for v in V]
    Llo=None
    for (p_,r_,sgn) in eo:
        d0=q[r_][0]-q[p_][0]; d1=q[r_][1]-q[p_][1]
        L=(d0.sqr()+d1.sqr()).sqrt().lo
        Llo=L if Llo is None else min(Llo,L)
    if Llo<=1e-6: return None,False,'short-edge'
    Lam=math.sqrt(2*(R3+12.0/Llo)**2+48.0/Llo**2)
    # Delta: max geodesic distance from center to a cell point <= thw + phw*max|sin th|
    sh=max(abs(math.sin(th-thw)),abs(math.sin(th+thw)),1.0 if (th-thw)<=math.pi/2<=(th+thw) else 0.0)
    Delta=thw+phw*sh
    d=d_pt- (1.0000001*Lam)*Delta   # tiny safety on Lam float eval
    if d<=0: return None,False,'lip-eats'
    return d,True,'ok(Lam=%.1f)'%Lam
