"""r82: second-order-tight rigorous bounds for u(th,ph) . u*, u* = (1,1,0)/sqrt2.

LEMMA (exact identity): with u(th,ph) = (sin th cos ph, sin th sin ph, cos th),
u* = u(pi/2, pi/4), and t := th - pi/2, p := ph - pi/4,
    u(th,ph) . u* = sin(th) cos(ph - pi/4) = cos(t) cos(p).
Proof: sin th = cos(th - pi/2); cos(ph-pi/4) = (cos ph + sin ph)/sqrt2; multiply out.

Bounds (classical alternating series, valid for x^2 <= 12):
    1 - x^2/2  <=  cos x  <=  1 - x^2/2 + x^4/24,
and 1 - x^2/2 + x^4/24 is decreasing in x on [0, sqrt(6)].
On the sphere grid |t| <= pi/2 + slack, |p| <= 3pi/4; cos of both is handled by
sign-guards below. All arithmetic outward-rounded FI; pi/2, pi/4 enclosed to 1 ulp.

dot_us_bounds(th,thw,ph,phw) -> (lo, hi): rigorous bounds of u . u* over the
closed cell [th-thw, th+thw] x [ph-phw, ph+phw] (enclosed outward). Clamps to
[-1, 1] (sound: both are unit vectors). hi < 1 strictly iff the cell provably
avoids u*, which is what the sbbox citation needs.
"""
import math
from fast_interval import FI, rd, ru

PI_HALF = FI(rd(math.pi/2), ru(math.pi/2))
PI_QUART = FI(rd(math.pi/4), ru(math.pi/4))

def _absrange(I):
    if I.lo <= 0.0 <= I.hi:
        amin = 0.0
    else:
        amin = min(abs(I.lo), abs(I.hi))
    return amin, max(abs(I.lo), abs(I.hi))

def _cos_lo(xup):
    # cos x >= 1 - x^2/2 for all real x
    return max((FI(1.0) - FI(xup).sqr()*FI(0.5)).lo, -1.0)

def _cos_hi(xdn):
    # for |x| >= xdn with xdn in [0, sqrt(6)]: cos x <= 1 - xdn^2/2 + xdn^4/24
    x = FI(xdn)
    return min((FI(1.0) - x.sqr()*FI(0.5) + x.sqr().sqr()*FI(1.0/24.0)).hi, 1.0)

def dot_us_bounds(th, thw, ph, phw):
    tI = FI((FI(th) - FI(thw)).lo, (FI(th) + FI(thw)).hi) - PI_HALF
    pI = FI((FI(ph) - FI(phw)).lo, (FI(ph) + FI(phw)).hi) - PI_QUART
    tmin, tmax = _absrange(tI)
    pmin, pmax = _absrange(pI)
    if tmin > 2.4 or pmin > 2.4:          # outside series-monotone domain for the hi bound
        return -1.0, 1.0                  # (never the citation regime; sound trivial bounds)
    clo_t = _cos_lo(tmax); clo_p = _cos_lo(pmax)
    if clo_t < 0.0 or clo_p < 0.0:
        lo = -1.0
    else:
        lo = max((FI(clo_t) * FI(clo_p)).lo, -1.0)
    hi = min((FI(_cos_hi(tmin)) * FI(_cos_hi(pmin))).hi, 1.0)
    return lo, hi
