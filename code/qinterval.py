"""Exact rational interval arithmetic + evaluator for sympy expressions in (c,s).
All endpoints are Fractions; sqrt enclosures via integer isqrt. NO floating point.
Sound for expression trees built from: Rational, Add, Mul, Pow(expr, int),
Pow(rational, +-1/2), sqrt of nonneg interval, symbols c,s (given boxes).
"""
from fractions import Fraction as Fr
from math import isqrt
import sympy as sp

S = 10**30  # sqrt enclosure precision

def sqrt_iv(lo, hi):
    assert lo >= 0
    # sqrt(p/q) = sqrt(p*q)/q
    def lo_s(x):
        p,q = x.numerator, x.denominator
        n = isqrt(p*q*S*S)
        return Fr(n, q*S)
    def hi_s(x):
        p,q = x.numerator, x.denominator
        n = isqrt(p*q*S*S)
        if Fr(n,q*S)**2 < x: n += 1
        return Fr(n, q*S)
    return (lo_s(lo), hi_s(hi))

def add(a,b): return (a[0]+b[0], a[1]+b[1])
def neg(a): return (-a[1], -a[0])
def mul(a,b):
    ps = (a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1])
    return (min(ps), max(ps))
def inv(a):
    assert a[0] > 0 or a[1] < 0, "interval contains 0"
    return (1/a[1], 1/a[0])
def ipow(a, n):
    if n == 0: return (Fr(1), Fr(1))
    if n < 0: return ipow(inv(a), -n)
    r = (Fr(1), Fr(1))
    for _ in range(n): r = mul(r, a)
    if n % 2 == 0 and a[0] < 0 < a[1]:
        r = (Fr(0), r[1])   # even power: tighten lower to 0
    return r
def iabs(a):
    if a[0] >= 0: return a
    if a[1] <= 0: return neg(a)
    return (Fr(0), max(-a[0], a[1]))
def hull(a,b): return (min(a[0],b[0]), max(a[1],b[1]))

def eval_iv(e, env):
    """env: dict sympy symbol -> (Fr lo, Fr hi)."""
    if e.is_Rational:
        f = Fr(e.p, e.q); return (f, f)
    if e.is_Symbol:
        return env[e]
    if e.is_Add:
        r = (Fr(0), Fr(0))
        for a in e.args: r = add(r, eval_iv(a, env))
        return r
    if e.is_Mul:
        r = (Fr(1), Fr(1))
        for a in e.args: r = mul(r, eval_iv(a, env))
        return r
    if e.is_Pow:
        b, ex = e.args
        if ex.is_Integer:
            return ipow(eval_iv(b, env), int(ex))
        if ex == sp.Rational(1,2):
            bl, bh = eval_iv(b, env); assert bl >= 0
            return sqrt_iv(bl, bh)
        if ex == sp.Rational(-1,2):
            bl, bh = eval_iv(b, env); assert bl > 0
            return inv(sqrt_iv(bl, bh))
        raise ValueError(f"pow {e}")
    raise ValueError(f"node {e}")

# ---- chart boxes for (c,s) = (cos phi, sin phi) ----
# cos(phi_w) = 9/sqrt(323), sin(phi_w) = 11*sqrt(2)/sqrt(323)
CWl, CWh = sqrt_iv(Fr(81,323), Fr(81,323))     # enclosure of 9/sqrt323
SWl, SWh = sqrt_iv(Fr(242,323), Fr(242,323))   # enclosure of 11sqrt2/sqrt323

def chart_box(cfg, csym, ssym):
    if cfg == 'pentA':   # phi in [0, phi_w]  (mirror covers negatives)
        return {csym: (CWl, Fr(1)), ssym: (Fr(0), SWh)}
    if cfg == 'hex1':    # phi in [phi_w, pi - phi_w]
        return {csym: (-CWh, CWh), ssym: (SWl, Fr(1))}
    if cfg == 'pentB':   # phi in [pi - phi_w, pi]
        return {csym: (Fr(-1), -CWl), ssym: (Fr(0), SWh)}
    raise ValueError(cfg)

def split_box(box, sym, n):
    lo, hi = box[sym]; step = (hi-lo)/n
    out=[]
    for i in range(n):
        b = dict(box); b[sym] = (lo+i*step, lo+(i+1)*step)
        out.append(b)
    return out
