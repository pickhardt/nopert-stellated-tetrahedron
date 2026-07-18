# SHIM: use the compiled Cython FI (bit-identical, ~5x). Falls back to pure python if unbuilt.
try:
    from fast_interval_cy import (FI, dot, cross, vnorm, normalize, frame, applyW, rodrigues,
        proj, verts, iu2, idir, G_lambda_mv, trig_fi, vsub, E3, R2, PI, _V4)
    import math
    from math import nextafter, inf
    def rd(x): return nextafter(x,-inf)
    def ru(x): return nextafter(x, inf)
    _CY=True
except Exception as _e:
    _CY=False
    exec(open(__file__.replace('fast_interval.py','fast_interval_pure.py')).read())
