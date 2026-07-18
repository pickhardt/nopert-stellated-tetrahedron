//! Rigorous float64 interval arithmetic with directed (outward) rounding.
//! Ports certificates/fast_interval_pure.py operation-for-operation. Every endpoint is
//! pushed outward one ULP after each op, so a positive `lo` remains a rigorous proof.
//!
//! rd/ru use next_down()/next_up(), the exact analogues of Python's math.nextafter(x,-/+inf).
//! Verified bit-identical to the Python kernel for +,-,*,/,sqrt on this platform; trig only
//! needs outward-containment (libm cos/sin may differ in the last ULP across platforms, which
//! the outward pad + rounding absorbs soundly).

#[inline(always)]
pub fn rd(x: f64) -> f64 { x.next_down() }
#[inline(always)]
pub fn ru(x: f64) -> f64 { x.next_up() }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fi {
    pub lo: f64,
    pub hi: f64,
}

impl Fi {
    #[inline(always)]
    pub fn new(lo: f64, hi: f64) -> Fi { Fi { lo, hi } }
    #[inline(always)]
    pub fn pt(x: f64) -> Fi { Fi { lo: x, hi: x } }

    #[inline(always)]
    pub fn add(a: Fi, b: Fi) -> Fi { Fi::new(rd(a.lo + b.lo), ru(a.hi + b.hi)) }

    #[inline(always)]
    pub fn sub(a: Fi, b: Fi) -> Fi { Fi::new(rd(a.lo - b.hi), ru(a.hi - b.lo)) }

    #[inline(always)]
    pub fn neg(a: Fi) -> Fi { Fi::new(-a.hi, -a.lo) }

    pub fn mul(a: Fi, b: Fi) -> Fi {
        // mirror the exact if-comparison sequence in fast_interval_pure.py (not min/max builtins)
        let p0 = a.lo * b.lo;
        let p1 = a.lo * b.hi;
        let p2 = a.hi * b.lo;
        let p3 = a.hi * b.hi;
        let mut lo = p0;
        let mut hi = p0;
        if p1 < lo { lo = p1; }
        if p1 > hi { hi = p1; }
        if p2 < lo { lo = p2; }
        if p2 > hi { hi = p2; }
        if p3 < lo { lo = p3; }
        if p3 > hi { hi = p3; }
        Fi::new(rd(lo), ru(hi))
    }

    /// Division; caller must ensure 0 is not in `b` (Python raises ZeroDivisionError).
    pub fn div(a: Fi, b: Fi) -> Fi {
        let p0 = a.lo / b.lo;
        let p1 = a.lo / b.hi;
        let p2 = a.hi / b.lo;
        let p3 = a.hi / b.hi;
        let lo = p0.min(p1).min(p2).min(p3);
        let hi = p0.max(p1).max(p2).max(p3);
        Fi::new(rd(lo), ru(hi))
    }

    pub fn sqr(a: Fi) -> Fi {
        if a.lo >= 0.0 {
            Fi::new(rd(a.lo * a.lo), ru(a.hi * a.hi))
        } else if a.hi <= 0.0 {
            Fi::new(rd(a.hi * a.hi), ru(a.lo * a.lo))
        } else {
            let m = (a.lo * a.lo).max(a.hi * a.hi);
            Fi::new(0.0, ru(m))
        }
    }

    pub fn sqrt(a: Fi) -> Fi {
        let lo = if a.lo > 0.0 { rd(a.lo.sqrt()) } else { 0.0 };
        let hi = if a.hi > 0.0 { ru(a.hi.sqrt()) } else { 0.0 };
        Fi::new(lo, hi)
    }
}

const PI: f64 = std::f64::consts::PI;

/// Interval enclosures of (cos, sin) over the angle range [lo, hi].
/// Ports trig_fi in fast_interval_pure.py: pad outward 1e-15, then take min/max over the
/// endpoints plus any interior extrema (k*PI for cos, PI/2+k*PI for sin) inside the range.
pub fn trig_fi(lo_in: f64, hi_in: f64) -> (Fi, Fi) {
    let lo = rd(lo_in - 1e-15);
    let hi = ru(hi_in + 1e-15);

    // crange(a,b,fn,base,ext_val): fn = cos or sin; extrema at base+k*PI with value (+/-)ext_val.
    fn crange(a: f64, b: f64, is_cos: bool, base: f64, ext_val: f64) -> (f64, f64) {
        let f = |x: f64| if is_cos { x.cos() } else { x.sin() };
        // vals = [f(a), f(b), extrema...]; result = (min(vals), max(vals))
        let fa = f(a);
        let fb = f(b);
        let mut lo = fa.min(fb);
        let mut hi = fa.max(fb);
        // extrema base+k*PI in (a,b]: k from ceil((a-base)/PI)
        let mut k = ((a - base) / PI).ceil() as i64;
        while base + (k as f64) * PI <= b {
            let val = if k.rem_euclid(2) == 0 { ext_val } else { -ext_val };
            if val < lo { lo = val; }
            if val > hi { hi = val; }
            k += 1;
        }
        (lo, hi)
    }

    let (cl, ch) = crange(lo, hi, true, 0.0, 1.0); // cos extrema at k*PI: +1,-1
    let (sl, sh) = crange(lo, hi, false, PI / 2.0, 1.0); // sin extrema at PI/2+k*PI
    (Fi::new(rd(cl), ru(ch)), Fi::new(rd(sl), ru(sh)))
}
