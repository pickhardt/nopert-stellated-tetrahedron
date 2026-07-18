//! Degree-2 Taylor model in N=5 shared symbols, outward-rounded.
//! Ports certificates/tm2.py operation-for-operation (incl. _usum sequential rounding, _gap
//! upward rounding of every op's round-off, to_FI's eps^2 in [0,1] sharpening, and the
//! nonneg=true sum-of-squares floor in _series). Bit-identity vs Python tm2 is the gate.

use crate::interval::{rd, ru, Fi, trig_fi};

pub const N: usize = 5;
// 15 pairs (i,j) with i<=j, i,j in 0..5, in the same order as tm2.py _IDX
pub const NQ: usize = 15;
pub const IDX: [(usize, usize); NQ] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 2), (2, 3), (2, 4),
    (3, 3), (3, 4),
    (4, 4),
];
// position of diagonal (i,i) in IDX: 0,5,9,12,14
const DIAG_POS: [usize; N] = [0, 5, 9, 12, 14];

#[inline(always)]
fn gap(z: f64) -> f64 { ru(ru(z) - rd(z)) }

/// _usum: s=0; for x in xs: s = ru(s + x). Order-sensitive; must match Python exactly.
#[inline]
fn usum(xs: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for &x in xs { s = ru(s + x); }
    s
}

#[derive(Clone)]
pub struct Tm2 {
    pub c: f64,
    pub l: [f64; N],
    pub q: [f64; NQ],
    pub r: f64,
}

impl Tm2 {
    pub fn cst(x: f64) -> Tm2 { Tm2 { c: x, l: [0.0; N], q: [0.0; NQ], r: 0.0 } }

    pub fn sym(c: f64, k: usize, hw: f64) -> Tm2 {
        let mut l = [0.0; N];
        l[k] = hw;
        Tm2 { c, l, q: [0.0; NQ], r: 0.0 }
    }

    pub fn from_fi(fi: Fi) -> Tm2 {
        let c = 0.5 * (fi.lo + fi.hi);
        let r = ru(ru(fi.hi - c).max(ru(c - fi.lo)));
        Tm2 { c, l: [0.0; N], q: [0.0; NQ], r }
    }

    fn bl(&self) -> f64 {
        let mut t = [0.0f64; N];
        for i in 0..N { t[i] = self.l[i].abs(); }
        usum(&t)
    }
    fn bq(&self) -> f64 {
        let mut t = [0.0f64; NQ];
        for i in 0..NQ { t[i] = self.q[i].abs(); }
        usum(&t)
    }
    pub fn rad(&self) -> f64 { usum(&[self.bl(), self.bq(), self.r]) }

    pub fn to_fi(&self) -> Fi {
        // sharp on the quadratic diagonal: eps^2 in [0,1]
        let bl = self.bl();
        let mut qpos = 0.0f64;
        let mut qneg = 0.0f64;
        let mut qcross = 0.0f64;
        for k in 0..NQ {
            let (i, j) = IDX[k];
            let v = self.q[k];
            if i == j {
                if v > 0.0 { qpos = ru(qpos + v); } else { qneg = ru(qneg - v); }
            } else {
                qcross = ru(qcross + v.abs());
            }
        }
        let hi = ru(self.c + usum(&[bl, qpos, qcross, self.r]));
        let lo = rd(self.c - usum(&[bl, qneg, qcross, self.r]));
        Fi::new(lo, hi)
    }
    pub fn hi(&self) -> f64 { self.to_fi().hi }
    pub fn lo(&self) -> f64 { self.to_fi().lo }

    pub fn neg(&self) -> Tm2 {
        let mut l = [0.0; N]; let mut q = [0.0; NQ];
        for i in 0..N { l[i] = -self.l[i]; }
        for i in 0..NQ { q[i] = -self.q[i]; }
        Tm2 { c: -self.c, l, q, r: self.r }
    }

    pub fn add_scalar(&self, k: f64) -> Tm2 {
        let c = self.c + k;
        Tm2 { c, l: self.l, q: self.q, r: usum(&[self.r, gap(c)]) }
    }

    pub fn add(&self, b: &Tm2) -> Tm2 {
        let c = self.c + b.c;
        let mut l = [0.0; N]; let mut q = [0.0; NQ];
        // r = _usum((a.r, b.r, gap(c)) + gaps(l sums) + gaps(q sums))
        let mut gaps: Vec<f64> = Vec::with_capacity(3 + N + NQ);
        gaps.push(self.r); gaps.push(b.r); gaps.push(gap(c));
        for i in 0..N { l[i] = self.l[i] + b.l[i]; gaps.push(gap(self.l[i] + b.l[i])); }
        for i in 0..NQ { q[i] = self.q[i] + b.q[i]; gaps.push(gap(self.q[i] + b.q[i])); }
        Tm2 { c, l, q, r: usum(&gaps) }
    }

    pub fn sub(&self, b: &Tm2) -> Tm2 { self.add(&b.neg()) }

    pub fn mul_scalar(&self, k: f64) -> Tm2 {
        let c = self.c * k;
        let mut l = [0.0; N]; let mut q = [0.0; NQ];
        let mut gaps: Vec<f64> = Vec::with_capacity(2 + N + NQ);
        gaps.push(ru(k.abs() * self.r));
        gaps.push(gap(c));
        for i in 0..N { l[i] = self.l[i] * k; gaps.push(gap(self.l[i] * k)); }
        for i in 0..NQ { q[i] = self.q[i] * k; gaps.push(gap(self.q[i] * k)); }
        Tm2 { c, l, q, r: usum(&gaps) }
    }

    pub fn mul(&self, b: &Tm2) -> Tm2 {
        let ca = self.c; let cb = b.c;
        let c = ca * cb;
        let mut l = [0.0f64; N];
        let mut q = [0.0f64; NQ];
        let mut gaps: Vec<f64> = Vec::new();
        gaps.push(gap(c));
        for i in 0..N {
            let v = ca * b.l[i] + cb * self.l[i];
            l[i] = v;
            gaps.push(gap(ca * b.l[i]));
            gaps.push(gap(cb * self.l[i]));
            gaps.push(gap(v));
        }
        for k in 0..NQ {
            let (i, j) = IDX[k];
            let mut v = ca * b.q[k] + cb * self.q[k];
            gaps.push(gap(ca * b.q[k]));
            gaps.push(gap(cb * self.q[k]));
            gaps.push(gap(v));
            if i == j {
                let w = self.l[i] * b.l[i];
                v += w;
                gaps.push(gap(w));
            } else {
                let w1 = self.l[i] * b.l[j];
                let w2 = self.l[j] * b.l[i];
                v += w1 + w2;
                gaps.push(gap(w1));
                gaps.push(gap(w2));
                gaps.push(gap(w1 + w2));
            }
            q[k] = v;
            gaps.push(gap(v));
        }
        // remainder: degree>=3 products + cross-remainder
        let bla = self.bl(); let bqa = self.bq(); let ra = self.r;
        let blb = b.bl(); let bqb = b.bq(); let rb = b.r;
        let maga = usum(&[ca.abs(), bla, bqa, ra]);
        let magb = usum(&[cb.abs(), blb, bqb, rb]);
        let rr = usum(&[ru(bla * bqb), ru(blb * bqa), ru(bqa * bqb),
                        ru(ra * magb), ru(rb * maga)]);
        let mut allr: Vec<f64> = Vec::with_capacity(1 + gaps.len());
        allr.push(rr);
        allr.extend_from_slice(&gaps);
        Tm2 { c, l, q, r: usum(&allr) }
    }

    /// f(x) enclosure via degree-2 Taylor at center with Lagrange R3.
    /// f0,f1,f2 = FI enclosures of f(xc), f'(xc), f''(xc)/2; d3max >= sup|f'''|/6.
    fn taylor2(&self, f0: Fi, f1: Fi, f2: Fi, d3max: f64) -> Tm2 {
        // u = deviation from center (same L,Q,r, c=0)
        let u = Tm2 { c: 0.0, l: self.l, q: self.q, r: self.r };
        let u2 = u.mul(&u);
        let f1c = 0.5 * (f1.lo + f1.hi);
        let f1r = ru(ru(f1.hi - f1c).max(ru(f1c - f1.lo)));
        let f2c = 0.5 * (f2.lo + f2.hi);
        let f2r = ru(ru(f2.hi - f2c).max(ru(f2c - f2.lo)));
        let f0c = 0.5 * (f0.lo + f0.hi);
        let f0r = ru(ru(f0.hi - f0c).max(ru(f0c - f0.lo)));
        let radu = u.rad();
        // out = u*f1c + u2*f2c + f0c
        let out = u.mul_scalar(f1c).add(&u2.mul_scalar(f2c)).add_scalar(f0c);
        let extra = usum(&[f0r, ru(f1r * radu), ru(f2r * ru(radu * radu)),
                           ru(d3max * ru(ru(radu * radu) * radu))]);
        Tm2 { c: out.c, l: out.l, q: out.q, r: usum(&[out.r, extra]) }
    }

    pub fn sin(&self) -> Tm2 {
        let (cc, ss) = trig_fi(self.c, self.c); // (cos, sin) at center
        let f0 = ss;
        let f1 = cc;
        let f2 = Fi::mul(Fi::sub(Fi::pt(0.0), ss), Fi::pt(0.5)); // -sin/2
        self.taylor2(f0, f1, f2, ru(1.0 / 6.0))
    }
    pub fn cos(&self) -> Tm2 {
        let (cc, ss) = trig_fi(self.c, self.c);
        let f0 = cc;
        let f1 = Fi::sub(Fi::pt(0.0), ss); // -sin
        let f2 = Fi::mul(Fi::sub(Fi::pt(0.0), cc), Fi::pt(0.5)); // -cos/2
        self.taylor2(f0, f1, f2, ru(1.0 / 6.0))
    }
}

// ---- Rodrigues series (polynomial in TM2; alternating tails), nonneg z = |eta|^2 ----
const FCOEF: [f64; 5] = [1.0, -1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0, 1.0 / 362880.0];
const GCOEF: [f64; 5] = [0.5, -1.0 / 24.0, 1.0 / 720.0, -1.0 / 40320.0, 1.0 / 3628800.0];
const FTAIL: f64 = 1.0 / 39916800.0;
const GTAIL: f64 = 1.0 / 479001600.0;

fn series(z: &Tm2, coef: &[f64; 5], tailc: f64) -> Tm2 {
    let zfi = z.to_fi();
    // nonneg=true (z is a sum of squares): floor lo at 0 (sound; tightens a provable fact)
    let lo_eff = if zfi.lo < 0.0 { 0.0 } else { zfi.lo };
    assert!(zfi.hi <= 1.0 && lo_eff >= -1e-9, "series domain guard");
    // Horner: acc = coef[4]; for c in coef[3..0]: acc = acc*z + c
    let mut acc = Tm2::cst(coef[4]);
    for i in (0..4).rev() {
        acc = acc.mul(z).add_scalar(coef[i]);
    }
    let zm = lo_eff.abs().max(zfi.hi.abs());
    let tail = ru(1.1 * ru(tailc * ru(zm.powi(5))));
    Tm2 { c: acc.c, l: acc.l, q: acc.q, r: usum(&[acc.r, tail]) }
}

pub fn sinc_sq(z: &Tm2) -> Tm2 { series(z, &FCOEF, FTAIL) }
pub fn oneminuscos_over(z: &Tm2) -> Tm2 { series(z, &GCOEF, GTAIL) }

/// Rodrigues rotation matrix from axis-angle vector eta (3 TM2s). Returns 3x3 of TM2.
pub fn rot_from_vec(eta: &[Tm2; 3]) -> [[Tm2; 3]; 3] {
    let z = eta[0].mul(&eta[0]).add(&eta[1].mul(&eta[1])).add(&eta[2].mul(&eta[2]));
    let f = sinc_sq(&z);
    let g = oneminuscos_over(&z);
    let (ex, ey, ez) = (&eta[0], &eta[1], &eta[2]);
    let outer: [[Tm2; 3]; 3] = [
        [ex.mul(ex), ex.mul(ey), ex.mul(ez)],
        [ey.mul(ex), ey.mul(ey), ey.mul(ez)],
        [ez.mul(ex), ez.mul(ey), ez.mul(ez)],
    ];
    let sk: [[Tm2; 3]; 3] = [
        [Tm2::cst(0.0), f.mul(ez).neg(), f.mul(ey)],
        [f.mul(ez), Tm2::cst(0.0), f.mul(ex).neg()],
        [f.mul(ey).neg(), f.mul(ex), Tm2::cst(0.0)],
    ];
    let mut rmat: [[Tm2; 3]; 3] = [
        [Tm2::cst(0.0), Tm2::cst(0.0), Tm2::cst(0.0)],
        [Tm2::cst(0.0), Tm2::cst(0.0), Tm2::cst(0.0)],
        [Tm2::cst(0.0), Tm2::cst(0.0), Tm2::cst(0.0)],
    ];
    for i in 0..3 {
        for j in 0..3 {
            let mut t = sk[i][j].add(&g.mul(&outer[i][j]));
            if i == j {
                t = t.add_scalar(1.0).sub(&g.mul(&z));
            }
            rmat[i][j] = t;
        }
    }
    rmat
}

pub fn matvec(rmat: &[[Tm2; 3]; 3], v: &[Tm2; 3]) -> [Tm2; 3] {
    [
        rmat[0][0].mul(&v[0]).add(&rmat[0][1].mul(&v[1])).add(&rmat[0][2].mul(&v[2])),
        rmat[1][0].mul(&v[0]).add(&rmat[1][1].mul(&v[1])).add(&rmat[1][2].mul(&v[2])),
        rmat[2][0].mul(&v[0]).add(&rmat[2][1].mul(&v[1])).add(&rmat[2][2].mul(&v[2])),
    ]
}
