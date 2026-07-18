//! DK geometry (geom_aa) + certificate (certify_cell). Ports certificates/dk_kernel.py.
//! Pure tm2 (no LP): certify_cell is bit-identical to Python given the same triples+theta,
//! so it yields identical Hhi/Hadj/lam_lo -> identical pass decision.

use crate::interval::{rd, ru, Fi};
use crate::tm2::{matvec, rot_from_vec, Tm2};

// V4 rows (float)
const V4: [[f64; 3]; 4] = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
];

#[inline]
fn sq2() -> f64 { 2.0f64.sqrt() }

/// geom_aa: TM2 enclosures q (8x2) and y (8x2) over the cell cen +- hw.
pub fn geom_aa(cen: &[f64; 5], hw: &[f64; 5]) -> ([[Tm2; 2]; 8], [[Tm2; 2]; 8]) {
    let d = Tm2::sym(cen[0], 0, hw[0]);
    let ph = Tm2::sym(cen[1], 1, hw[1]);
    let p1 = Tm2::sym(cen[2], 2, hw[2]);
    let p2 = Tm2::sym(cen[3], 3, hw[3]);
    let g = Tm2::sym(cen[4], 4, hw[4]);
    let s2 = Tm2::from_fi(Fi::new(rd(1.0 / sq2()), ru(1.0 / sq2())));
    let sph = ph.sin();
    let cph = ph.cos();
    // eta = [ (-(cph*s2))*d, (-(cph*s2))*d, (-(sph))*d ]
    let cs = cph.mul(&s2);
    let eta0 = cs.neg().mul(&d);
    let eta = [eta0.clone(), eta0.clone(), sph.neg().mul(&d)];
    let rw = rot_from_vec(&eta);
    // ws rows
    let ws: [[Tm2; 3]; 3] = [
        [Tm2::cst(0.0), Tm2::cst(0.0), Tm2::cst(1.0)],
        [s2.neg(), s2.neg(), Tm2::cst(0.0)],
        [s2.clone(), s2.neg(), Tm2::cst(0.0)],
    ];
    // W[i][j] = ws[i][0]*Rw[j][0] + ws[i][1]*Rw[j][1] + ws[i][2]*Rw[j][2]
    let mut w: [[Tm2; 3]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| Tm2::cst(0.0)));
    for i in 0..3 {
        for j in 0..3 {
            w[i][j] = ws[i][0].mul(&rw[j][0])
                .add(&ws[i][1].mul(&rw[j][1]))
                .add(&ws[i][2].mul(&rw[j][2]));
        }
    }
    let a_f = Tm2::from_fi(Fi::new(rd(11.0 / 20.0), ru(11.0 / 20.0)));
    // X vertices
    let mut xv: Vec<[Tm2; 3]> = Vec::with_capacity(8);
    for vi in 0..8 {
        let comp: [Tm2; 3] = if vi < 4 {
            std::array::from_fn(|k| Tm2::cst(V4[vi][k]))
        } else {
            std::array::from_fn(|k| {
                let v = V4[vi - 4][k];
                if v >= 0.0 {
                    a_f.mul_scalar(v).neg()      // -(aF * V4)   (V4=+1 -> -(aF*1))
                } else {
                    a_f.mul_scalar(-v)           // aF * (-V4)   (V4=-1 -> aF*1)
                }
            })
        };
        xv.push(matvec(&w, &comp));
    }
    let q: [[Tm2; 2]; 8] = std::array::from_fn(|j| [xv[j][0].clone(), xv[j][1].clone()]);
    // xi = [ -(d*p2), d*p1, d*g ]
    let xi = [d.mul(&p2).neg(), d.mul(&p1), d.mul(&g)];
    let rx = rot_from_vec(&xi);
    let y: [[Tm2; 2]; 8] = std::array::from_fn(|j| {
        let yj = matvec(&rx, &xv[j]);
        [yj[0].clone(), yj[1].clone()]
    });
    (q, y)
}

pub struct CertOut {
    pub ok: bool,
    pub hhi: f64,
    pub hadj: f64,
    pub charge: f64,
    pub hlo: f64,
    pub lam_lo: f64,
    pub val_hi: f64,
}

/// certify_cell: (V),(L),(K) over the cell given triples+theta. Bit-identical to Python.
/// triples: each is 3 (a,b,j); theta: weight per triple.
pub fn certify_cell(
    cen: &[f64; 5], hw: &[f64; 5],
    triples: &[[(usize, usize, usize); 3]], theta: &[f64],
) -> CertOut {
    let (q, y) = geom_aa(cen, hw);
    // halfplane cache keyed by (a,b)
    let mut hp: std::collections::HashMap<(usize, usize), ([Tm2; 2], f64)> =
        std::collections::HashMap::new();
    let halfplane = |a: usize, b: usize,
                     hp: &mut std::collections::HashMap<(usize, usize), ([Tm2; 2], f64)>|
     -> ([Tm2; 2], f64) {
        if let Some(v) = hp.get(&(a, b)) { return v.clone(); }
        let ex = q[b][0].sub(&q[a][0]);
        let ey = q[b][1].sub(&q[a][1]);
        let n = [ey, ex.neg()]; // [ey, -ex]
        let mut vhi = -1e30f64;
        for m in 0..8 {
            if m == a || m == b { continue; }
            let s = n[0].mul(&q[m][0].sub(&q[a][0])).add(&n[1].mul(&q[m][1].sub(&q[a][1])));
            let sh = s.hi();
            if sh > vhi { vhi = sh; }
        }
        let out = (n, vhi);
        hp.insert((a, b), out.clone());
        out
    };

    let mut hsum = Tm2::cst(0.0);
    let mut lam_lo = 1e30f64;
    let mut val_hi = -1e30f64;
    let mut charge = 0.0f64;
    for (t, &th) in triples.iter().zip(theta.iter()) {
        if th == 0.0 { continue; }
        let mut ns: Vec<[Tm2; 2]> = Vec::with_capacity(3);
        let mut etas: Vec<f64> = Vec::with_capacity(3);
        for &(a, b, _j) in t.iter() {
            let (n, vhi) = halfplane(a, b, &mut hp);
            if vhi > val_hi { val_hi = vhi; }
            ns.push(n);
            etas.push(if vhi > 0.0 { vhi } else { 0.0 });
        }
        // lam = 2x2 dets of the other two normals
        let lam = [
            ns[1][0].mul(&ns[2][1]).sub(&ns[1][1].mul(&ns[2][0])),
            ns[2][0].mul(&ns[0][1]).sub(&ns[2][1].mul(&ns[0][0])),
            ns[0][0].mul(&ns[1][1]).sub(&ns[0][1].mul(&ns[1][0])),
        ];
        for l in lam.iter() {
            let ll = l.lo();
            if ll < lam_lo { lam_lo = ll; }
        }
        let mut gk = Tm2::cst(0.0);
        for i in 0..3 {
            let (a, _b, j) = t[i];
            let diff0 = q[a][0].sub(&y[j][0]);
            let diff1 = q[a][1].sub(&y[j][1]);
            let dotv = ns[i][0].mul(&diff0).add(&ns[i][1].mul(&diff1));
            gk = gk.add(&lam[i].mul(&dotv));
        }
        hsum = hsum.add(&gk.mul_scalar(th));
        for i in 0..3 {
            let eta = etas[i];
            if eta > 0.0 {
                let lhi = lam[i].hi();
                charge += th * (if lhi > 0.0 { lhi } else { 0.0 }) * eta;
            }
        }
    }
    let hhi = hsum.hi();
    let mut hadj = hhi + charge;
    hadj += 1e-15 * hadj.abs() + 1e-300;
    let ok = hadj < 0.0 && lam_lo > 0.0;
    CertOut { ok, hhi, hadj, charge, hlo: hsum.lo(), lam_lo, val_hi }
}
