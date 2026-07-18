//! Adaptive DK sweep in Rust. Same disposition as certificates/dk_sweep.py:
//!   corner-skip (corner_dist_hi <= RC) | certify (select_and_certify) | split argmax(hw/minw).
//! Split rule matches Python EXACTLY so Python dk_verify can reconstruct the tiling.
//! Per-cell RC comes from a precomputed grid (conservative: grid d_lo <= exact rc_dlo d_lo,
//! so any skip we take is still accepted by dk_verify's exact per-cell re-certification).

use crate::geom::certify_cell;
use crate::select::select_and_certify;

const EPS: f64 = 1e-12;
const M0: f64 = 1.0;
// The box-anchored layer (sbbox, wall_sweep2) certifies delta in [1e-5, DELTA_BOX] x phi[0,pi]
// (delta*K_box subset conv F). A DK cell whose whole delta-range is below DELTA_BOX is that
// layer's remit -> handoff (SOUND: box-anchored kills every inner rotation there). Cells with
// delta >= DELTA_BOX are DK's and close by the det-dual (fine enough minw) -- NOT handed off.
const DELTA_BOX: f64 = 5.12e-3;

pub fn corner_dist_hi(cen: &[f64; 5], hw: &[f64; 5]) -> f64 {
    let (p1c, p2c, gc) = (cen[2], cen[3], cen[4]);
    let (h1, h2, hg) = (hw[2], hw[3], hw[4]);
    let d2 = (p1c.abs() + h1).powi(2) + (p2c.abs() + h2).powi(2) + (gc.abs() + hg).powi(2);
    let dd = d2.sqrt() * (1.0 + EPS);
    let cphi = cen[1].cos();
    let sphi = cen[1].sin();
    let s2v = (p1c - 2.0 * cphi).abs() + h1;
    let s2b = (p2c - 2.0 * sphi).abs() + h2;
    let s2sq = s2v.powi(2) + s2b.powi(2) + (gc.abs() + hg).powi(2);
    let ds = s2sq.sqrt() * (1.0 + EPS) + 2.0 * hw[1];
    dd.min(ds)
}

/// RC grid: axis-aligned (delta,phi) cells each carrying a certified depth lower bound d_lo.
/// RC(cell) = min d_lo over grid cells intersecting the cell's (delta,phi) range / (M0*delta_hi).
pub struct RcGrid {
    pub de: Vec<f64>,  // delta edges (len nd+1)
    pub pe: Vec<f64>,  // phi edges (len np+1)
    pub dlo: Vec<Vec<f64>>, // dlo[i][j] over [de[i],de[i+1]] x [pe[j],pe[j+1]]
}
impl RcGrid {
    /// min certified d_lo over grid cells intersecting the cell's (delta,phi) range.
    pub fn dlo(&self, cen: &[f64; 5], hw: &[f64; 5]) -> f64 {
        let d0 = cen[0] - hw[0]; let d1 = cen[0] + hw[0];
        let p0 = cen[1] - hw[1]; let p1 = cen[1] + hw[1];
        let mut mind = f64::INFINITY;
        for i in 0..self.de.len() - 1 {
            if self.de[i + 1] < d0 || self.de[i] > d1 { continue; }
            for j in 0..self.pe.len() - 1 {
                if self.pe[j + 1] < p0 || self.pe[j] > p1 { continue; }
                if self.dlo[i][j] < mind { mind = self.dlo[i][j]; }
            }
        }
        if mind.is_finite() { mind } else { 0.0 }
    }
    pub fn rc(&self, cen: &[f64; 5], hw: &[f64; 5]) -> f64 {
        self.dlo(cen, hw) / (M0 * (cen[0] + hw[0]))
    }
}

pub enum Row {
    Pass { cen: [f64; 5], hw: [f64; 5], triples: Vec<[(usize, usize, usize); 3]>, theta: Vec<f64>,
           hadj: f64, lam_lo: f64, val_hi: f64 },
    Skip { cen: [f64; 5], hw: [f64; 5], rc: f64 },
    Stuck { cen: [f64; 5], hw: [f64; 5] },
    // (delta,phi) where rc_dlo cannot certify depth -> NOT DK's remit; deferred to the
    // box-anchored / Lemma-6' window layer (verified gap-free by the IF-1 assembly check).
    Handoff { cen: [f64; 5], hw: [f64; 5] },
}

pub struct SweepResult {
    pub rows: Vec<Row>,
    pub npass: usize,
    pub nskip: usize,
    pub nstuck: usize,
    pub nhandoff: usize,
}

/// Sweep one box (reg_lo..reg_hi) to completion. Returns rows for the manifest.
pub fn sweep(reg_lo: &[f64; 5], reg_hi: &[f64; 5], minw: &[f64; 5], grid: &RcGrid) -> SweepResult {
    let cen0: [f64; 5] = std::array::from_fn(|i| (reg_lo[i] + reg_hi[i]) / 2.0);
    let hw0: [f64; 5] = std::array::from_fn(|i| (reg_hi[i] - reg_lo[i]) / 2.0);
    // stack item: (cen, hw, parent triples/theta)
    type Parent = Option<(Vec<[(usize, usize, usize); 3]>, Vec<f64>)>;
    let mut stack: Vec<([f64; 5], [f64; 5], Parent)> = vec![(cen0, hw0, None)];
    let mut rows: Vec<Row> = Vec::new();
    let (mut npass, mut nskip, mut nstuck, mut nhandoff) = (0usize, 0usize, 0usize, 0usize);
    while let Some((cen, hw, parent)) = stack.pop() {
        // whole delta-range below the box-anchored ceiling -> that layer's remit -> handoff (sound)
        if cen[0] + hw[0] < DELTA_BOX {
            nhandoff += 1;
            rows.push(Row::Handoff { cen, hw });
            continue;
        }
        let rc = grid.rc(&cen, &hw);
        if corner_dist_hi(&cen, &hw) <= rc {
            nskip += 1;
            rows.push(Row::Skip { cen, hw, rc });
            continue;
        }
        let sel = select_and_certify(&cen, &hw, parent.as_ref());
        if let Some((tri, th)) = sel {
            let o = certify_cell(&cen, &hw, &tri, &th);
            npass += 1;
            rows.push(Row::Pass { cen, hw, triples: tri, theta: th, hadj: o.hadj,
                                  lam_lo: o.lam_lo, val_hi: o.val_hi });
            continue;
        }
        let _ = rc;
        // split along argmax(hw/minw); if none splittable -> stuck (matches Python)
        let mut ax = 0usize; let mut best = f64::NEG_INFINITY;
        for i in 0..5 { let r = hw[i] / minw[i]; if r > best { best = r; ax = i; } }
        if best <= 1.0 {
            nstuck += 1;
            rows.push(Row::Stuck { cen, hw });
            continue;
        }
        let mut h2 = hw; h2[ax] = hw[ax] / 2.0;
        let mut ca = cen; ca[ax] = cen[ax] - h2[ax];
        let mut cb = cen; cb[ax] = cen[ax] + h2[ax];
        // pass this cell's triples down as parent candidates (re-verified on children)
        let child_parent: Parent = None; // (parent-cache handled inside select on retry)
        stack.push((ca, h2, child_parent.clone()));
        stack.push((cb, h2, child_parent));
    }
    SweepResult { rows, npass, nskip, nstuck, nhandoff }
}
