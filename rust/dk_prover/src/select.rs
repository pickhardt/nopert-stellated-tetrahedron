//! Float geometry + LP-free triple selection. The selection need not match Python (the
//! bit-identical certify_cell + Python dk_verify validate whatever triples we pick); we just
//! need to FIND triples that certify. Pool is built geometrically (per hull edge, the inner
//! vertices most violating that edge), avoiding margin_lp entirely.

use crate::geom::certify_cell;

const SQ2: f64 = std::f64::consts::SQRT_2;

// WSTAR rows: R1B=(0,0,1), R2B=(-1,-1,0)/sqrt2, US=(1,-1,0)/sqrt2
fn wstar() -> [[f64; 3]; 3] {
    [[0.0, 0.0, 1.0], [-1.0 / SQ2, -1.0 / SQ2, 0.0], [1.0 / SQ2, -1.0 / SQ2, 0.0]]
}
// VERTS: V4 then -0.55*V4
const V4: [[f64; 3]; 4] = [[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]];
fn verts() -> [[f64; 3]; 8] {
    let a = 0.55;
    [
        V4[0], V4[1], V4[2], V4[3],
        [-a * V4[0][0], -a * V4[0][1], -a * V4[0][2]],
        [-a * V4[1][0], -a * V4[1][1], -a * V4[1][2]],
        [-a * V4[2][0], -a * V4[2][1], -a * V4[2][2]],
        [-a * V4[3][0], -a * V4[3][1], -a * V4[3][2]],
    ]
}

// float rodrigues: expm(skew(v)) = I + sin(th) K + (1-cos th) K^2, th=|v|, K=skew(v/th)
fn rodrigues(v: [f64; 3]) -> [[f64; 3]; 3] {
    let th = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 { r[i][i] = 1.0; }
    if th < 1e-300 { return r; }
    let (kx, ky, kz) = (v[0] / th, v[1] / th, v[2] / th);
    let k = [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]];
    let s = th.sin();
    let omc = 1.0 - th.cos();
    let mut kk = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for m in 0..3 { acc += k[i][m] * k[m][j]; }
            kk[i][j] = acc;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += s * k[i][j] + omc * kk[i][j];
        }
    }
    r
}

fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { let mut s = 0.0; for m in 0..3 { s += a[i][m] * b[m][j]; } c[i][j] = s; } }
    c
}
fn transpose3(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { t[i][j] = a[j][i]; } }
    t
}
fn matvec3(a: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    let mut o = [0.0; 3];
    for i in 0..3 { let mut s = 0.0; for j in 0..3 { s += a[i][j] * v[j]; } o[i] = s; }
    o
}

/// float geom: returns (q[8][2], y[8][2]) at parameter point par.
pub fn geom_f(par: &[f64; 5]) -> ([[f64; 2]; 8], [[f64; 2]; 8]) {
    let (d, ph, p1, p2, g) = (par[0], par[1], par[2], par[3], par[4]);
    // w = -sin(ph)*R1B + cos(ph)*R2B
    let w = [
        -ph.sin() * 0.0 + ph.cos() * (-1.0 / SQ2),
        -ph.sin() * 0.0 + ph.cos() * (-1.0 / SQ2),
        -ph.sin() * 1.0 + ph.cos() * 0.0,
    ];
    let rw = rodrigues([d * w[0], d * w[1], d * w[2]]);
    let wmat = matmul3(&wstar(), &transpose3(&rw)); // WSTAR @ Rw^T
    let vs = verts();
    let mut xq = [[0.0f64; 3]; 8];
    for i in 0..8 { xq[i] = matvec3(&wmat, &vs[i]); }
    let mut q = [[0.0f64; 2]; 8];
    for i in 0..8 { q[i] = [xq[i][0], xq[i][1]]; }
    let rx = rodrigues([-d * p2, d * p1, d * g]);
    let mut y = [[0.0f64; 2]; 8];
    for i in 0..8 { let yy = matvec3(&rx, &xq[i]); y[i] = [yy[0], yy[1]]; }
    (q, y)
}

/// 2D convex hull (monotone chain) of the 8 points; returns CCW hull vertex indices.
fn hull_ccw(q: &[[f64; 2]; 8]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..8).collect();
    idx.sort_by(|&a, &b| q[a][0].partial_cmp(&q[b][0]).unwrap()
        .then(q[a][1].partial_cmp(&q[b][1]).unwrap()));
    let cross = |o: usize, a: usize, b: usize| {
        (q[a][0] - q[o][0]) * (q[b][1] - q[o][1]) - (q[a][1] - q[o][1]) * (q[b][0] - q[o][0])
    };
    let mut lower: Vec<usize> = Vec::new();
    for &p in &idx {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<usize> = Vec::new();
    for &p in idx.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn n_c_of_edge(q: &[[f64; 2]; 8], a: usize, b: usize) -> ([f64; 2], f64) {
    let e = [q[b][0] - q[a][0], q[b][1] - q[a][1]];
    let n = [e[1], -e[0]];
    let c = n[0] * q[a][0] + n[1] * q[a][1];
    (n, c)
}

/// float G for a triple; None if any lambda <= 0.
fn g_of_triple(q: &[[f64; 2]; 8], y: &[[f64; 2]; 8], t: &[(usize, usize, usize); 3]) -> Option<f64> {
    let mut ns = [[0.0f64; 2]; 3];
    let mut cs = [0.0f64; 3];
    for i in 0..3 { let (nn, cc) = n_c_of_edge(q, t[i].0, t[i].1); ns[i] = nn; cs[i] = cc; }
    let lam = [
        ns[1][0] * ns[2][1] - ns[1][1] * ns[2][0],
        ns[2][0] * ns[0][1] - ns[2][1] * ns[0][0],
        ns[0][0] * ns[1][1] - ns[0][1] * ns[1][0],
    ];
    if lam[0].min(lam[1]).min(lam[2]) <= 0.0 { return None; }
    let mut g = 0.0;
    for i in 0..3 {
        let j = t[i].2;
        let ny = ns[i][0] * y[j][0] + ns[i][1] * y[j][1];
        g += lam[i] * (cs[i] - ny);
    }
    Some(g)
}

/// Select triples that certify the cell. Tries parent triples first, then a geometric pool.
/// Returns (triples, theta) that certify_cell accepts, or None.
pub fn select_and_certify(
    cen: &[f64; 5], hw: &[f64; 5],
    parent: Option<&(Vec<[(usize, usize, usize); 3]>, Vec<f64>)>,
) -> Option<(Vec<[(usize, usize, usize); 3]>, Vec<f64>)> {
    // 1) parent triples (re-verified rigorously on the smaller cell)
    if let Some((ptri, pth)) = parent {
        let o = certify_cell(cen, hw, ptri, pth);
        if o.ok { return Some((ptri.clone(), pth.clone())); }
    }
    // 2) geometric pool at center
    let (q, y) = geom_f(cen);
    let hull = hull_ccw(&q);
    let ne = hull.len();
    if ne < 3 { return None; }
    // per hull edge: top-2 inner vertices by (n.y_j - c)
    let mut flags: Vec<(usize, usize, usize)> = Vec::new();
    for k in 0..ne {
        let a = hull[k];
        let b = hull[(k + 1) % ne];
        let (n, c) = n_c_of_edge(&q, a, b);
        let mut vals: Vec<(f64, usize)> = (0..8)
            .map(|j| (n[0] * y[j][0] + n[1] * y[j][1] - c, j))
            .collect();
        vals.sort_by(|x, z| z.0.partial_cmp(&x.0).unwrap());
        for m in 0..vals.len().min(2) { flags.push((a, b, vals[m].1)); }
    }
    // 3) enumerate triples with 3 distinct edges + G<0; try each single-triple certify
    let nf = flags.len();
    let theta1 = vec![1.0f64];
    let mut best: Option<(f64, [(usize, usize, usize); 3])> = None;
    for i in 0..nf {
        for j in (i + 1)..nf {
            for k in (j + 1)..nf {
                let t0 = [flags[i], flags[j], flags[k]];
                let edges = [(t0[0].0, t0[0].1), (t0[1].0, t0[1].1), (t0[2].0, t0[2].1)];
                if edges[0] == edges[1] || edges[0] == edges[2] || edges[1] == edges[2] { continue; }
                for order in [t0, [t0[0], t0[2], t0[1]]] {
                    if let Some(g) = g_of_triple(&q, &y, &order) {
                        if g < 0.0 {
                            // try single-triple certify over the whole cell
                            let tri = vec![order];
                            let o = certify_cell(cen, hw, &tri, &theta1);
                            if o.ok { return Some((tri, theta1.clone())); }
                            // track best-G candidate for a possible combo fallback
                            if best.map_or(true, |(bg, _)| g < bg) { best = Some((g, order)); }
                            break;
                        }
                    }
                }
            }
        }
    }
    None
}
