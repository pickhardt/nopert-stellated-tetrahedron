// Parallel sharded DK sweep. Args:
//   dkrun reg_lo[5] reg_hi[5] minw[5] grid_file out_dir ndelta nphi
// Shards the (delta,phi) region into ndelta*nphi boxes x full p, rayon-parallel sweeps each,
// writes one JSONL manifest per box (out_dir/box_XXXXX.jsonl). Boxes tile the region.
use dk_prover::sweep::{sweep, RcGrid, Row, SweepResult};
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
fn f(s: &str) -> f64 { s.parse::<f64>().unwrap() }
fn arr5(a: &[String], off: usize) -> [f64; 5] { std::array::from_fn(|i| f(&a[off + i])) }
fn jf(x: f64) -> String { format!("{}", x) }
fn j5(a: &[f64; 5]) -> String { format!("[{},{},{},{},{}]", jf(a[0]), jf(a[1]), jf(a[2]), jf(a[3]), jf(a[4])) }

fn write_manifest(path: &str, reg_lo: &[f64;5], reg_hi: &[f64;5], minw: &[f64;5], rows: &[Row]) {
    let file = fs::File::create(path).unwrap();
    let mut w = BufWriter::new(file);
    writeln!(w, "{{\"kind\":\"header\",\"reg_lo\":{},\"reg_hi\":{},\"RC\":\"percell\",\"minw\":{}}}",
        j5(reg_lo), j5(reg_hi), j5(minw)).unwrap();
    for r in rows {
        match r {
            Row::Skip { cen, hw, rc } =>
                writeln!(w, "{{\"kind\":\"skip\",\"cen\":{},\"hw\":{},\"RC\":{}}}", j5(cen), j5(hw), jf(*rc)).unwrap(),
            Row::Stuck { cen, hw } =>
                writeln!(w, "{{\"kind\":\"stuck\",\"cen\":{},\"hw\":{},\"why\":\"cert\"}}", j5(cen), j5(hw)).unwrap(),
            Row::Handoff { cen, hw } =>
                writeln!(w, "{{\"kind\":\"handoff\",\"cen\":{},\"hw\":{}}}", j5(cen), j5(hw)).unwrap(),
            Row::Pass { cen, hw, triples, theta, hadj, lam_lo, val_hi } => {
                let mut ts = String::from("[");
                for (ti, t) in triples.iter().enumerate() {
                    if ti > 0 { ts.push(','); }
                    ts.push_str(&format!("[[{},{},{}],[{},{},{}],[{},{},{}]]",
                        t[0].0,t[0].1,t[0].2, t[1].0,t[1].1,t[1].2, t[2].0,t[2].1,t[2].2));
                }
                ts.push(']');
                let th: Vec<String> = theta.iter().map(|x| jf(*x)).collect();
                writeln!(w, "{{\"kind\":\"pass\",\"cen\":{},\"hw\":{},\"triples\":{},\"theta\":[{}],\"Hadj\":{},\"lam_lo\":{},\"val_hi\":{}}}",
                    j5(cen), j5(hw), ts, th.join(","), jf(*hadj), jf(*lam_lo), jf(*val_hi)).unwrap();
            }
        }
    }
}

fn load_grid(path: &str) -> RcGrid {
    let g = fs::read_to_string(path).unwrap();
    let mut it = g.lines();
    let hdr: Vec<usize> = it.next().unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    let (nd, np) = (hdr[0], hdr[1]);
    let de: Vec<f64> = it.next().unwrap().split_whitespace().map(f).collect();
    let pe: Vec<f64> = it.next().unwrap().split_whitespace().map(f).collect();
    let mut dlo = vec![vec![0.0f64; np]; nd];
    for i in 0..nd { let row: Vec<f64> = it.next().unwrap().split_whitespace().map(f).collect(); for j in 0..np { dlo[i][j] = row[j]; } }
    RcGrid { de, pe, dlo }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let reg_lo = arr5(&a, 1);
    let reg_hi = arr5(&a, 6);
    let minw = arr5(&a, 11);
    let grid = load_grid(&a[16]);
    let out_dir = &a[17];
    let ndelta: usize = a[18].parse().unwrap();
    let nphi: usize = a[19].parse().unwrap();
    fs::create_dir_all(out_dir).unwrap();
    // log-spaced delta boxes, uniform phi
    let (d0, d1) = (reg_lo[0], reg_hi[0]);
    let (p0, p1) = (reg_lo[1], reg_hi[1]);
    let dedges: Vec<f64> = (0..=ndelta).map(|k| d0 * (d1 / d0).powf(k as f64 / ndelta as f64)).collect();
    let mut boxes: Vec<(usize, [f64;5], [f64;5])> = Vec::new();
    let mut bid = 0;
    for i in 0..ndelta {
        for j in 0..nphi {
            let rl = [dedges[i], p0 + (p1-p0)*j as f64/nphi as f64, reg_lo[2], reg_lo[3], reg_lo[4]];
            let rh = [dedges[i+1], p0 + (p1-p0)*(j+1) as f64/nphi as f64, reg_hi[2], reg_hi[3], reg_hi[4]];
            boxes.push((bid, rl, rh)); bid += 1;
        }
    }
    // Schedule EASY-FIRST: descending delta (high delta closes fast, 0 stuck) so early
    // completions bank visible progress; the dense low-delta wall boxes come last.
    boxes.sort_by(|a, b| b.1[0].partial_cmp(&a.1[0]).unwrap());
    let ntot = boxes.len();
    let np = AtomicUsize::new(0); let ns = AtomicUsize::new(0); let nst = AtomicUsize::new(0); let nho = AtomicUsize::new(0);
    let ndone = AtomicUsize::new(0);
    let t0 = Instant::now();
    let progress_path = format!("{}/../progress.txt", out_dir);
    boxes.par_iter().for_each(|(id, rl, rh)| {
        let bt = Instant::now();
        let SweepResult { rows, npass, nskip, nstuck, nhandoff } = sweep(rl, rh, &minw, &grid);
        write_manifest(&format!("{}/box_{:05}.jsonl", out_dir, id), rl, rh, &minw, &rows);
        np.fetch_add(npass, Ordering::Relaxed);
        ns.fetch_add(nskip, Ordering::Relaxed);
        nst.fetch_add(nstuck, Ordering::Relaxed);
        nho.fetch_add(nhandoff, Ordering::Relaxed);
        let k = ndone.fetch_add(1, Ordering::Relaxed) + 1;
        let (cp, cs, cst, ch) = (np.load(Ordering::Relaxed), ns.load(Ordering::Relaxed),
                                 nst.load(Ordering::Relaxed), nho.load(Ordering::Relaxed));
        let el = t0.elapsed().as_secs_f64();
        // one progress line per box: cumulative counts + rate + ETA (flush so the 120s log-sync sees it)
        let eta = if k > 0 { el / k as f64 * (ntot - k) as f64 } else { 0.0 };
        println!("[dkrun {:>5}/{:<5}] box{:05} delta[{:.4},{:.4}] took {:.1}s | cum pass={} skip={} STUCK={} handoff={} | {:.0}s elapsed, ~{:.0}s left",
            k, ntot, id, rl[0], rh[0], bt.elapsed().as_secs_f64(), cp, cs, cst, ch, el, eta);
        let _ = std::io::stdout().flush();
        // rewrite a compact progress file every box (tiny; the 120s S3 sync uploads it)
        let _ = fs::write(&progress_path, format!(
            "boxes_done {}/{}\ncum_pass {}\ncum_skip {}\ncum_stuck {}\ncum_handoff {}\nelapsed_s {:.0}\neta_s {:.0}\nlast_box {:05} delta[{:.4},{:.4}]\n",
            k, ntot, cp, cs, cst, ch, el, eta, id, rl[0], rh[0]));
    });
    println!("{{\"boxes\":{},\"npass\":{},\"nskip\":{},\"nstuck\":{},\"nhandoff\":{}}}",
        ntot, np.load(Ordering::Relaxed), ns.load(Ordering::Relaxed), nst.load(Ordering::Relaxed), nho.load(Ordering::Relaxed));
}
