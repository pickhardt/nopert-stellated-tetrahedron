// dksweep reg_lo[5] reg_hi[5] minw[5] grid_file out_file
// grid_file: "nd np"\n (nd+1 delta edges)\n (np+1 phi edges)\n then nd lines of np dlo values.
// Writes JSONL manifest (header/pass/skip/stuck) compatible with Python dk_verify.py.
use dk_prover::sweep::{sweep, RcGrid, Row, SweepResult};
use std::fs;
use std::io::{BufWriter, Write};
fn f(s: &str) -> f64 { s.parse::<f64>().unwrap() }
fn arr5(a: &[String], off: usize) -> [f64; 5] { std::array::from_fn(|i| f(&a[off + i])) }
fn jf(x: f64) -> String { format!("{}", x) } // shortest round-trip repr
fn j5(a: &[f64; 5]) -> String { format!("[{},{},{},{},{}]", jf(a[0]), jf(a[1]), jf(a[2]), jf(a[3]), jf(a[4])) }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let reg_lo = arr5(&a, 1);
    let reg_hi = arr5(&a, 6);
    let minw = arr5(&a, 11);
    let grid_file = &a[16];
    let out_file = &a[17];
    // parse grid
    let g = fs::read_to_string(grid_file).unwrap();
    let mut it = g.lines();
    let hdr: Vec<usize> = it.next().unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    let (nd, np) = (hdr[0], hdr[1]);
    let de: Vec<f64> = it.next().unwrap().split_whitespace().map(f).collect();
    let pe: Vec<f64> = it.next().unwrap().split_whitespace().map(f).collect();
    let mut dlo = vec![vec![0.0f64; np]; nd];
    for i in 0..nd {
        let row: Vec<f64> = it.next().unwrap().split_whitespace().map(f).collect();
        for j in 0..np { dlo[i][j] = row[j]; }
    }
    let grid = RcGrid { de, pe, dlo };
    let SweepResult { rows, npass, nskip, nstuck, nhandoff } = sweep(&reg_lo, &reg_hi, &minw, &grid);
    let file = fs::File::create(out_file).unwrap();
    let mut w = BufWriter::new(file);
    writeln!(w, "{{\"kind\":\"header\",\"reg_lo\":{},\"reg_hi\":{},\"RC\":\"percell\",\"minw\":{}}}",
        j5(&reg_lo), j5(&reg_hi), j5(&minw)).unwrap();
    for r in &rows {
        match r {
            Row::Skip { cen, hw, rc } => {
                writeln!(w, "{{\"kind\":\"skip\",\"cen\":{},\"hw\":{},\"RC\":{}}}", j5(cen), j5(hw), jf(*rc)).unwrap();
            }
            Row::Stuck { cen, hw } => {
                writeln!(w, "{{\"kind\":\"stuck\",\"cen\":{},\"hw\":{},\"why\":\"cert\"}}", j5(cen), j5(hw)).unwrap();
            }
            Row::Handoff { cen, hw } => {
                writeln!(w, "{{\"kind\":\"handoff\",\"cen\":{},\"hw\":{}}}", j5(cen), j5(hw)).unwrap();
            }
            Row::Pass { cen, hw, triples, theta, hadj, lam_lo, val_hi } => {
                let mut ts = String::from("[");
                for (ti, t) in triples.iter().enumerate() {
                    if ti > 0 { ts.push(','); }
                    ts.push_str(&format!("[[{},{},{}],[{},{},{}],[{},{},{}]]",
                        t[0].0, t[0].1, t[0].2, t[1].0, t[1].1, t[1].2, t[2].0, t[2].1, t[2].2));
                }
                ts.push(']');
                let th: Vec<String> = theta.iter().map(|x| jf(*x)).collect();
                writeln!(w, "{{\"kind\":\"pass\",\"cen\":{},\"hw\":{},\"triples\":{},\"theta\":[{}],\"Hadj\":{},\"lam_lo\":{},\"val_hi\":{}}}",
                    j5(cen), j5(hw), ts, th.join(","), jf(*hadj), jf(*lam_lo), jf(*val_hi)).unwrap();
            }
        }
    }
    eprintln!("{{\"npass\":{},\"nskip\":{},\"nstuck\":{},\"nhandoff\":{}}}", npass, nskip, nstuck, nhandoff);
}
