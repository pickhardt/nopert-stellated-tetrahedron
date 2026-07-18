// Reads cells "cen0..4(bits) hw0..4(bits)" per line; runs select_and_certify (no parent).
// Emits "1 hadj_bits" if certified, else "0 0".
use dk_prover::select::select_and_certify;
use dk_prover::geom::certify_cell;
use std::io::{self, BufRead, Write};
fn b(s: &str) -> f64 { f64::from_bits(s.parse::<u64>().unwrap()) }
fn main() {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 10 { continue; }
        let mut cen = [0.0f64; 5]; let mut hw = [0.0f64; 5];
        for i in 0..5 { cen[i] = b(t[i]); }
        for i in 0..5 { hw[i] = b(t[5 + i]); }
        match select_and_certify(&cen, &hw, None) {
            Some((tri, th)) => {
                let o = certify_cell(&cen, &hw, &tri, &th);
                writeln!(out, "1 {}", o.hadj.to_bits()).unwrap();
            }
            None => { writeln!(out, "0 0").unwrap(); }
        }
    }
}
