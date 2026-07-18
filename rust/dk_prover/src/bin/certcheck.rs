// certify_cell check. Protocol per line:
//   cert  cen0..4(bits) hw0..4(bits)  nt  [a b j a b j a b j thbits]*nt
// Output: hhi hadj charge hlo lam_lo val_hi ok   (first 6 as u64 bits, ok as 0/1)
use dk_prover::geom::certify_cell;
use std::io::{self, BufRead, Write};
fn b(s: &str) -> f64 { f64::from_bits(s.parse::<u64>().unwrap()) }
fn u(s: &str) -> usize { s.parse::<usize>().unwrap() }
fn main() {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() || t[0] != "cert" { continue; }
        let mut cen = [0.0f64; 5];
        let mut hw = [0.0f64; 5];
        for i in 0..5 { cen[i] = b(t[1 + i]); }
        for i in 0..5 { hw[i] = b(t[6 + i]); }
        let nt = u(t[11]);
        let mut triples: Vec<[(usize, usize, usize); 3]> = Vec::with_capacity(nt);
        let mut theta: Vec<f64> = Vec::with_capacity(nt);
        let mut p = 12;
        for _ in 0..nt {
            let tri = [
                (u(t[p]), u(t[p + 1]), u(t[p + 2])),
                (u(t[p + 3]), u(t[p + 4]), u(t[p + 5])),
                (u(t[p + 6]), u(t[p + 7]), u(t[p + 8])),
            ];
            theta.push(b(t[p + 9]));
            triples.push(tri);
            p += 10;
        }
        let o = certify_cell(&cen, &hw, &triples, &theta);
        writeln!(out, "{} {} {} {} {} {} {}",
            o.hhi.to_bits(), o.hadj.to_bits(), o.charge.to_bits(), o.hlo.to_bits(),
            o.lam_lo.to_bits(), o.val_hi.to_bits(), if o.ok { 1 } else { 0 }).unwrap();
    }
}
