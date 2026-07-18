// TM2 op check. Protocol (args = u64 bits of f64; a TM2 = 22 f64: c, L[5], Q[15], r):
//   add A B  -> R           (44 in, 22 out)
//   mul A B  -> R           (44 in, 22 out)
//   rot E0 E1 E2 -> 9 TM2s  (66 in, 198 out)  [rot_from_vec]
use dk_prover::tm2::{Tm2, rot_from_vec};
use std::io::{self, BufRead, Write};
fn b(s: &str) -> f64 { f64::from_bits(s.parse::<u64>().unwrap()) }
fn read_tm2(t: &[&str], off: usize) -> Tm2 {
    let c = b(t[off]);
    let mut l = [0.0; 5]; for i in 0..5 { l[i] = b(t[off+1+i]); }
    let mut q = [0.0; 15]; for i in 0..15 { q[i] = b(t[off+6+i]); }
    let r = b(t[off+21]);
    Tm2 { c, l, q, r }
}
fn emit(out: &mut impl Write, m: &Tm2) {
    write!(out, "{} ", m.c.to_bits()).unwrap();
    for i in 0..5 { write!(out, "{} ", m.l[i].to_bits()).unwrap(); }
    for i in 0..15 { write!(out, "{} ", m.q[i].to_bits()).unwrap(); }
    write!(out, "{} ", m.r.to_bits()).unwrap();
}
fn main() {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() { continue; }
        match t[0] {
            "add" | "mul" => {
                let a = read_tm2(&t, 1);
                let c = read_tm2(&t, 23);
                let r = if t[0] == "add" { a.add(&c) } else { a.mul(&c) };
                emit(&mut out, &r);
            }
            "rot" => {
                let e = [read_tm2(&t, 1), read_tm2(&t, 23), read_tm2(&t, 45)];
                let m = rot_from_vec(&e);
                for i in 0..3 { for j in 0..3 { emit(&mut out, &m[i][j]); } }
            }
            _ => {}
        }
        writeln!(out).unwrap();
    }
}
