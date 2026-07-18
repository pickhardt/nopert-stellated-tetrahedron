// Reads ops from stdin, one per line, args as u64 bit patterns of f64; emits result bits.
//   add|sub|mul|div  lo1 hi1 lo2 hi2  -> reslo reshi
//   sqr|sqrt         lo1 hi1          -> reslo reshi
//   trig             lo hi            -> clo chi slo shi
use dk_prover::interval::{Fi, trig_fi};
use std::io::{self, BufRead, Write};
fn b(s: &str) -> f64 { f64::from_bits(s.parse::<u64>().unwrap()) }
fn main() {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() { continue; }
        match t[0] {
            "add" | "sub" | "mul" | "div" => {
                let a = Fi::new(b(t[1]), b(t[2]));
                let c = Fi::new(b(t[3]), b(t[4]));
                let r = match t[0] { "add" => Fi::add(a, c), "sub" => Fi::sub(a, c),
                    "mul" => Fi::mul(a, c), _ => Fi::div(a, c) };
                writeln!(out, "{} {}", r.lo.to_bits(), r.hi.to_bits()).unwrap();
            }
            "sqr" | "sqrt" => {
                let a = Fi::new(b(t[1]), b(t[2]));
                let r = if t[0] == "sqr" { Fi::sqr(a) } else { Fi::sqrt(a) };
                writeln!(out, "{} {}", r.lo.to_bits(), r.hi.to_bits()).unwrap();
            }
            "trig" => {
                let (c, s) = trig_fi(b(t[1]), b(t[2]));
                writeln!(out, "{} {} {} {}", c.lo.to_bits(), c.hi.to_bits(),
                         s.lo.to_bits(), s.hi.to_bits()).unwrap();
            }
            _ => {}
        }
    }
}
