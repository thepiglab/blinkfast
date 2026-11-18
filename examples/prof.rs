use blinkfast::{Model, Precision, Scratch};
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = Model::load(std::path::Path::new(&a[1]), Precision::F32, None).unwrap();
    let texts: Vec<String> = std::fs::read_to_string(&a[2]).unwrap().lines().take(20000)
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().unwrap().to_string()).collect();
    let mut s = Scratch::default();
    let mut out = vec![0f32; m.dim];
    for _ in 0..2 {
        let t = Instant::now(); let mut n = 0usize;
        for x in &texts { m.tok.for_each_id(x, &mut s, &mut |_| { n += 1; true }); }
        let tt = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for x in &texts { m.embed_into(x, &mut s, &mut out); }
        let te = t.elapsed().as_secs_f64();
        let t = Instant::now(); let mut z = 0usize;
        let mut norm = String::new();
        for x in &texts { norm.clear(); for c in x.chars() { norm.push(c.to_ascii_lowercase()); } z += norm.len(); }
        let tc = t.elapsed().as_secs_f64();
        println!("tokens {n} tokenize {:.3}s embed {:.3}s (pool share {:.0}%) lowercase-copy {:.3}s {z}", tt, te, 100.0*(te-tt)/te, tc);
    }
}
