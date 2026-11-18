//! blinkfast: a fast CPU runner for Model2Vec static embedding models, specialised for the BERT-style WordPiece
//! pipelines of minishlab's Blink models and bit-compatible with Python `model2vec` / `model2vec-rs` token ids.
pub mod model;
pub mod tokenizer;

pub use model::{Model, Precision};
pub use tokenizer::{Scratch, Tokenizer};

use rayon::prelude::*;

/// Embed `texts` into a row-major `[n, dim]` buffer, in parallel, with one scratch per worker.
pub fn embed_batch(m: &Model, texts: &[&str], out: &mut [f32]) {
    let d = m.dim;
    assert_eq!(out.len(), texts.len() * d);
    out.par_chunks_mut(d * 64).zip(texts.par_chunks(64)).for_each_init(Scratch::default, |s, (o, t)| {
        for (row, text) in o.chunks_mut(d).zip(t) {
            m.embed_into(text, s, row);
        }
    });
}
