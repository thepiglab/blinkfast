//! Static model: token table + fused tokenize -> row sum -> mean -> L2 normalise, written straight into the
//! caller's output row. Semantics follow Python model2vec `StaticModel.encode` (0.10):
//!   text truncated to `max_length * median_token_length` chars; first `max_length` ids; then [UNK] ids dropped;
//!   mean of rows (x `weights[id]` and through `token_mapping` when the model has them); L2 normalised when the
//!   config says `normalize` (a text with no known token gives a zero vector).
use crate::tokenizer::{Scratch, Tokenizer};
use anyhow::{bail, Context, Result};
use half::f16;
use safetensors::{tensor::Dtype, SafeTensors};
use std::fs;
use std::path::Path;

/// f32 rows, either owned or memory-mapped straight from model.safetensors (zero-copy when 4-byte aligned).
pub enum F32Rows {
    Owned(Vec<f32>),
    Mapped { map: memmap2::Mmap, start: usize, len: usize },
}

impl F32Rows {
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        match self {
            F32Rows::Owned(v) => v,
            // SAFETY: alignment and bounds checked in Model::load; the map lives as long as self
            F32Rows::Mapped { map, start, len } => unsafe { std::slice::from_raw_parts(map.as_ptr().add(*start) as *const f32, *len) },
        }
    }
}

/// Embedding table storage. `F32` is exact; `F16` halves memory traffic (cos >= 0.9999 to F32, checked by tests).
pub enum Table {
    F32(F32Rows),
    F16(Vec<f16>),
}

pub struct Model {
    pub tok: Tokenizer,
    table: Table,
    weights: Option<Vec<f32>>,
    mapping: Option<Vec<u32>>,
    pub dim: usize,
    pub normalize: bool,
    pub max_length: Option<usize>,
    /// false = add one whole row per token (ablation); true = column-blocked sum (default). Same sums.
    pub blocked: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
}

fn f32s(t: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    let d = t.data();
    Ok(match t.dtype() {
        Dtype::F32 => d.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        Dtype::F16 => d.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        Dtype::BF16 => d.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        x => bail!("unsupported table dtype {x:?}"),
    })
}

impl Model {
    /// `dir` holds model.safetensors + tokenizer.json + config.json (model2vec layout) or the sentence-transformers
    /// StaticEmbedding layout (0_StaticEmbedding/).
    pub fn load(dir: &Path, precision: Precision, max_length: Option<usize>) -> Result<Self> {
        let base = if dir.join("model.safetensors").exists() { dir.to_path_buf() } else { dir.join("0_StaticEmbedding") };
        let tok = Tokenizer::from_json(&fs::read(base.join("tokenizer.json")).context("tokenizer.json")?)?;
        let file = fs::File::open(base.join("model.safetensors")).context("model.safetensors")?;
        // SAFETY: read-only map of a model file we do not modify while loaded
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&map)?;
        let name = ["embeddings", "embedding.weight"].into_iter().find(|n| st.tensor(n).is_ok()).context("no embeddings tensor")?;
        let t = st.tensor(name)?;
        let shape = t.shape().to_vec();
        if shape.len() != 2 {
            bail!("embeddings must be 2-D");
        }
        let dim = shape[1];
        let data = t.data();
        let start = data.as_ptr() as usize - map.as_ptr() as usize;
        let mappable = precision == Precision::F32 && t.dtype() == Dtype::F32 && cfg!(target_endian = "little")
            && (data.as_ptr() as usize) % std::mem::align_of::<f32>() == 0;
        let v = if mappable { Vec::new() } else { f32s(&t)? };
        let len = shape[0] * dim;
        let weights = st.tensor("weights").ok().map(|w| f32s(&w)).transpose()?;
        let mapping = match st.tensor("mappings") {
            Ok(m) => Some(match m.dtype() {
                Dtype::I64 => m.data().chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as u32).collect(),
                Dtype::I32 => m.data().chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect(),
                x => bail!("unsupported mapping dtype {x:?}"),
            }),
            Err(_) => None,
        };
        let cfg: serde_json::Value = fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let normalize = cfg["normalize"].as_bool().unwrap_or(true);
        drop(st);
        let table = match precision {
            Precision::F32 if mappable => Table::F32(F32Rows::Mapped { map, start, len }),
            Precision::F32 => Table::F32(F32Rows::Owned(v)),
            Precision::F16 => Table::F16(v.iter().map(|x| f16::from_f32(*x)).collect()),
        };
        Ok(Model { tok, table, weights, mapping, dim, normalize, max_length, blocked: true })
    }

    /// Truncate texts with model2vec-rs 0.4's characters-per-token unit instead of Python model2vec's, so a run
    /// reproduces model2vec-rs bit for bit (they differ only for texts longer than max_length x median chars).
    pub fn model2vec_rs_truncation(&mut self) {
        self.tok.median_token_len = self.tok.median_token_len_m2v_rs;
    }

    fn truncate<'a>(&self, text: &'a str) -> &'a str {
        match self.max_length {
            None => text,
            Some(m) => {
                let n = m.saturating_mul(self.tok.median_token_len);
                text.char_indices().nth(n).map_or(text, |(i, _)| &text[..i])
            }
        }
    }

    /// Embed one text into `out` (len == dim). Returns the number of pooled tokens.
    pub fn embed_into(&self, text: &str, s: &mut Scratch, out: &mut [f32]) -> usize {
        out.fill(0.0);
        let text = self.truncate(text);
        let unk = self.tok.unk_id;
        let cap = self.max_length.unwrap_or(usize::MAX);
        let mut ids = std::mem::take(&mut s.ids);
        ids.clear();
        let mut seen = 0usize; // tokens before [UNK] removal: the tokenizer truncates first (model2vec)
        self.tok.for_each_id(text, s, &mut |id| {
            if seen >= cap {
                return false;
            }
            seen += 1;
            if id != unk {
                ids.push(id);
            }
            true
        });
        let cnt = ids.len();
        if cnt > 0 {
            let map = self.mapping.as_deref();
            match (&self.table, &self.weights) {
                (Table::F32(t), None) if self.blocked => pool(t.as_slice(), self.dim, &ids, map, out),
                (Table::F16(t), None) if self.blocked => pool(t, self.dim, &ids, map, out),
                (t, w) => {
                    // weighted (Python `_encode_helper`): row through the mapping, weight by the original id
                    for &id in &ids {
                        let row = map.map_or(id, |m| m[id as usize]) as usize * self.dim;
                        let wt = w.as_ref().map_or(1.0, |w| w[id as usize]);
                        match t {
                            Table::F32(t) => add_row(out, &t.as_slice()[row..row + self.dim], wt),
                            Table::F16(t) => add_row_f16(out, &t[row..row + self.dim], wt),
                        }
                    }
                }
            }
            // divisions, not reciprocal multiplies: the same f32 roundings as model2vec-rs (and numpy's mean / norm)
            let denom = cnt as f32;
            if self.normalize {
                let mut sq = 0f32;
                for x in out.iter_mut() {
                    *x /= denom;
                    sq += *x * *x;
                }
                let n = sq.sqrt().max(1e-12);
                out.iter_mut().for_each(|x| *x /= n);
            } else {
                out.iter_mut().for_each(|x| *x /= denom);
            }
        }
        s.ids = ids;
        cnt
    }
}

/// Column-blocked row sum: for each block of B columns, add that block of every token's row into a register-resident
/// accumulator. Same additions in the same (token) order per element as a row-at-a-time sum.
const B: usize = 32;

trait Elem: Copy {
    fn f(self) -> f32;
}
impl Elem for f32 {
    #[inline(always)]
    fn f(self) -> f32 {
        self
    }
}
impl Elem for f16 {
    #[inline(always)]
    fn f(self) -> f32 {
        f16_to_f32(self.to_bits())
    }
}

/// IEEE half -> single, branch-light so it inlines and vectorises inside the dispatched kernels (with F16C/AVX-512
/// enabled LLVM lowers it to vcvtph2ps). Exact for every finite value, inf and NaN; subnormals via the magic-number trick.
#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let em = (h as u32) & 0x7fff;
    let bits = if em >= 0x7c00 {
        0x7f80_0000 | ((em & 0x3ff) << 13) // inf / nan
    } else if em >= 0x0400 {
        (em << 13) + ((127 - 15) << 23) // normal
    } else {
        // subnormal / zero: value = em * 2^-24
        let f = em as f32 * (1.0 / 16_777_216.0);
        return f32::from_bits(f.to_bits() | sign);
    };
    f32::from_bits(bits | sign)
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16_conversion_exact() {
        for h in 0..=u16::MAX {
            let a = half::f16::from_bits(h).to_f32();
            let b = super::f16_to_f32(h);
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{h:#x}: {a} vs {b}");
        }
    }
}

/// Runtime SIMD dispatch: the same column-blocked kernel compiled for AVX-512, AVX2+FMA+F16C, or the baseline.
/// (aarch64 always has NEON, which the baseline build already uses.)
fn pool<T: Elem>(t: &[T], d: usize, ids: &[u32], map: Option<&[u32]>, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: features checked at runtime
            return unsafe { pool_avx512(t, d, ids, map, out) };
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") {
            return unsafe { pool_avx2(t, d, ids, map, out) };
        }
    }
    pool_base(t, d, ids, map, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma,f16c")]
unsafe fn pool_avx512<T: Elem>(t: &[T], d: usize, ids: &[u32], map: Option<&[u32]>, out: &mut [f32]) {
    pool_base(t, d, ids, map, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn pool_avx2<T: Elem>(t: &[T], d: usize, ids: &[u32], map: Option<&[u32]>, out: &mut [f32]) {
    pool_base(t, d, ids, map, out)
}

#[inline(always)]
fn pool_base<T: Elem>(t: &[T], d: usize, ids: &[u32], map: Option<&[u32]>, out: &mut [f32]) {
    match map {
        None => pool_with(t, d, ids, |id| id as usize * d, out),
        Some(m) => pool_with(t, d, ids, |id| m[id as usize] as usize * d, out),
    }
}

#[inline(always)]
fn pool_with<T: Elem>(t: &[T], d: usize, ids: &[u32], row: impl Fn(u32) -> usize, out: &mut [f32]) {
    if d % B != 0 {
        for &id in ids {
            let r = row(id);
            for (o, x) in out.iter_mut().zip(&t[r..r + d]) {
                *o += x.f();
            }
        }
        return;
    }
    for c0 in (0..d).step_by(B) {
        let mut acc = [0f32; B];
        for &id in ids {
            let r: &[T; B] = t[row(id) + c0..][..B].try_into().unwrap();
            for k in 0..B {
                acc[k] += r[k].f();
            }
        }
        out[c0..c0 + B].copy_from_slice(&acc);
    }
}

#[inline(always)]
fn add_row(out: &mut [f32], row: &[f32], w: f32) {
    if w == 1.0 {
        for (o, r) in out.iter_mut().zip(row) {
            *o += *r;
        }
    } else {
        for (o, r) in out.iter_mut().zip(row) {
            *o += *r * w;
        }
    }
}

#[inline(always)]
fn add_row_f16(out: &mut [f32], row: &[f16], w: f32) {
    for (o, r) in out.iter_mut().zip(row) {
        *o += r.to_f32() * w;
    }
}
