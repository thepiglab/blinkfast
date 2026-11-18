//! blinkfast CLI.
//!
//!   blinkfast embed --model DIR --input FILE.{jsonl,parquet} --output FILE.{parquet,npy} [--text-col text] [--id-col id]
//!       Streams the input in batches (bounded memory; the next batch is read while the current one embeds),
//!       embeds in parallel, writes rows in input order. parquet out: id, embedding fixed_size_list<f16|f32>[dim]
//!       (zstd); npy out: [n, dim] f16/f32.
//!   blinkfast bench --model DIR --input FILE.jsonl [--engine blinkfast|model2vec-rs] [--threads N] [--n N]
//!       Times embedding of the first N texts (after a warm-up pass); prints docs/s, tokens/s, peak RSS (JSON).
//!
//! --max-length is model2vec's token cap (0 = none). The Blink configs set `max_length: null` (no cap).
use anyhow::{bail, Context, Result};
use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float16Array, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use blinkfast::{embed_batch, Model, Precision, Scratch};
use clap::{Parser, Subcommand, ValueEnum};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(version, about = "Fast CPU runner for Model2Vec static embedding models (Blink)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Engine {
    Blinkfast,
    #[value(name = "model2vec-rs")]
    Model2vecRs,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Prec {
    F32,
    F16,
}

#[derive(Subcommand)]
enum Cmd {
    Embed {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "text")]
        text_col: String,
        #[arg(long, default_value = "id")]
        id_col: String,
        #[arg(long, default_value_t = 0)]
        max_length: usize,
        /// table precision: f32 = model2vec arithmetic, f16 = half the memory traffic
        #[arg(long, value_enum, default_value_t = Prec::F32)]
        table: Prec,
        /// output vector dtype
        #[arg(long, value_enum, default_value_t = Prec::F16)]
        out_dtype: Prec,
        #[arg(long, default_value_t = 65536)]
        batch: usize,
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, value_enum, default_value_t = Engine::Blinkfast)]
        engine: Engine,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        #[arg(long, default_value_t = 20000)]
        n: usize,
        #[arg(long, default_value_t = 0)]
        max_length: usize,
        #[arg(long, value_enum, default_value_t = Prec::F32)]
        table: Prec,
        #[arg(long, default_value_t = 3)]
        reps: usize,
        /// write the vectors (f32 npy) of the timed run, for parity checks
        #[arg(long)]
        dump: Option<PathBuf>,
        /// ablation: reference tokenizer path (no fused ASCII fast path)
        #[arg(long)]
        no_fused: bool,
        /// ablation: row-at-a-time pooling
        #[arg(long)]
        no_blocked: bool,
    },
}

fn max_len(v: usize) -> Option<usize> {
    (v > 0).then_some(v)
}

fn prec(p: Prec) -> Precision {
    match p {
        Prec::F32 => Precision::F32,
        Prec::F16 => Precision::F16,
    }
}

#[repr(C)]
struct Rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    rest: [i64; 13],
}
extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}

fn peak_rss_mb() -> f64 {
    let mut ru: Rusage = unsafe { std::mem::zeroed() };
    unsafe { getrusage(0, &mut ru) };
    // ru_maxrss: KiB on Linux, bytes on macOS
    if cfg!(target_os = "macos") { ru.ru_maxrss as f64 / 1048576.0 } else { ru.ru_maxrss as f64 / 1024.0 }
}

fn read_jsonl_texts(p: &Path, col: &str, n: usize) -> Result<Vec<String>> {
    let mut v = Vec::new();
    for l in BufReader::new(File::open(p)?).lines() {
        let l = l?;
        if l.trim().is_empty() {
            continue;
        }
        let j: serde_json::Value = serde_json::from_str(&l)?;
        v.push(j[col].as_str().unwrap_or("").to_string());
        if v.len() >= n {
            break;
        }
    }
    Ok(v)
}

fn write_npy(p: &Path, v: &[f32], cols: usize, f16out: bool) -> Result<()> {
    let rows = v.len() / cols.max(1);
    let mut h = format!("{{'descr': '{}', 'fortran_order': False, 'shape': ({rows}, {cols}), }}", if f16out { "<f2" } else { "<f4" });
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut f = BufWriter::new(File::create(p)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(h.len() as u16).to_le_bytes())?;
    f.write_all(h.as_bytes())?;
    for x in v {
        if f16out {
            f.write_all(&half::f16::from_f32(*x).to_le_bytes())?;
        } else {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(f.flush()?)
}

/// Batches of (ids, texts) from jsonl or parquet, read lazily.
enum Input {
    Jsonl(std::io::Lines<BufReader<File>>),
    Parquet(parquet::arrow::arrow_reader::ParquetRecordBatchReader),
}

type Batch = (Vec<Option<String>>, Vec<String>);

impl Input {
    fn open(p: &Path, text_col: &str, id_col: &str, batch: usize) -> Result<Self> {
        if p.extension().is_some_and(|e| e == "parquet") {
            let b = ParquetRecordBatchReaderBuilder::try_new(File::open(p)?)?;
            let sd = b.parquet_schema();
            let fields = sd.root_schema().get_fields();
            let idx: Vec<usize> = (0..fields.len()).filter(|&i| [text_col, id_col].contains(&fields[i].name())).collect();
            let mask = ProjectionMask::roots(sd, idx);
            Ok(Input::Parquet(b.with_projection(mask).with_batch_size(batch).build()?))
        } else {
            Ok(Input::Jsonl(BufReader::new(File::open(p)?).lines()))
        }
    }

    fn next(&mut self, text_col: &str, id_col: &str, batch: usize) -> Result<Option<Batch>> {
        match self {
            Input::Jsonl(lines) => {
                let (mut ids, mut texts) = (Vec::new(), Vec::new());
                for l in lines.by_ref() {
                    let l = l?;
                    if l.trim().is_empty() {
                        continue;
                    }
                    let j: serde_json::Value = serde_json::from_str(&l)?;
                    ids.push(match &j[id_col] {
                        serde_json::Value::Null => None,
                        serde_json::Value::String(s) => Some(s.clone()),
                        x => Some(x.to_string()),
                    });
                    texts.push(j[text_col].as_str().unwrap_or("").to_string());
                    if texts.len() >= batch {
                        break;
                    }
                }
                Ok((!texts.is_empty()).then_some((ids, texts)))
            }
            Input::Parquet(r) => match r.next() {
                None => Ok(None),
                Some(b) => {
                    let b = b?;
                    let t = b.column_by_name(text_col).context("text column")?;
                    let t = t.as_any().downcast_ref::<StringArray>().context("text column must be utf8")?;
                    let texts = (0..t.len()).map(|i| if t.is_null(i) { String::new() } else { t.value(i).to_string() }).collect();
                    let ids = match b.column_by_name(id_col).and_then(|c| c.as_any().downcast_ref::<StringArray>().cloned()) {
                        Some(a) => (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i).to_string())).collect(),
                        None => vec![None; t.len()],
                    };
                    Ok(Some((ids, texts)))
                }
            },
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Embed { model, input, output, text_col, id_col, max_length, table, out_dtype, batch, threads } => {
            if threads > 0 {
                rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
            }
            let m = Model::load(&model, prec(table), max_len(max_length))?;
            let d = m.dim;
            let f16out = out_dtype == Prec::F16;
            let t0 = Instant::now();
            let mut inp = Input::open(&input, &text_col, &id_col, batch)?;
            let pq = output.extension().is_some_and(|e| e == "parquet");
            let item = if f16out { DataType::Float16 } else { DataType::Float32 };
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, true),
                Field::new("embedding", DataType::FixedSizeList(Arc::new(Field::new("item", item.clone(), false)), d as i32), false),
            ]));
            let tmp = output.with_extension("partial");
            let mut writer = if pq {
                let props = WriterProperties::builder()
                    .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
                    .set_dictionary_enabled(false)
                    .set_max_row_group_size(65536)
                    .build();
                Some(ArrowWriter::try_new(File::create(&tmp)?, schema.clone(), Some(props))?)
            } else {
                None
            };
            let mut all = Vec::new();
            let mut n = 0usize;
            let mut next = inp.next(&text_col, &id_col, batch)?;
            while let Some((ids, texts)) = next.take() {
                let mut out = vec![0f32; texts.len() * d];
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                let ((), nb) = rayon::join(|| embed_batch(&m, &refs, &mut out), || inp.next(&text_col, &id_col, batch));
                next = nb?;
                n += texts.len();
                match writer.as_mut() {
                    Some(w) => {
                        let vals: ArrayRef = if f16out {
                            Arc::new(Float16Array::from(out.iter().map(|x| half::f16::from_f32(*x)).collect::<Vec<_>>()))
                        } else {
                            Arc::new(Float32Array::from(out))
                        };
                        let emb = FixedSizeListArray::try_new(Arc::new(Field::new("item", item.clone(), false)), d as i32, vals, None)?;
                        w.write(&RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(ids)), Arc::new(emb)])?)?;
                    }
                    None => all.extend_from_slice(&out),
                }
            }
            match writer {
                Some(w) => {
                    w.close()?;
                }
                None => write_npy(&tmp, &all, d, f16out)?,
            }
            std::fs::rename(&tmp, &output)?;
            let s = t0.elapsed().as_secs_f64();
            println!("{}", json!({"rows": n, "dim": d, "seconds": s, "docs_per_s": n as f64 / s, "peak_rss_mb": peak_rss_mb(),
                "threads": rayon::current_num_threads(), "output": output}));
        }
        Cmd::Bench { model, input, engine, threads, n, max_length, table, reps, dump, no_fused, no_blocked } => {
            let texts = read_jsonl_texts(&input, "text", n)?;
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
            let nt = pool.current_num_threads();
            let chars: usize = texts.iter().map(|t| t.chars().count()).sum();
            let ml = max_len(max_length);
            let (secs, out, dim, tokens) = match engine {
                Engine::Blinkfast => {
                    let mut m = Model::load(&model, prec(table), ml)?;
                    m.tok.fused = !no_fused;
                    m.blocked = !no_blocked;
                    let mut s = Scratch::default();
                    let tokens: usize = refs.iter().map(|t| m.tok.ids(t, &mut s).len()).sum();
                    let mut out = vec![0f32; refs.len() * m.dim];
                    pool.install(|| embed_batch(&m, &refs, &mut out));
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        pool.install(|| embed_batch(&m, &refs, &mut out));
                    }
                    (t0.elapsed().as_secs_f64() / reps as f64, out, m.dim, tokens)
                }
                Engine::Model2vecRs => {
                    #[cfg(feature = "compare")]
                    {
                        if table != Prec::F32 {
                            bail!("model2vec-rs runs its own f32 path");
                        }
                        let m = model2vec_rs::model::StaticModel::from_pretrained(&model, None, None, None)?;
                        pool.install(|| m.encode_flat_with_args(&texts, ml));
                        let t0 = Instant::now();
                        let mut out = Vec::new();
                        for _ in 0..reps {
                            out = pool.install(|| m.encode_flat_with_args(&texts, ml));
                        }
                        (t0.elapsed().as_secs_f64() / reps as f64, out, m.dim(), 0)
                    }
                    #[cfg(not(feature = "compare"))]
                    bail!("built without the `compare` feature")
                }
            };
            if let Some(p) = dump {
                write_npy(&p, &out, dim, false)?;
            }
            let name = match engine {
                Engine::Blinkfast => "blinkfast",
                Engine::Model2vecRs => "model2vec-rs",
            };
            println!("{}", json!({"engine": name, "model": model.file_name().map(|x| x.to_string_lossy().to_string()),
                "table": if table == Prec::F32 {"f32"} else {"f16"}, "fused": !no_fused, "blocked": !no_blocked, "threads": nt, "texts": refs.len(),
                "mean_chars": chars / refs.len().max(1), "seconds": secs, "docs_per_s": refs.len() as f64 / secs,
                "tokens_per_s": if tokens > 0 { tokens as f64 / secs } else { f64::NAN }, "peak_rss_mb": peak_rss_mb()}));
        }
    }
    Ok(())
}
