# blinkfast

Fast CPU inference for [Model2Vec](https://github.com/MinishLab/model2vec) static embedding models, written in Rust
and specialised for Minish's **Blink** models:

- **[minishlab/blink-8M](https://huggingface.co/minishlab/blink-8M)**: English, 30,000 x 256 token table.
- **[minishlab/blink-multilingual-64M](https://huggingface.co/minishlab/blink-multilingual-64M)**: 100+ languages,
  250,000 x 256 token table.

A static model has no transformer: embedding a text is tokenize → look up one row per token → mean → L2 normalise.
That makes the tokenizer the bottleneck, so blinkfast re-implements exactly the tokenizer pipelines these models
ship, and fuses everything after it.

**About the name.** "Blink fast", as in "think fast".

## Results at a glance

AMD EPYC 9655P (Zen 5), 32 cores, 20,000 real PubMed Central text chunks (mean 1,489 characters, ~340 tokens).

- **Same vectors.** Max abs difference 1.2e-7 and cosine 1.000000 to Python `model2vec` on every text, and identical
  token ids to Hugging Face `tokenizers` on 20,000 PMC chunks plus Unicode edge cases.
- **Faster than model2vec-rs** on both models: **1.4x** on Blink-8M and **7.8x** on Blink-multilingual-64M at 32
  cores.
- **Faster than Python model2vec**: **62x** and **49x** at 32 cores.
- **Bit-identical at scale.** Re-embedding all of PubMed Central Open Access (251,240,487 text chunks plus
  9,027,525 titles, both models) in model2vec-rs-compatible mode reproduced model2vec-rs 0.4's float16 vectors
  byte for byte on every one of the 245 shards. Encode time summed over the 4-thread shard tasks: Blink-8M 2,769 s vs 2,662 s (on par),
  Blink-multilingual-64M 3,962 s vs 34,580 s (8.7x faster).
- **Smaller.** 79-228 MB peak RSS, against 106-602 MB for model2vec-rs and up to 1.6 GB for Python.

### Throughput: docs/s (threads: 1 / 4 / 32)

Blink-8M:

| Engine | 1 | 4 | 32 | Peak RSS |
|---|---:|---:|---:|---:|
| **blinkfast** | **52,342** | **215,935** | **1,050,581** | **79-89 MB** |
| model2vec-rs 0.4.0 | 46,794 | 189,234 | 753,688 | 106-126 MB |
| model2vec 0.10 (Python) | 2,834 | 10,499 | 16,811 | 247-1,301 MB |

Blink-multilingual-64M:

| Engine | 1 | 4 | 32 | Peak RSS |
|---|---:|---:|---:|---:|
| **blinkfast** | **35,424** | **150,805** | **717,774** | **218-228 MB** |
| model2vec-rs 0.4.0 | 3,686 | 14,746 | 92,423 | 602 MB |
| model2vec 0.10 (Python) | 2,718 | 10,026 | 14,553 | 648-1,598 MB |

```
docs/s at 32 cores (each # = 25,000 docs/s)

Blink-8M
  blinkfast       ##########################################  1,050,581
  model2vec-rs    ##############################                753,688
  model2vec (py)  #                                              16,811

Blink-multilingual-64M
  blinkfast       #############################                 717,774
  model2vec-rs    ####                                           92,423
  model2vec (py)  #                                              14,553
```

Python is the best of batch sizes 64 and 1,024 per thread count. Tokens/s for blinkfast at 32 cores: 3.6e8 (Blink-8M)
and 2.5e8 (Blink-multilingual-64M).

### Where the speed comes from (ablation, 1 thread)

Each row adds one change to the row above. All rows give the same token ids and vectors.

| Step | Blink-8M | | Blink-multilingual-64M | |
|---|---:|---:|---:|---:|
| reference pipeline: normalise into a string, then pre-tokenise, row-at-a-time sum | 15,130 | 1.00x | 11,674 | 1.00x |
| + fused single-pass normaliser / pre-tokeniser with an ASCII class table | 41,320 | 2.73x | 24,818 | 2.13x |
| + column-blocked pooling (register accumulators, AVX-512 / AVX2 dispatch) | 47,389 | 3.13x | 32,217 | 2.76x |
| f16 table instead of f32 (not adopted) | 36,510 | 2.41x | 30,300 | 2.60x |

The ablation and headline runs are separate runs on a shared node, so the same configuration differs by up to ~10%
between them (47,389 vs 52,342 on Blink-8M). The f16 table halves table memory but is slower here, because both
tables fit in cache. It is kept as `--table f16` for larger models. Its vectors differ by at most 9.3e-5.

**Why model2vec-rs is slow on Blink-multilingual-64M.** That model's tokenizer adds a `FixedLength(100)`
pre-tokenizer and 30,000 CJK words as added tokens. model2vec-rs's fast WordPiece path does not cover either, so it
falls back to the generic `tokenizers` pipeline. blinkfast covers both.

## Usage

```bash
cargo build --release
```

```bash
# a parquet or jsonl file in, a parquet (id, embedding f16[dim]) or .npy file out; streamed in 65,536-row batches
./target/release/blinkfast embed --model path/to/blink-8M --input chunks.parquet --text-col text --id-col unit_id \
  --output vectors.parquet
```

```bash
# benchmark one engine on a jsonl {"text": ...}; --dump writes the vectors for parity checks
./target/release/blinkfast bench --model path/to/blink-8M --input texts.jsonl --threads 32 --engine blinkfast
./target/release/blinkfast bench --model path/to/blink-8M --input texts.jsonl --threads 32 --engine model2vec-rs
```

Library:

```rust
use blinkfast::{embed_batch, Model, Precision};
let m = Model::load("path/to/blink-8M".as_ref(), Precision::F32, None)?;
let texts = ["IL-6 blockade in cytokine storm", "porcine model of myocardial infarction"];
let mut out = vec![0f32; texts.len() * m.dim];
embed_batch(&m, &texts, &mut out); // row-major [n, dim], L2-normalised
```

Models load from a local Hugging Face snapshot directory (model2vec layout, or the sentence-transformers
`0_StaticEmbedding/` layout). blinkfast never downloads anything.

## Method

- **Inputs.** 20,000 random commercial-use text chunks from the PigMed PubMed Central corpus (body chunks with their
  section heading, table text, captions), the same file for every engine.
- **Threads.** Each engine is pinned to the same first N cores with `taskset`. Python model2vec uses its threaded
  `encode` (`use_multiprocessing`) above 1 thread.
- **Timing.** One warm-up pass, then the mean of 3 timed passes over all 20,000 texts. Model load is excluded.
- **Parity.** `bench/parity.py` compares each engine's vectors to Python model2vec. `tests/parity.rs` compares
  token ids to `tokenizers` 0.23 on the built-in edge cases and on the file in `BLINKFAST_TEXTS`.
- **Raw results.** [bench/results-epyc-9655p.jsonl](bench/results-epyc-9655p.jsonl). Reproduce with
  `bench/run_bench.sh MODELS_DIR TEXTS.jsonl OUT_DIR PYTHON`.

## Design

- **Tokenizer** (`src/tokenizer.rs`). It accepts only the pipeline shapes it reproduces exactly:
  - BertNormalizer, BertPreTokenizer, optional FixedLength, WordPiece with `##` continuations;
  - non-normalized added tokens, matched leftmost-longest with Aho-Corasick, as `tokenizers` does.

  Anything else (BPE or Unigram models such as Gemma) loads through `tokenizers` (feature `generic`, on by default)
  and still gets blinkfast's pooling and I/O.
- **One pass per text.** Every BertNormalizer step is a per-character map, so each character is normalised and
  classified for pre-tokenisation as it is read:
  - ASCII bytes go through a 128-entry class table built from the exact Unicode predicates at load time;
  - other characters go through the full NFD / accent-strip / lowercase / CJK-padding path.

  Words collect in a reused buffer. WordPiece lookups use one hash map for word-initial pieces and one for `##`
  pieces stored without the prefix, so no strings are built.
- **Pooling** (`src/model.rs`). Token ids go into a reused buffer, then the sum runs 32 columns at a time with the
  accumulator in registers. The kernel is compiled for AVX-512, AVX2+FMA+F16C and baseline, picked at run time.
  - Additions run in token order, so results match a row-at-a-time sum.
  - f32 tables are memory-mapped straight from `model.safetensors` (zero copy when aligned).
- **Semantics** follow Python `model2vec` `StaticModel.encode`:
  - truncate to `max_length x median token length` characters, then to `max_length` tokens;
  - drop `[UNK]`;
  - mean, then L2 normalise; weights and vocabulary mappings are supported.

  `--max-length 0` (the default) means no limit, as in the Blink configs.

## Licence

MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the projects whose behaviour blinkfast reproduces or
benchmarks against, and for the models' own licences (MIT).
