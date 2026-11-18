# Third-party notices

blinkfast is written from scratch. It reproduces the *behaviour* (not the code) of these projects, and links some of
them only for comparison and testing:

- **Hugging Face tokenizers** (Apache-2.0): blinkfast's tokenizer re-implements the documented semantics of
  `BertNormalizer`, `BertPreTokenizer`, `FixedLength`, `WordPiece` and non-normalized added tokens, and is tested
  against `tokenizers` 0.23 for identical token ids (dev-dependency only).
- **model2vec** (MIT, MinishLab): pooling follows `StaticModel.encode` (truncation, [UNK] removal, mean, L2 norm).
  Python model2vec is the reference in the benchmark.
- **model2vec-rs** (MIT, MinishLab): linked behind the default `compare` feature as the `--engine model2vec-rs`
  benchmark arm. Build with `--no-default-features` to drop it.

Models (not included; downloaded by the user from the Hugging Face Hub):

- **minishlab/blink-8M** and **minishlab/blink-multilingual-64M** (MIT). Per their model cards, Blink-8M starts from
  a distillation of BAAI/bge-base-en-v1.5 and is trained on LightOn embedding data; Blink-multilingual-64M is
  initialised from and distilled from BAAI/bge-m3, then trained on hotchpotch/bekko-embedding-v1-unsupervised.
