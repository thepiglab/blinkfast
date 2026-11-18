"""Python model2vec reference: time encode() on the same texts and threads, dump vectors for parity.

  python py_bench.py --model DIR --input texts.jsonl --n 20000 --threads T --batch-size B [--dump out.npy]
Threads: model2vec batches through the HF tokenizer (Rust, parallel within a batch) and numpy; both are capped to T
via RAYON_NUM_THREADS / OMP / OPENBLAS env set by the caller before import (run_bench.sh does this).
"""
import argparse, json, os, resource, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--threads", type=int, required=True)
ap.add_argument("--batch-size", type=int, default=1024)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--dump")
a = ap.parse_args()
from model2vec import StaticModel

texts = []
for line in open(a.input):
    texts.append(json.loads(line)["text"])
    if len(texts) >= a.n:
        break
if os.path.isdir(os.path.join(a.model, "0_StaticEmbedding")):
    # sentence-transformers StaticEmbedding model (e.g. static-retrieval-mrl-en-v1): its own runtime is the reference
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(a.threads)
    st = SentenceTransformer(a.model, device="cpu")
    enc = lambda xs: st.encode(xs, batch_size=a.batch_size, convert_to_numpy=True, normalize_embeddings=True)
    engine = "sentence-transformers (python)"
else:
    m = StaticModel.from_pretrained(a.model)
    kw = dict(max_length=None, batch_size=a.batch_size, use_multiprocessing=a.threads > 1, show_progress_bar=False)
    enc = lambda xs: m.encode(xs, **kw)
    engine = "model2vec (python)"
enc(texts[:2000])
t0 = time.perf_counter()
for _ in range(a.reps):
    out = enc(texts)
s = (time.perf_counter() - t0) / a.reps
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024)
if a.dump:
    np.save(a.dump, out.astype(np.float32))
print(json.dumps({"engine": engine, "model": os.path.basename(a.model.rstrip("/")), "threads": a.threads,
                  "batch_size": a.batch_size, "texts": len(texts), "seconds": s, "docs_per_s": len(texts) / s,
                  "peak_rss_mb": rss}))
