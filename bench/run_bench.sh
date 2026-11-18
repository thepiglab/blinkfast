#!/usr/bin/env bash
# Three-way benchmark (Python model2vec, model2vec-rs, blinkfast) + blinkfast ablation, one machine.
#   bench/run_bench.sh MODELS_DIR TEXTS.jsonl OUT_DIR PYTHON
# MODELS_DIR holds one HF snapshot dir per model (env MODELS lists them). Writes OUT_DIR/results.jsonl.
set -euo pipefail
M=$1; X=$2; O=$3; PY=$4; N=${N:-20000}
B=$(dirname "$0")/../target/release/blinkfast
mkdir -p "$O"; R=$O/results.jsonl; : > "$R"
ALL=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
host=$( (lscpu 2>/dev/null | sed -n 's/^Model name: *//p') || sysctl -n machdep.cpu.brand_string)
echo "{\"host_cpu\":\"$host\",\"cores\":$ALL,\"n\":$N}" >> "$R"
for model in ${MODELS:-blink-8M blink-multilingual-64M}; do
  for t in 1 4 $ALL; do
    # pin to t cores so every engine (incl. joblib's n_jobs=-1 threads in model2vec) sees the same CPUs
    PIN=""; command -v taskset >/dev/null && PIN="taskset -c $(python3 -c "import os;print(','.join(map(str,sorted(os.sched_getaffinity(0))[:$t])))")"
    export RAYON_NUM_THREADS=$t OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t TOKENIZERS_PARALLELISM=true
    for bs in 64 1024; do $PIN "$PY" -I "$(dirname "$0")/py_bench.py" --model "$M/$model" --input "$X" --n $N --threads $t --batch-size $bs \
      $( [[ $t == 1 && $bs == 1024 ]] && echo --dump "$O/$model-python.npy" ) >> "$R"; done
    $PIN "$B" bench --model "$M/$model" --input "$X" --n $N --threads $t --engine model2vec-rs $( [[ $t == 1 ]] && echo --dump "$O/$model-m2vrs.npy" ) >> "$R"
    $PIN "$B" bench --model "$M/$model" --input "$X" --n $N --threads $t $( [[ $t == 1 ]] && echo --dump "$O/$model-blinkfast.npy" ) >> "$R"
    $PIN "$B" bench --model "$M/$model" --input "$X" --n $N --threads $t --table f16 $( [[ $t == 1 ]] && echo --dump "$O/$model-blinkfast-f16.npy" ) >> "$R"
  done
  # ablation, 1 thread: reference tokenizer + row pooling -> + fused tokenizer -> + blocked pooling -> + f16 table
  "$B" bench --model "$M/$model" --input "$X" --n $N --threads 1 --no-fused --no-blocked >> "$R"
  "$B" bench --model "$M/$model" --input "$X" --n $N --threads 1 --no-blocked >> "$R"
  "$B" bench --model "$M/$model" --input "$X" --n $N --threads 1 >> "$R"
  "$B" bench --model "$M/$model" --input "$X" --n $N --threads 1 --table f16 >> "$R"
  for e in m2vrs blinkfast blinkfast-f16; do $PIN "$PY" -I "$(dirname "$0")/parity.py" "$O/$model-python.npy" "$O/$model-$e.npy" >> "$R"; done
done
echo done
