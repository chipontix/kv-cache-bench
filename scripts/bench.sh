#!/bin/sh
# Full benchmark sweep. Writes results/<label>/{env.txt,bw.csv,bench.csv,parts.csv,gqa.csv,accuracy.csv}.
set -eu
cd "$(dirname "$0")/.."
label=${1:-local}
out=results/$label
mkdir -p "$out"
./scripts/env.sh > "$out/env.txt"
cat "$out/env.txt"
./build/kvbench --bw | tee "$out/bw.csv"
./build/kvbench | tee "$out/bench.csv"
# Split-K partition sweep at one page size (no CPU / naive rows).
./build/kvbench --lens=512,2048,8192,32768 --pages=32 --parts=64,128,256,512,1024,2048 --no-cpu --no-naive --types=f32,f16 \
    | tee "$out/parts.csv"
# GQA ablation: one threadgroup per KV head (K/V read once per group) vs per query head.
./build/kvbench --lens=512,2048,8192,32768 --pages=32 --per-head --no-cpu --no-naive --types=f32,f16 | tee "$out/gqa.csv"
# Error of each storage type against the fp64 reference on the same fp32 data (no timing).
./build/kvbench --accuracy --pages=16,128 | tee "$out/accuracy.csv"
{
    echo "load after: $(uptime | sed 's/.*load averages*: //')"
    echo "top processes by CPU after the run:"
    top -l 2 -o cpu -n 6 -stats command,cpu 2>/dev/null | tail -7 | sed 's/^/  /'
} >> "$out/env.txt"
