#!/bin/sh
# Full benchmark sweep. Writes results/<label>/{env.txt,bw.csv,bench.csv,parts.csv,gqa.csv}.
set -eu
cd "$(dirname "$0")/.."
label=${1:-local}
out=results/$label
mkdir -p "$out"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo "macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "cpu: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo ?)P + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo ?)E)"
    echo "memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
    system_profiler SPDisplaysDataType | grep -E 'Chipset Model|Total Number of Cores|Metal Support' | sed 's/^ */gpu /'
    echo "compiler: $(c++ --version | head -1)"
    echo "load before: $(uptime | sed 's/.*load averages*: //')"
    echo "docker containers running: $(docker ps -q 2>/dev/null | wc -l | tr -d ' ')"
    echo "top processes by CPU before the run (second sample of top -l 2):"
    top -l 2 -o cpu -n 6 -stats command,cpu 2>/dev/null | tail -7 | sed 's/^/  /'
} > "$out/env.txt"
cat "$out/env.txt"
./build/kvbench --bw | tee "$out/bw.csv"
./build/kvbench | tee "$out/bench.csv"
# Split-K partition sweep at one page size (no CPU / naive rows).
./build/kvbench --lens=512,2048,8192,32768 --pages=32 --parts=64,128,256,512,1024,2048 --no-cpu --no-naive \
    | tee "$out/parts.csv"
# GQA ablation: one threadgroup per KV head (K/V read once per group) vs per query head.
./build/kvbench --lens=512,2048,8192,32768 --pages=32 --per-head --no-cpu --no-naive | tee "$out/gqa.csv"
{
    echo "load after: $(uptime | sed 's/.*load averages*: //')"
    echo "top processes by CPU after the run:"
    top -l 2 -o cpu -n 6 -stats command,cpu 2>/dev/null | tail -7 | sed 's/^/  /'
} >> "$out/env.txt"
