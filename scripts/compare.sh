#!/bin/sh
# Our kernel vs llama.cpp (ggml_flash_attn_ext, Metal) vs MLX (mx.fast.scaled_dot_product_attention)
# on the benchmark's decode shape, all timed the same way (README, "Comparison"). On first use builds
# llama.cpp b12000 from source and keeps only the (static) vs_llamacpp binary in build-cmp/, and
# installs the pinned MLX into build-cmp/venv; `make clean` removes both. Our kernel runs from build/.
# Writes results/<label>/{compare-env.txt,compare.csv} and fails if an output is out of tolerance.
set -eu
cd "$(dirname "$0")/.."
label=${1:-local}
out=results/$label
B=build-cmp
mkdir -p "$out" "$B/dump"
need_space() {  # the llama.cpp build takes ~300 MB while it runs, the venv ~300 MB
    free_gb=$(df -g . | awk 'NR == 2 { print $4 }')
    if [ "$free_gb" -lt 15 ]; then echo "only $free_gb GB free, need 15 for $1" >&2; exit 1; fi
}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j --target kvbench
if [ ! -x "$B/vs_llamacpp" ]; then
    need_space "the llama.cpp build"
    cmake -S . -B "$B/tmp" -DCMAKE_BUILD_TYPE=Release -DKVB_LLAMACPP=ON >/dev/null
    cmake --build "$B/tmp" -j --target vs_llamacpp
    cp "$B/tmp/vs_llamacpp" "$B/"
    rm -rf "$B/tmp"  # sources, objects and libraries: the binary is self-contained
fi
if [ ! -x "$B/venv/bin/python" ]; then
    need_space "the MLX venv"
    python3 -m venv "$B/venv"
    "$B/venv/bin/pip" install -q --no-cache-dir -r bench/requirements-compare.txt
fi
{
    ./scripts/env.sh
    echo "llama.cpp: b12000 (68d20b8), ggml Metal backend"
    echo "python: $("$B/venv/bin/python" --version), $("$B/venv/bin/pip" freeze | grep -E '^(mlx|numpy)==' | tr '\n' ' ')"
} > "$out/compare-env.txt"
cat "$out/compare-env.txt"
./build/kvbench --dump="$B/dump"  # inputs and fp64 references for MLX's correctness check
{
    ./build/kvbench --chain=50 --reps=15
    "$B/vs_llamacpp" --chain=50 --reps=15 | grep -v '^impl,'
    "$B/venv/bin/python" bench/vs_mlx.py --dump "$B/dump" --chain 50 --reps 15 | grep -v '^impl,'
} | tee "$out/compare.csv"
rm -rf "$B/dump"
echo "load after: $(uptime | sed 's/.*load averages*: //')" >> "$out/compare-env.txt"
python3 scripts/tables.py compare "$label"
