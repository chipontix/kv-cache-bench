#!/bin/sh
# Our kernel vs llama.cpp (ggml_flash_attn_ext, Metal) vs MLX (mx.fast.scaled_dot_product_attention)
# on the benchmark's decode shape, all timed the same way (README, "Comparison"). Builds llama.cpp
# b12000 from source into build-cmp/ and installs the pinned MLX into build-cmp/venv on first use.
# Writes results/<label>/{compare-env.txt,compare.csv} and fails if an output is out of tolerance.
set -eu
cd "$(dirname "$0")/.."
label=${1:-local}
out=results/$label
B=build-cmp
mkdir -p "$out" "$B/dump"
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DKVB_LLAMACPP=ON >/dev/null
cmake --build "$B" -j
if [ ! -x "$B/venv/bin/python" ]; then
    python3 -m venv "$B/venv"
    "$B/venv/bin/pip" install -q -r bench/requirements-compare.txt
fi
{
    ./scripts/env.sh
    echo "llama.cpp: b12000 (68d20b8), ggml Metal backend"
    echo "python: $("$B/venv/bin/python" --version), $("$B/venv/bin/pip" freeze | grep -E '^(mlx|numpy)==' | tr '\n' ' ')"
} > "$out/compare-env.txt"
cat "$out/compare-env.txt"
"$B/kvbench" --dump="$B/dump"  # inputs and fp64 references for MLX's correctness check
{
    "$B/kvbench" --chain=50 --reps=15
    "$B/vs_llamacpp" --chain=50 --reps=15 | grep -v '^impl,'
    "$B/venv/bin/python" bench/vs_mlx.py --dump "$B/dump" --chain 50 --reps 15 | grep -v '^impl,'
} | tee "$out/compare.csv"
rm -rf "$B/dump"
echo "load after: $(uptime | sed 's/.*load averages*: //')" >> "$out/compare-env.txt"
python3 scripts/tables.py compare "$label"
