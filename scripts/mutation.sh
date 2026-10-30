#!/bin/sh
# Injects known bugs into the paged kernel one at a time and checks that gpu_tests catches each.
# Restores src/kernels.metal and rebuilds afterwards.
set -u
cd "$(dirname "$0")/.."
src=src/kernels.metal
bak=$(mktemp)
cp "$src" "$bak"
trap 'cp "$bak" "$src"; rm -f "$bak"; cmake --build build -j 8 >/dev/null' EXIT
status=0
run() {  # $1 = description, $2 = sed expression
    cp "$bak" "$src"
    sed -i.tmp "$2" "$src" && rm -f "$src.tmp"
    if cmp -s "$src" "$bak"; then echo "mutation did not apply: $1"; status=1; return; fi
    cmake --build build -j 8 >/dev/null 2>&1
    log=$(./build/gpu_tests 2>&1)
    out=$(echo "$log" | tail -1)
    echo "$1: $out; $(echo "$log" | grep '^max |gpu')"
    case "$out" in *", 0 failed"*) echo "  NOT CAUGHT"; status=1 ;; esac
}
run "no online-softmax rescale (corr = 1)" 's/const float corr = exp(m\[g\] - m_new);/const float corr = 1.0f;/'
run "accumulator rounded to fp16 each step" 's/acc\[g\] = acc\[g\] \* corr + pr \* v;/acc[g] = f_vec(vec<half, DPL>(acc[g] * corr + pr * v));/'
run "off-by-one row inside a page" 's/t % P) \* D/(t + 1) % P) * D/'
run "simdgroup merge ignores max shift in L" 's/L += tg_ml\[s\]\[g\].y \* w;/L += tg_ml[s][g].y;/'
run "reduce drops the last partition" 's/for (int i = 0; i < parts; ++i) {/for (int i = 0; i < parts - 1; ++i) {/'
run "block table ignored (logical = physical)" 's/size_t(bt\[t \/ P\])/size_t(t \/ P)/'
exit $status
