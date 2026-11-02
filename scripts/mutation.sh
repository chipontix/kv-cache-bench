#!/bin/sh
# Injects known bugs into the paged kernel one at a time and checks that gpu_tests catches each.
# With the argument "boundary": bugs at the length boundaries, which boundary_tests must catch; gpu_tests
# is run as well, to show which of them only boundary_tests sees.
# Restores the sources and rebuilds afterwards.
set -u
cd "$(dirname "$0")/.."
src=src/kernels.metal
bak=$(mktemp)
cp "$src" "$bak"
trap 'cp "$bak" "$src"; rm -f "$bak"; cmake --build build -j 8 >/dev/null' EXIT
status=0
if [ "${1:-}" = boundary ]; then
    host=$(mktemp -d)
    trap 'cp "$bak" "$src"; rm -f "$bak"; cp "$host"/* src/; rm -rf "$host"; cmake --build build -j 8 >/dev/null' EXIT
    cp src/reference.cpp src/reference.hpp src/metal_attn.hpp src/metal_attn.mm "$host"
    run() {  # $1 = description, $2 = file, $3 = sed expression
        before=$(mktemp)
        cp "$2" "$before"
        sed -i.tmp "$3" "$2" && rm -f "$2.tmp"
        if cmp -s "$2" "$before"; then echo "mutation did not apply: $1"; status=1; rm -f "$before"; return; fi
        cmake --build build -j 8 >/dev/null 2>&1 || { echo "does not build: $1"; status=1; }
        out=$(./build/boundary_tests 2>&1 | tail -1)
        echo "$1: boundary_tests $out; gpu_tests $(./build/gpu_tests 2>&1 | tail -1)"
        case "$out" in *", 0 failed"*) echo "  NOT CAUGHT"; status=1 ;; esac
        cp "$before" "$2"
        rm -f "$before"
    }
    run "comparison ignores NaN" src/reference.hpp 's/if (std::isnan(e)) return INFINITY;//'
    run "reduce never writes the output" "$src" 's/out\[(b \* H + h) \* D + d\] = o \/ L;/(void)o; (void)L;/'
    run "reduce takes one partition too many at an exact multiple" "$src" 's|const int parts = (seq_lens\[b\] + p.part_tokens - 1) / p.part_tokens;|const int parts = seq_lens[b] / p.part_tokens + 1;|'
    run "reduce has no case for an empty sequence" "$src" 's/if (parts == 0) {/if (false) {/'
    run "reference has no case for an empty sequence" src/reference.cpp 's/len ? float(acc\[d\] \/ sum) : 0.0f/float(acc[d] \/ sum)/'
    run "capacity + 1 accepted" src/metal_attn.mm 's/size_t(len) > capacity)/size_t(len) > capacity + 1)/'
    run "exactly the capacity refused" src/metal_attn.mm 's/size_t(len) > capacity)/size_t(len) >= capacity)/'
    run "length past max_len accepted" src/metal_attn.mm 's/ || len > max_len//'
    run "auto_part_tokens threshold off by one" src/metal_attn.hpp 's/((max_len + part - 1) \/ part) < 512/((max_len + part - 1) \/ part) <= 512/'
    exit $status
fi
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
run "int8: K scale not applied" 's/k \*= k_scale\[blk \* p.num_kv_heads + kvh\];//'
run "int8: scale of KV head 0 for every head" 's/k_scale\[blk \* p.num_kv_heads + kvh\]/k_scale[blk * p.num_kv_heads]/'
run "int8: V dequantized with the K scale" 's/v \*= v_scale/v *= k_scale/'
run "int8: V scale of the sequence's first block for every block" 's/v_scale\[blk \* p.num_kv_heads + kvh\]/v_scale[bt[0] * p.num_kv_heads + kvh]/'
exit $status
