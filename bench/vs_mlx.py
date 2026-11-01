#!/usr/bin/env python3
"""MLX's fused attention, mx.fast.scaled_dot_product_attention, on the benchmark's decode shape.

Timed like `kvbench --chain`: a chain of --chain dependent decode steps (the output of one step is
the query of the next), built as a lazy graph and then evaluated with one mx.eval, whose wall
clock is divided by the chain length, warm caches, median
of --reps chains. With --dump DIR (written by `kvbench --dump=DIR`), the inputs of those lengths
are kvbench's own and the output is checked against its fp64 CPU reference. Prints CSV with the
columns of `kvbench --chain`. Versions are pinned in bench/requirements-compare.txt.
"""
import argparse
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

B, H, HKV, D = 2, 32, 8, 128


def inputs(n, dump):
    """q [B][H][1][D], K and V [B][Hkv][L][D] as fp32 numpy arrays, and the dump prefix if any."""
    pre = Path(dump) / f"L{n}_" if dump else None
    if pre and (pre.parent / (pre.name + "q.f32")).exists():
        rd = lambda name, shape: np.fromfile(f"{pre}{name}.f32", dtype=np.float32).reshape(shape)
        return rd("q", (B, H, 1, D)), rd("k", (B, HKV, n, D)), rd("v", (B, HKV, n, D)), pre
    rng = np.random.default_rng(n)  # same distribution as kvbench: q ~ U(-3, 3), K, V ~ U(-1, 1)
    u = lambda shape, a: rng.uniform(-a, a, shape).astype(np.float32)
    return u((B, H, 1, D), 3.0), u((B, HKV, n, D), 1.0), u((B, HKV, n, D), 1.0), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--chain", type=int, default=50)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    scale = 1.0 / math.sqrt(D)
    print(f"# mlx {mx.__version__} mx.fast.scaled_dot_product_attention, {mx.default_device()}; batch={B} heads={H} "
          f"kv_heads={HKV} head_dim={D} chain={a.chain} reps={a.reps} warmup={a.warmup}")
    print("impl,layout,dtype,batch,seq_len,chain,reps,median_us,min_us,max_us,gbps,max_err,err_vs")
    for n in map(int, a.lens.split(",")):
        qn, kn, vn, pre = inputs(n, a.dump)
        # f16: q, K, V and the output in fp16, MLX's own path (with an fp32 q it promotes K and V).
        for name, dt in (("f32", mx.float32), ("f16", mx.float16)):
            q, k, v = (mx.array(x).astype(dt) for x in (qn, kn, vn))
            mx.eval(q, k, v)
            err = float("nan")
            if pre:
                out = np.array(mx.fast.scaled_dot_product_attention(q, k, v, scale=scale).astype(mx.float32))
                ref = np.fromfile(f"{pre}ref_{name}.f32", dtype=np.float32).reshape(B, H, 1, D)
                err = float(np.abs(out.astype(np.float64) - ref).max())

            def chain():  # the lazy graph is built first; timed: mx.eval (encode, run, wait), as for the others
                x = q
                for _ in range(a.chain):
                    x = mx.fast.scaled_dot_product_attention(x, k, v, scale=scale)
                t0 = time.perf_counter()
                mx.eval(x)
                return (time.perf_counter() - t0) / a.chain * 1e6

            for _ in range(a.warmup):
                chain()
            us = sorted(chain() for _ in range(a.reps))
            med = us[len(us) // 2] if len(us) % 2 else (us[len(us) // 2 - 1] + us[len(us) // 2]) / 2
            size = 4 if dt == mx.float32 else 2
            min_bytes = 2 * B * HKV * n * D * size + 2 * B * H * D * size
            print(f"mlx,contiguous,{name},{B},{n},{a.chain},{a.reps},{med:.2f},{us[0]:.2f},{us[-1]:.2f},"
                  f"{min_bytes / (med * 1e-6) / 1e9:.1f},{err:.3g},stored", flush=True)


if __name__ == "__main__":
    main()
