#!/usr/bin/env python3
"""Renders the README's markdown tables from results/<label>/*.csv (stdlib only)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def tok(n):
    n = int(n)
    return f"{n // 1024}k" if n >= 1024 else str(n)


def main(label):
    d = Path("results") / label
    bench = load(d / "bench.csv")
    by = defaultdict(dict)  # (impl, dtype, len) -> {page: row}
    for r in bench:
        by[(r["impl"], r["dtype"], int(r["seq_len"]))][int(r["page"])] = r
    lens = sorted({int(r["seq_len"]) for r in bench})
    pages = sorted({int(r["page"]) for r in bench if r["impl"] == "gpu_paged"})
    bw = load(d / "bw.csv")[0]
    print(f"Read-bandwidth probe: {float(bw['gbps']):.1f} GB/s (median of {bw['reps']}, {int(bw['bytes']) >> 20} MiB)\n")

    for dtype in ("f32", "f16"):
        print(f"**Paged kernel, {dtype} cache: median µs (effective GB/s)**\n")
        print("| tokens | split-K part | " + " | ".join(f"page {p}" for p in pages) + " |")
        print("|---:|---:|" + "---:|" * len(pages))
        for n in lens:
            rows = by[("gpu_paged", dtype, n)]
            part = rows[pages[0]]["part"]
            cells = [f"{float(rows[p]['median_us']):.1f} ({float(rows[p]['gbps']):.0f})" for p in pages]
            print(f"| {tok(n)} | {part} | " + " | ".join(cells) + " |")
        print()

    for dtype in ("f32", "f16"):
        print(f"**Against the baselines, {dtype} cache (paged = page 16)**\n")
        print("| tokens | CPU fp64, ms | naive GPU, µs | paged GPU, µs | GB/s naive / paged | paged vs naive | paged vs CPU |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for n in lens:
            cpu = by[("cpu_ref_fp64", dtype, n)][0]
            nv = by[("gpu_naive", dtype, n)][0]
            pg = by[("gpu_paged", dtype, n)][16]
            c, v, p = float(cpu["median_us"]), float(nv["median_us"]), float(pg["median_us"])
            print(f"| {tok(n)} | {c / 1e3:.2f} | {v:.1f} | {p:.1f} | {float(nv['gbps']):.0f} / {float(pg['gbps']):.0f} "
                  f"| {v / p:.1f}× | {c / p:.0f}× |")
        print()

    parts = load(d / "parts.csv")
    pvals = sorted({int(r["part"]) for r in parts})
    print("**Split-K partition sweep, page 32: effective GB/s**\n")
    print("| dtype | tokens | " + " | ".join(f"part {p}" for p in pvals) + " |")
    print("|---|---:|" + "---:|" * len(pvals))
    cell = {(r["dtype"], int(r["seq_len"]), int(r["part"])): float(r["gbps"]) for r in parts}
    for dtype in ("f32", "f16"):
        for n in sorted({int(r["seq_len"]) for r in parts}):
            best = max(cell[(dtype, n, p)] for p in pvals)
            print(f"| {dtype} | {tok(n)} | " + " | ".join(
                (f"**{cell[(dtype, n, p)]:.0f}**" if cell[(dtype, n, p)] == best else f"{cell[(dtype, n, p)]:.0f}")
                for p in pvals) + " |")
    print()

    gqa = load(d / "gqa.csv")
    g = {(r["impl"], r["dtype"], int(r["seq_len"])): r for r in gqa}
    print("**GQA ablation, page 32: one threadgroup per KV head (4 query heads) vs per query head**\n")
    print("| dtype | tokens | per KV head, µs | per query head, µs | slowdown | GB/s per KV head / per query head |")
    print("|---|---:|---:|---:|---:|---:|")
    for dtype in ("f32", "f16"):
        for n in sorted({int(r["seq_len"]) for r in gqa}):
            a, b = g[("gpu_paged", dtype, n)], g[("gpu_paged_per_head", dtype, n)]
            ta, tb = float(a["median_us"]), float(b["median_us"])
            print(f"| {dtype} | {tok(n)} | {ta:.1f} | {tb:.1f} | {tb / ta:.2f}× | {float(a['gbps']):.0f} / {float(b['gbps']):.0f} |")


def spread(label_a, label_b):
    """Run-to-run spread of the median times between two full sweeps."""
    def keyed(label, name):
        return {(r["impl"], r["dtype"], r["seq_len"], r["page"], r["part"]): float(r["median_us"])
                for r in load(Path("results") / label / name)}
    print(f"**Run-to-run spread, {label_a} vs {label_b}: |a − b| / mean of the two medians**\n")
    print("| file | impl | points | median spread | max spread | worst point |")
    print("|---|---|---:|---:|---:|---|")
    for name in ("bench.csv", "parts.csv", "gqa.csv", "bw.csv"):
        if name == "bw.csv":
            a, b = (float(load(Path("results") / l / name)[0]["median_us"]) for l in (label_a, label_b))
            print(f"| bw.csv | read_bw | 1 | {abs(a - b) / ((a + b) / 2):.1%} | | |")
            continue
        A, B = keyed(label_a, name), keyed(label_b, name)
        by_impl = {}
        for k in A:
            by_impl.setdefault(k[0], []).append((abs(A[k] - B[k]) / ((A[k] + B[k]) / 2), k))
        for impl, v in by_impl.items():
            v.sort()
            worst = v[-1][1]
            print(f"| {name} | {impl} | {len(v)} | {v[len(v) // 2][0]:.1%} | {v[-1][0]:.1%} "
                  f"| {worst[1]} {tok(worst[2])} page {worst[3]} part {worst[4]}: {A[worst]:.1f} vs {B[worst]:.1f} µs |")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        spread(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "local")
