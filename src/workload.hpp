// Random decode workloads shared by the GPU tests and the benchmark.
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "metal_attn.hpp"

namespace kvb {

// Calls f with a value of the contiguous cache's element type: half for f16, float otherwise
// (an i8 workload keeps its unquantized fp32 values; PagedCache quantizes them per block).
template <class F> void with_type(KvType t, F&& f) {
    if (t == KvType::f16) f(half_t{});
    else f(float{});
}

// q ~ U(-q_amp, q_amp), K and V ~ U(-1, 1). With D = 128 and scale 1/sqrt(D), a score has
// standard deviation q_amp / 3, so q_amp sets how peaked the softmax is. With `varied`, every
// 16-token chunk of K and of V is scaled by its own factor 2^-U(0,3), so blocks and heads differ
// in magnitude (as real caches do) and an int8 scale applied to the wrong block or head shows;
// the factors come from a separate generator, so the base values do not change.
struct Workload {
    Shape shape;
    KvType type;
    std::vector<int> lens;
    int max_len;
    std::unique_ptr<Buffer> q, k, v, seq_lens;  // k, v: contiguous [B][Hkv][max_len][D] (with_type)
    std::vector<float> k32, v32;                // fp32 originals, kept only if asked (fp16 checks)

    Workload(Shape s, KvType t, std::vector<int> lens_, uint64_t seed, float q_amp, bool keep_f32 = false,
             bool varied = false)
        : shape(s), type(t), lens(std::move(lens_)), max_len(*std::max_element(lens.begin(), lens.end())) {
        shape.batch = int(lens.size());
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        const size_t nq = size_t(shape.batch) * shape.num_heads * shape.head_dim;
        const size_t nkv = size_t(shape.batch) * shape.num_kv_heads * max_len * shape.head_dim;
        q = std::make_unique<Buffer>(nq * sizeof(float));
        for (size_t i = 0; i < nq; ++i) q->as<float>()[i] = q_amp * u(rng);
        seq_lens = std::make_unique<Buffer>(lens.size() * sizeof(int));
        std::copy(lens.begin(), lens.end(), seq_lens->as<int>());
        const size_t chunk = size_t(16) * shape.head_dim;
        std::vector<float> amp(varied ? 2 * (nkv / chunk + 1) : 0);
        std::mt19937_64 amp_rng(seed ^ 0x9e3779b97f4a7c15ull);
        for (float& a : amp) a = std::exp2(-1.5f * (u(amp_rng) + 1.0f));
        with_type(t, [&]<class T>(T) {
            k = std::make_unique<Buffer>(nkv * sizeof(T));
            v = std::make_unique<Buffer>(nkv * sizeof(T));
            if (keep_f32) k32.resize(nkv), v32.resize(nkv);
            T* kp = k->as<T>();
            T* vp = v->as<T>();
            for (size_t i = 0; i < nkv; ++i) {  // padding past a sequence's length is harmless data
                float kf = u(rng), vf = u(rng);
                if (varied) kf *= amp[2 * (i / chunk)], vf *= amp[2 * (i / chunk) + 1];
                kp[i] = T(kf);
                vp[i] = T(vf);
                if (keep_f32) k32[i] = kf, v32[i] = vf;
            }
        });
    }

    size_t out_floats() const { return size_t(shape.batch) * shape.num_heads * shape.head_dim; }

    // Bytes a decode step must move at minimum: K and V of every live token, q and out.
    double min_bytes() const {
        size_t tokens = 0;
        for (int l : lens) tokens += size_t(l);
        return double(tokens) * shape.num_kv_heads * shape.head_dim * 2 * elem_size(type) +
               double(out_floats()) * 2 * sizeof(float);
    }

    // Fills `out` (out_floats() floats) with the fp64 reference on the contiguous cache: the stored
    // fp16 values for f16, the unquantized fp32 values for f32 and i8.
    void reference(float* out, int threads = 1) const {
        with_type(type, [&]<class T>(T) {
            attention_reference<T>(shape, lens, max_len, q->as<float>(), k->as<T>(), v->as<T>(), out,
                                   1.0f / std::sqrt(float(shape.head_dim)), threads);
        });
    }

    NaiveArgs naive_args(const Buffer& out) const {
        return {shape, type, max_len, q.get(), k.get(), v.get(), seq_lens.get(), &out};
    }
};

// The workload's cache in the paged layout, in a shuffled block order; for i8, quantized per
// (block, KV head). With `contiguous`, one block per sequence in order (page = max_len): the
// layout [B][Hkv][max_len][D] of a plain contiguous cache, read by the same kernel.
struct PagedCache {
    PagedLayout layout;
    std::unique_ptr<Buffer> k, v, table, k_scale, v_scale;  // scales: i8 only

    PagedCache(const Workload& w, int page_size, uint64_t seed, bool contiguous = false) {
        if (contiguous) {
            layout = {w.max_len, 1, w.shape.batch, {}};
            for (int b = 0; b < w.shape.batch; ++b) layout.block_table.push_back(b);
        } else {
            layout = make_paged_layout(w.lens, page_size, seed);
        }
        const size_t n = size_t(layout.num_blocks) * w.shape.num_kv_heads * layout.page_size * w.shape.head_dim;
        k = std::make_unique<Buffer>(n * elem_size(w.type));
        v = std::make_unique<Buffer>(n * elem_size(w.type));
        table = std::make_unique<Buffer>(layout.block_table.size() * sizeof(int32_t));
        std::copy(layout.block_table.begin(), layout.block_table.end(), table->as<int32_t>());
        if (w.type == KvType::i8) {
            const size_t ns = size_t(layout.num_blocks) * w.shape.num_kv_heads;
            k_scale = std::make_unique<Buffer>(ns * sizeof(float));
            v_scale = std::make_unique<Buffer>(ns * sizeof(float));
            quantize_to_paged_i8(w.shape, layout, w.lens, w.max_len, w.k->as<float>(), k->as<int8_t>(),
                                 k_scale->as<float>());
            quantize_to_paged_i8(w.shape, layout, w.lens, w.max_len, w.v->as<float>(), v->as<int8_t>(),
                                 v_scale->as<float>());
            return;
        }
        with_type(w.type, [&]<class T>(T) {
            scatter_to_paged<T>(w.shape, layout, w.lens, w.max_len, w.k->as<T>(), k->as<T>());
            scatter_to_paged<T>(w.shape, layout, w.lens, w.max_len, w.v->as<T>(), v->as<T>());
        });
    }

    // Bytes of the K and V pools, scales included.
    size_t bytes() const { return k->size() + v->size() + (k_scale ? k_scale->size() + v_scale->size() : 0); }

    // fp64 reference on the values the kernel reads: the dequantized cache for i8, otherwise the
    // same as Workload::reference.
    void reference(const Workload& w, float* out, int threads = 1) const {
        if (w.type != KvType::i8) return w.reference(out, threads);
        std::vector<float> kd(w.k->size() / sizeof(float)), vd(kd.size());
        dequantize_paged_i8(w.shape, layout, w.lens, w.max_len, k->as<int8_t>(), k_scale->as<float>(), kd.data());
        dequantize_paged_i8(w.shape, layout, w.lens, w.max_len, v->as<int8_t>(), v_scale->as<float>(), vd.data());
        attention_reference<float>(w.shape, w.lens, w.max_len, w.q->as<float>(), kd.data(), vd.data(), out,
                                   1.0f / std::sqrt(float(w.shape.head_dim)), threads);
    }

    PagedArgs args(const Workload& w, int part_tokens, const Buffer& out) const {
        PagedArgs a{w.shape, w.type, layout.page_size, layout.max_blocks, w.max_len, part_tokens,
                    w.q.get(), k.get(), v.get(), table.get(), w.seq_lens.get(), &out};
        a.k_scale = k_scale.get();
        a.v_scale = v_scale.get();
        return a;
    }
};

}  // namespace kvb
