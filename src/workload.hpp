// Random decode workloads shared by the GPU tests and the benchmark.
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "metal_attn.hpp"

namespace kvb {

template <class F> void with_type(KvType t, F&& f) {
    if (t == KvType::f32) f(float{});
    else f(half_t{});
}

// q ~ U(-q_amp, q_amp), K and V ~ U(-1, 1). With D = 128 and scale 1/sqrt(D), a score has
// standard deviation q_amp / 3, so q_amp sets how peaked the softmax is.
struct Workload {
    Shape shape;
    KvType type;
    std::vector<int> lens;
    int max_len;
    std::unique_ptr<Buffer> q, k, v, seq_lens;  // k, v: contiguous [B][Hkv][max_len][D] of `type`
    std::vector<float> k32, v32;                // fp32 originals, kept only if asked (fp16 checks)

    Workload(Shape s, KvType t, std::vector<int> lens_, uint64_t seed, float q_amp, bool keep_f32 = false)
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
        k = std::make_unique<Buffer>(nkv * elem_size(t));
        v = std::make_unique<Buffer>(nkv * elem_size(t));
        if (keep_f32) k32.resize(nkv), v32.resize(nkv);
        with_type(t, [&]<class T>(T) {
            T* kp = k->as<T>();
            T* vp = v->as<T>();
            for (size_t i = 0; i < nkv; ++i) {  // padding past a sequence's length is harmless data
                const float kf = u(rng), vf = u(rng);
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

    // Fills `out` (out_floats() floats) with the fp64 reference on the stored (possibly fp16) cache.
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

struct PagedCache {
    PagedLayout layout;
    std::unique_ptr<Buffer> k, v, table;

    PagedCache(const Workload& w, int page_size, uint64_t seed) {
        layout = make_paged_layout(w.lens, page_size, seed);
        const size_t n = size_t(layout.num_blocks) * w.shape.num_kv_heads * page_size * w.shape.head_dim;
        k = std::make_unique<Buffer>(n * elem_size(w.type));
        v = std::make_unique<Buffer>(n * elem_size(w.type));
        table = std::make_unique<Buffer>(layout.block_table.size() * sizeof(int32_t));
        std::copy(layout.block_table.begin(), layout.block_table.end(), table->as<int32_t>());
        with_type(w.type, [&]<class T>(T) {
            scatter_to_paged<T>(w.shape, layout, w.lens, w.max_len, w.k->as<T>(), k->as<T>());
            scatter_to_paged<T>(w.shape, layout, w.lens, w.max_len, w.v->as<T>(), v->as<T>());
        });
    }

    PagedArgs args(const Workload& w, int part_tokens, const Buffer& out) const {
        return {w.shape, w.type, layout.page_size, layout.max_blocks, w.max_len, part_tokens,
                w.q.get(), k.get(), v.get(), table.get(), w.seq_lens.get(), &out};
    }
};

}  // namespace kvb
