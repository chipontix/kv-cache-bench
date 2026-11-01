// llama.cpp's attention op, ggml_flash_attn_ext on its Metal backend, on the benchmark's decode
// shape and data. Timed like `kvbench --chain`: a graph of --chain dependent decode steps (the
// output of one step is the query of the next), wall clock of ggml_backend_graph_compute divided by
// the chain length, warm caches, median of --reps graphs. Checked against the fp64 CPU reference on
// the same inputs. Prints CSV (same columns as `kvbench --chain`).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml.h"
#include "workload.hpp"

using namespace kvb;

namespace {

std::vector<int> parse_ints(const char* s) {
    std::vector<int> v;
    for (const char* p = s; p; p = std::strchr(p, ',') ? std::strchr(p, ',') + 1 : nullptr) v.push_back(std::atoi(p));
    return v;
}

struct Graph {
    ggml_context* ctx;
    ggml_cgraph* gf;
    ggml_tensor* out;
};

// `steps` decode steps in a chain: step i + 1 takes the output of step i, [D][H][1][B], as its
// query, [D][1][H][B] (the same memory for one query token).
Graph build(ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, const Shape& s, int steps) {
    const size_t nodes = size_t(2 * steps + 8);
    ggml_init_params ip{nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true};
    Graph g{ggml_init(ip), nullptr, nullptr};
    g.gf = ggml_new_graph_custom(g.ctx, nodes, false);
    ggml_tensor* cur = q;
    for (int i = 0; i < steps; ++i) {
        g.out = ggml_flash_attn_ext(g.ctx, cur, k, v, nullptr, 1.0f / std::sqrt(float(s.head_dim)), 0.0f, 0.0f);
        cur = ggml_reshape_4d(g.ctx, g.out, s.head_dim, 1, s.num_heads, s.batch);
    }
    ggml_set_output(g.out);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

}  // namespace

int main(int argc, char** argv) {
    Shape shape{2, 32, 8, 128};
    std::vector<int> lens{512, 1024, 2048, 4096, 8192, 16384, 32768};
    int chain = 50, reps = 15, warmup = 3;
    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--lens=", 7)) lens = parse_ints(argv[i] + 7);
        else if (!std::strncmp(argv[i], "--chain=", 8)) chain = std::atoi(argv[i] + 8);
        else if (!std::strncmp(argv[i], "--reps=", 7)) reps = std::atoi(argv[i] + 7);
        else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    ggml_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
    }, nullptr);
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        std::fprintf(stderr, "no Metal backend\n");
        return 1;
    }
    const int threads = int(std::max(1u, std::thread::hardware_concurrency()));
    std::printf("# llama.cpp ggml_flash_attn_ext, Metal backend; batch=%d heads=%d kv_heads=%d head_dim=%d chain=%d "
                "reps=%d warmup=%d\n", shape.batch, shape.num_heads, shape.num_kv_heads, shape.head_dim, chain, reps, warmup);
    std::printf("impl,layout,dtype,batch,seq_len,chain,reps,median_us,min_us,max_us,gbps,max_err,err_vs\n");

    // q8_0: llama.cpp's int8 KV type (32 values share one fp16 scale), against our i8.
    struct Kind { const char* name; ggml_type type; KvType data; };
    for (int L : lens)
        for (const Kind kind : {Kind{"f32", GGML_TYPE_F32, KvType::f32}, Kind{"f16", GGML_TYPE_F16, KvType::f16},
                                Kind{"q8_0", GGML_TYPE_Q8_0, KvType::f32}}) {
            // Same seed, q amplitude and values as kvbench; the cache [B][Hkv][L][D] is ggml's [D][L][Hkv][B].
            const Workload w(shape, kind.data, std::vector<int>(size_t(shape.batch), L), uint64_t(L), 3.0f);
            std::vector<float> ref(w.out_floats());
            w.reference(ref.data(), threads);  // f32, f16: stored values; q8_0: the unquantized fp32 data

            ggml_init_params ip{4 * ggml_tensor_overhead(), nullptr, true};
            ggml_context* ctx = ggml_init(ip);
            ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape.head_dim, 1, shape.num_heads, shape.batch);
            ggml_tensor* k = ggml_new_tensor_4d(ctx, kind.type, shape.head_dim, L, shape.num_kv_heads, shape.batch);
            ggml_tensor* v = ggml_new_tensor_4d(ctx, kind.type, shape.head_dim, L, shape.num_kv_heads, shape.batch);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(q, w.q->data(), 0, ggml_nbytes(q));
            const int64_t rows = int64_t(shape.batch) * shape.num_kv_heads * L;
            for (auto [t, src] : {std::pair{k, w.k.get()}, std::pair{v, w.v.get()}}) {
                if (kind.type == GGML_TYPE_Q8_0) {
                    std::vector<uint8_t> packed(ggml_nbytes(t));
                    ggml_quantize_chunk(GGML_TYPE_Q8_0, src->as<float>(), packed.data(), 0, rows, shape.head_dim, nullptr);
                    ggml_backend_tensor_set(t, packed.data(), 0, packed.size());
                } else {
                    ggml_backend_tensor_set(t, src->data(), 0, ggml_nbytes(t));  // _Float16 has ggml_fp16_t's bits
                }
            }
            const double kv_bytes = double(ggml_nbytes(k) + ggml_nbytes(v));
            const double min_bytes = kv_bytes + double(w.out_floats()) * 2 * sizeof(float);

            ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            Graph one = build(q, k, v, shape, 1);
            if (!ggml_backend_supports_op(backend, one.out)) {
                std::printf("llamacpp,contiguous,%s,%d,%d,%d,%d,nan,nan,nan,nan,nan,unsupported\n", kind.name,
                            shape.batch, L, chain, reps);
                ggml_free(one.ctx), ggml_gallocr_free(alloc), ggml_backend_buffer_free(buf), ggml_free(ctx);
                continue;
            }
            ggml_gallocr_alloc_graph(alloc, one.gf);
            ggml_backend_graph_compute(backend, one.gf);
            std::vector<float> out(w.out_floats());
            ggml_backend_tensor_get(one.out, out.data(), 0, out.size() * sizeof(float));
            double err = 0;
            for (size_t i = 0; i < out.size(); ++i) err = std::max(err, std::fabs(double(out[i]) - ref[i]));
            ggml_free(one.ctx);

            Graph g = build(q, k, v, shape, chain);
            ggml_gallocr_alloc_graph(alloc, g.gf);
            for (int i = 0; i < warmup; ++i) ggml_backend_graph_compute(backend, g.gf);
            std::vector<double> us;
            for (int i = 0; i < reps; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(backend, g.gf);
                const std::chrono::duration<double, std::micro> dt = std::chrono::steady_clock::now() - t0;
                us.push_back(dt.count() / chain);
            }
            std::sort(us.begin(), us.end());
            const double med = reps % 2 ? us[reps / 2] : (us[reps / 2 - 1] + us[reps / 2]) / 2;
            std::printf("llamacpp,contiguous,%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.1f,%.3g,%s\n", kind.name, shape.batch, L,
                        chain, reps, med, us.front(), us.back(), min_bytes / (med * 1e-6) / 1e9, err,
                        kind.type == GGML_TYPE_Q8_0 ? "fp32_data" : "stored");
            std::fflush(stdout);
            ggml_free(g.ctx);
            ggml_gallocr_free(alloc);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    ggml_backend_free(backend);
    return 0;
}
