// GPU kernels against the fp64 CPU reference over a grid of shapes, lengths, page sizes,
// storage types and seeds. Exit code 77 (CTest "skipped") when there is no Metal device.
#include <cmath>
#include <cstdio>
#include <thread>

#include "workload.hpp"

using namespace kvb;

namespace {

// Kernel vs reference on the same stored values, |v| <= 1. fp16 -> fp32 conversion is exact,
// so this bound covers fp32 arithmetic only and is the same for both storage types (README,
// "Tolerances").
constexpr double kTolKernel = 2e-5;
constexpr double kHalfUnit = 1.0 / 2048;  // fp16 unit roundoff, 2^-11

double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

// First-order bound on how much rounding K and V to fp16 can move the exact output: each score
// moves by at most E = u * scale * max_t sum_d |q_d k_td|, which changes every softmax weight by
// a factor within exp(+-2E); rounding V adds u * max|v|.
double fp16_storage_bound(const Workload& w) {
    const Shape& s = w.shape;
    const double scale = 1.0 / std::sqrt(double(s.head_dim));
    double E = 0, vmax = 0;
    for (int b = 0; b < s.batch; ++b)
        for (int h = 0; h < s.num_heads; ++h)
            for (int t = 0; t < w.lens[b]; ++t) {
                double acc = 0;
                for (int d = 0; d < s.head_dim; ++d) {
                    const size_t i = contig_index(s, w.max_len, b, h / s.group(), t, d);
                    acc += std::fabs(w.q->as<float>()[(b * s.num_heads + h) * s.head_dim + d]) * std::fabs(w.k32[i]);
                    vmax = std::max(vmax, double(std::fabs(w.v32[i])));
                }
                E = std::max(E, kHalfUnit * scale * acc);
            }
    return vmax * (kHalfUnit + std::expm1(2 * E)) + kTolKernel;
}

// The same argument for int8 storage: element x of a (block, KV head) slice with scale s moves by
// at most s / 2, so score t moves by at most scale * sum_d |q_d| * s_k / 2 (max over t is E) and
// each V element by s_v / 2. k32/v32 are the unquantized values (the workload's own for i8).
double i8_storage_bound(const Workload& w, const PagedCache& c) {
    const Shape& s = w.shape;
    const double scale = 1.0 / std::sqrt(double(s.head_dim));
    const float* ks = c.k_scale->as<float>();
    const float* vs = c.v_scale->as<float>();
    double E = 0, vq = 0, vmax = 0;
    for (int b = 0; b < s.batch; ++b)
        for (int h = 0; h < s.num_heads; ++h) {
            double q1 = 0;
            for (int d = 0; d < s.head_dim; ++d) q1 += std::fabs(w.q->as<float>()[(b * s.num_heads + h) * s.head_dim + d]);
            for (int i = 0; i * c.layout.page_size < w.lens[b]; ++i) {
                const size_t slot = size_t(c.layout.block_table[size_t(b) * c.layout.max_blocks + i]) * s.num_kv_heads +
                                    h / s.group();
                E = std::max(E, scale * q1 * ks[slot] / 2);
                vq = std::max(vq, double(vs[slot]) / 2);
                vmax = std::max(vmax, 127.0 * vs[slot]);  // = max|v| over the slice
            }
        }
    return vmax * std::expm1(2 * E) + vq + kTolKernel;
}

}  // namespace

int main() {
    if (!metal_available()) {
        std::puts("SKIP: no Metal device on this machine, GPU correctness tests not run");
        return 77;
    }
    std::printf("device: %s\n", device_name().c_str());
    const int threads = int(std::max(1u, std::thread::hardware_concurrency()));

    const Shape shapes[] = {{1, 8, 2, 128}, {1, 4, 4, 64}};  // GQA group 4, and MHA with D = 64
    const int lengths[] = {1, 7, 100, 513, 2049, 4097};
    const int pages[] = {16, 32, 64, 128};
    struct SeedCfg { uint64_t seed; int part_tokens; float q_amp; };
    const SeedCfg seeds[] = {{1, 64, 1.0f}, {2, 256, 8.0f}, {3, 512, 3.0f}};

    int checks = 0, failed = 0;
    double worst[3] = {0, 0, 0}, worst_storage_ratio = 0, worst_i8_ratio = 0, worst_i8_err = 0;
    auto check = [&](bool ok, const char* what, const Shape& s, int L, KvType t, int page, uint64_t seed,
                     double err, double tol) {
        ++checks;
        if (ok) return;
        ++failed;
        std::printf("FAIL %s H=%d Hkv=%d D=%d L=%d %s page=%d seed=%llu: err %.3g > tol %.3g\n", what,
                    s.num_heads, s.num_kv_heads, s.head_dim, L, name(t), page, (unsigned long long)seed, err, tol);
    };

    for (const Shape& shape : shapes)
        for (int L : lengths)
            for (KvType type : {KvType::f32, KvType::f16, KvType::i8})
                for (const SeedCfg& sc : seeds) {
                    const bool i8 = type == KvType::i8;
                    const int ti = int(type);
                    // Three sequences of different lengths, none a multiple of every page size. int8
                    // gets magnitudes that differ per block and head, so a misapplied scale shows.
                    Workload w(shape, type, {L, L / 3 + 1, L + 5}, sc.seed, sc.q_amp, type == KvType::f16, i8);
                    const size_t n = w.out_floats();
                    std::vector<float> ref(n);  // f32, f16: on the stored values; i8: on the fp32 data
                    w.reference(ref.data(), threads);
                    Buffer out(n * sizeof(float));
                    double err = 0;

                    if (!i8) {  // the naive kernels have no int8 path
                        run_naive(w.naive_args(out));
                        err = max_abs_diff(out.as<float>(), ref.data(), n);
                        check(err <= kTolKernel, "naive", w.shape, L, type, 0, sc.seed, err, kTolKernel);
                        worst[ti] = std::max(worst[ti], err);
                    }

                    std::vector<float> ref32;
                    double storage_tol = 0;
                    if (type == KvType::f16) {  // reference on the unrounded fp32 data
                        ref32.resize(n);
                        attention_reference<float>(w.shape, w.lens, w.max_len, w.q->as<float>(), w.k32.data(),
                                                   w.v32.data(), ref32.data(), 1.0f / std::sqrt(float(shape.head_dim)),
                                                   threads);
                        storage_tol = fp16_storage_bound(w);
                    }

                    for (int page : pages) {
                        PagedCache cache(w, page, sc.seed * 1000 + page);
                        std::vector<float> deq;  // i8: reference on the dequantized values the kernel reads
                        if (i8) deq.resize(n), cache.reference(w, deq.data(), threads);
                        const std::vector<float>& kref = i8 ? deq : ref;
                        std::fill_n(out.as<float>(), n, NAN);
                        run_paged(cache.args(w, sc.part_tokens, out));
                        err = max_abs_diff(out.as<float>(), kref.data(), n);
                        if (std::isnan(err)) err = INFINITY;
                        check(err <= kTolKernel, "paged", w.shape, L, type, page, sc.seed, err, kTolKernel);
                        worst[ti] = std::max(worst[ti], err);
                        if (page == 16 && shape.group() > 1) {  // one threadgroup per query head
                            PagedArgs args = cache.args(w, sc.part_tokens, out);
                            args.share_kv = false;
                            std::fill_n(out.as<float>(), n, NAN);
                            run_paged(args);
                            double e = max_abs_diff(out.as<float>(), kref.data(), n);
                            if (std::isnan(e)) e = INFINITY;
                            check(e <= kTolKernel, "paged-per-head", w.shape, L, type, page, sc.seed, e, kTolKernel);
                            std::fill_n(out.as<float>(), n, NAN);
                            run_paged(cache.args(w, sc.part_tokens, out));
                        }
                        if (type == KvType::f16) {
                            const double e32 = max_abs_diff(out.as<float>(), ref32.data(), n);
                            check(e32 <= storage_tol, "paged-f16-vs-f32-data", w.shape, L, type, page, sc.seed,
                                  e32, storage_tol);
                            worst_storage_ratio = std::max(worst_storage_ratio, e32 / storage_tol);
                        }
                        if (i8) {
                            const double e32 = max_abs_diff(out.as<float>(), ref.data(), n);
                            const double tol = i8_storage_bound(w, cache);
                            check(e32 <= tol, "paged-i8-vs-f32-data", w.shape, L, type, page, sc.seed, e32, tol);
                            worst_i8_ratio = std::max(worst_i8_ratio, e32 / tol);
                            worst_i8_err = std::max(worst_i8_err, e32);
                        }
                    }
                }

    std::printf("max |gpu - ref|: f32 storage %.3g, f16 storage %.3g, i8 storage %.3g (tolerance %.0e)\n",
                worst[0], worst[1], worst[2], kTolKernel);
    std::printf("f16 storage vs fp32 data: worst error / derived bound = %.3f\n", worst_storage_ratio);
    std::printf("i8 storage vs fp32 data: worst error %.3g, worst error / derived bound = %.3f\n", worst_i8_err,
                worst_i8_ratio);
    std::printf("%d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}
