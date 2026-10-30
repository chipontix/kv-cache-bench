// Decode-step benchmark: paged fused kernel vs naive 3-pass GPU kernel vs the CPU reference.
// Prints CSV to stdout. GPU time is the command buffer's GPUEndTime - GPUStartTime; caches are
// flushed before every timed run; the median of --reps runs is reported.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>

#include "workload.hpp"

using namespace kvb;

namespace {

std::vector<int> parse_ints(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        p = std::strchr(p, ',');
        if (!p) break;
        ++p;
    }
    return v;
}

struct Stats { double median, min, max; };

Stats stats(std::vector<double> t) {
    std::sort(t.begin(), t.end());
    const size_t n = t.size();
    return {n % 2 ? t[n / 2] : (t[n / 2 - 1] + t[n / 2]) / 2, t.front(), t.back()};
}

double max_abs_diff(const Buffer& a, const Buffer& b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a.as<float>()[i] - b.as<float>()[i])));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    Shape shape{2, 32, 8, 128};
    std::vector<int> lens{512, 1024, 2048, 4096, 8192, 16384, 32768}, pages{16, 32, 64, 128}, parts{0};  // part 0 = auto
    std::vector<KvType> types{KvType::f32, KvType::f16};
    int reps = 30, warmup = 3, cpu_reps = 3;
    bool naive = true, cpu = true, flush = true, per_head = false;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto val = [&](const char* key) -> const char* {
            const size_t n = std::strlen(key);
            return std::strncmp(a, key, n) == 0 && a[n] == '=' ? a + n + 1 : nullptr;
        };
        if (auto v = val("--batch")) shape.batch = std::atoi(v);
        else if (auto v = val("--heads")) shape.num_heads = std::atoi(v);
        else if (auto v = val("--kv-heads")) shape.num_kv_heads = std::atoi(v);
        else if (auto v = val("--dim")) shape.head_dim = std::atoi(v);
        else if (auto v = val("--lens")) lens = parse_ints(v);
        else if (auto v = val("--pages")) pages = parse_ints(v);
        else if (auto v = val("--parts")) parts = parse_ints(v);
        else if (auto v = val("--reps")) reps = std::atoi(v);
        else if (auto v = val("--warmup")) warmup = std::atoi(v);
        else if (auto v = val("--cpu-reps")) cpu_reps = std::atoi(v);
        else if (auto v = val("--types")) types = std::strcmp(v, "f32") == 0 ? std::vector{KvType::f32}
                                                 : std::strcmp(v, "f16") == 0 ? std::vector{KvType::f16} : types;
        else if (!std::strcmp(a, "--no-naive")) naive = false;
        else if (!std::strcmp(a, "--no-cpu")) cpu = false;
        else if (!std::strcmp(a, "--no-flush")) flush = false;
        else if (!std::strcmp(a, "--per-head")) per_head = true;  // also time the GQA ablation
        else {
            std::fprintf(stderr, "unknown argument %s\n", a);
            return 2;
        }
    }
    if (!metal_available()) {
        std::fprintf(stderr, "no Metal device\n");
        return 1;
    }
    const int threads = int(std::max(1u, std::thread::hardware_concurrency()));

    std::printf("# device=%s batch=%d heads=%d kv_heads=%d head_dim=%d reps=%d warmup=%d cpu_reps=%d "
                "cpu_threads=%d flush=%d simdgroups=%d\n",
                device_name().c_str(), shape.batch, shape.num_heads, shape.num_kv_heads, shape.head_dim, reps,
                warmup, cpu_reps, threads, flush, kSimdGroups);
    std::printf("impl,dtype,batch,seq_len,page,part,reps,median_us,min_us,max_us,gbps,max_diff_vs_naive\n");

    auto time_gpu = [&](const std::function<double()>& run) {
        for (int i = 0; i < warmup; ++i) run();
        std::vector<double> t;
        for (int i = 0; i < reps; ++i) {
            if (flush) flush_gpu_caches();
            t.push_back(run());
        }
        return stats(t);
    };
    auto row = [&](const char* impl, const Workload& w, int page, int part, int n, Stats s, double diff) {
        std::printf("%s,%s,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.1f,%.3g\n", impl, name(w.type), w.shape.batch, w.max_len,
                    page, part, n, s.median * 1e3, s.min * 1e3, s.max * 1e3, w.min_bytes() / (s.median * 1e-3) / 1e9,
                    diff);
        std::fflush(stdout);
    };

    for (int L : lens)
        for (KvType type : types) {
            Workload w(shape, type, std::vector<int>(size_t(shape.batch), L), uint64_t(L), 3.0f);
            const size_t n = w.out_floats();
            Buffer out_naive(n * sizeof(float)), out_paged(n * sizeof(float));
            if (naive) row("gpu_naive", w, 0, 0, reps, time_gpu([&] { return run_naive(w.naive_args(out_naive)); }), 0);
            if (cpu) {
                std::vector<double> t;
                std::vector<float> out(n);
                for (int i = 0; i <= cpu_reps; ++i) {  // first run is warmup
                    const auto t0 = std::chrono::steady_clock::now();
                    w.reference(out.data(), threads);
                    const std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - t0;
                    if (i) t.push_back(dt.count());
                }
                double diff = NAN;
                if (naive) {
                    diff = 0;
                    for (size_t i = 0; i < n; ++i) diff = std::max(diff, double(std::fabs(out[i] - out_naive.as<float>()[i])));
                }
                row("cpu_ref_fp64", w, 0, 0, cpu_reps, stats(t), diff);
            }
            for (int page : pages) {
                PagedCache cache(w, page, uint64_t(L) * 131 + page);
                for (int part : parts) {
                    const Stats s = time_gpu([&] { return run_paged(cache.args(w, part, out_paged)); });
                    row("gpu_paged", w, page, part ? part : auto_part_tokens(w.shape, w.max_len), reps, s, naive ? max_abs_diff(out_paged, out_naive, n) : NAN);
                    if (per_head) {
                        PagedArgs args = cache.args(w, part, out_paged);
                        args.share_kv = false;
                        const Stats sp = time_gpu([&] { return run_paged(args); });
                        row("gpu_paged_per_head", w, page, part ? part : auto_part_tokens(w.shape, w.max_len), reps,
                            sp, naive ? max_abs_diff(out_paged, out_naive, n) : NAN);
                    }
                }
            }
        }
    return 0;
}
