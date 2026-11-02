// Length boundaries of the paged cache: empty, 1, exactly a page, a partition or the capacity, and
// one past them. The grid of gpu_tests has no length that is a multiple of a page or a partition,
// and no empty sequence. Exit code 77 (CTest "skipped") when there is no Metal device; the checks
// that need no GPU have run by then.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "workload.hpp"

using namespace kvb;

namespace {

int checks = 0, failed = 0;

void expect(bool ok, const char* what) {
    ++checks;
    if (ok) return;
    ++failed;
    std::printf("FAIL %s\n", what);
}

int finish() {
    std::printf("%d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}

constexpr double kTolKernel = 2e-5;  // the tolerance of gpu_tests (README, "Tolerances")
constexpr KvType kTypes[] = {KvType::f32, KvType::f16, KvType::i8};

Workload make_workload(const Shape& shape, KvType type, std::vector<int> lens, uint64_t seed = 1) {
    return Workload(shape, type, std::move(lens), seed, 3.0f, false, type == KvType::i8);
}

void expect_close(const float* got, const float* want, const Workload& w, const char* what, int page, int part) {
    const double err = max_abs_diff(got, want, w.out_floats());
    if (err <= kTolKernel) return expect(true, what);
    std::string lens;
    for (int l : w.lens) lens += std::to_string(l) + " ";
    std::printf("  H=%d Hkv=%d D=%d %s lens=%spage=%d part=%d: err %.3g\n", w.shape.num_heads, w.shape.num_kv_heads,
                w.shape.head_dim, name(w.type), lens.c_str(), page, part, err);
    expect(false, what);
}

// The paged kernel (and, with a GQA shape, its per-query-head variant) against the fp64 reference.
void check_paged(const Workload& w, int page, int part, bool contiguous = false) {
    const size_t n = w.out_floats();
    const PagedCache cache(w, page, uint64_t(page) * 1000 + part, contiguous);
    std::vector<float> ref(n);
    cache.reference(w, ref.data());
    Buffer out(n * sizeof(float));
    PagedArgs args = cache.args(w, part, out);
    std::fill_n(out.as<float>(), n, NAN);
    run_paged(args);
    expect_close(out.as<float>(), ref.data(), w, contiguous ? "paged-contiguous" : "paged", cache.layout.page_size, part);
    if (w.shape.group() == 1) return;
    args.share_kv = false;
    std::fill_n(out.as<float>(), n, NAN);
    run_paged(args);
    expect_close(out.as<float>(), ref.data(), w, "paged-per-head", cache.layout.page_size, part);
}

void check_naive(const Workload& w) {
    const size_t n = w.out_floats();
    std::vector<float> ref(n);
    w.reference(ref.data());
    Buffer out(n * sizeof(float));
    std::fill_n(out.as<float>(), n, NAN);
    run_naive(w.naive_args(out));
    expect_close(out.as<float>(), ref.data(), w, "naive", 0, 0);
}

}  // namespace

int main() {
    {  // The comparison every test goes through: a NaN is a difference, wherever it is.
        const float a[] = {1.0f, NAN, 3.0f}, b[] = {1.0f, 2.0f, 3.0f}, c[] = {1.0f, 2.0f, 3.5f};
        expect(std::isinf(max_abs_diff(a, b, 3)), "max_abs_diff: NaN on one side counts as infinite");
        expect(std::isinf(max_abs_diff(a, a, 3)), "max_abs_diff: NaN on both sides counts as infinite");
        expect(max_abs_diff(b, c, 3) == 0.5, "max_abs_diff: finite values");
    }
    {  // auto_part_tokens: "at least 512 threadgroups" on both sides of each threshold, with 16
       // (sequence, KV head) pairs, so 32 partitions; 64 is the floor and 512 the ceiling.
        const Shape s{2, 32, 8, 128};
        const int want[][2] = {{0, 64},      {1, 64},       {3968, 64},    {3969, 128},  {7936, 128},
                               {7937, 256},  {15872, 256},  {15873, 512},  {1 << 20, 512}};
        for (const auto& [len, part] : want) expect(auto_part_tokens(s, len) == part, "auto_part_tokens threshold");
    }
    {  // An empty sequence attends to nothing: its output is exactly 0 and its neighbour's is untouched.
        const Shape s{3, 4, 2, 64};
        const std::vector<int> lens{0, 9, 0}, alone{9};
        const size_t nkv = size_t(3) * 2 * 9 * 64, nq = size_t(3) * 4 * 64, row = size_t(4) * 64;
        std::vector<float> q(nq, 0.5f), k(nkv, 0.25f), v(nkv), out(nq, NAN), one(row);
        for (size_t i = 0; i < nkv; ++i) v[i] = float(i % 7) / 7;
        attention_reference<float>(s, lens, 9, q.data(), k.data(), v.data(), out.data(), 0.125f);
        attention_reference<float>({1, 4, 2, 64}, alone, 9, q.data(), k.data() + size_t(2) * 9 * 64,
                                   v.data() + size_t(2) * 9 * 64, one.data(), 0.125f);
        const std::vector<float> zero(row, 0.0f);
        expect(max_abs_diff(out.data(), zero.data(), row) == 0 && max_abs_diff(out.data() + 2 * row, zero.data(), row) == 0,
               "reference: an empty sequence gives exactly 0");
        expect(max_abs_diff(out.data() + row, one.data(), row) == 0, "reference: an empty neighbour changes nothing");
        const PagedLayout L = make_paged_layout(lens, 16, 3), none = make_paged_layout(std::vector<int>{0, 0}, 16, 3);
        expect(L.num_blocks == 1 && L.max_blocks == 1 && L.block_table[0] == -1 && L.block_table[1] == 0 && L.block_table[2] == -1,
               "layout: an empty sequence takes no block");
        expect(none.num_blocks == 0 && none.max_blocks == 0 && none.block_table.empty(), "layout: an all-empty batch has no blocks");
    }
    if (!metal_available()) {
        std::puts("SKIP: no Metal device on this machine, GPU boundary tests not run");
        return failed ? 1 : 77;
    }

    const Shape shapes[] = {{1, 8, 2, 128}, {1, 4, 4, 64}};
    // Every length next to a page (1, 3: not a power of two, and the shift-free path), a partition (1, 3, 4:
    // no more tokens than simdgroups; 0: auto), or the simdgroup count, as {L, L + 1}: L = kP is a
    // sequence that fills its block table exactly, L + 1 opens a block with one token.
    for (const Shape& shape : shapes)
        for (KvType type : kTypes)
            for (int page : {1, 3, 16, 64})
                for (int part : {0, 1, 3, 4, 64}) {
                    const int K = part ? part : 64;
                    std::vector<int> lens{1, 2, 3, 4, 5, page - 1, page, page + 1, 2 * page, K - 1, K, K + 1, 2 * K};
                    std::sort(lens.begin(), lens.end());
                    lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
                    for (int L : lens)
                        if (L > 0) check_paged(make_workload(shape, type, {L, L + 1}), page, part);
                }
    // Partitions next to the threshold of auto_part_tokens, and the kernel using what it picks.
    for (int L : {3968, 3969}) check_paged(make_workload({2, 8, 8, 128}, KvType::f16, {L, L}), 16, 0);
    // One block per sequence (page = max_len, the layout of bench-compare), of any length.
    for (const Shape& shape : shapes)
        for (KvType type : kTypes)
            for (int L : {1, 2, 16, 17, 64, 65, 256}) {
                check_paged(make_workload(shape, type, {L}), L, 0, true);
                check_paged(make_workload(shape, type, {L, L + 1}), L + 1, 64, true);
            }
    // The naive kernels work in threadgroups of 256 threads.
    for (const Shape& shape : shapes)
        for (KvType type : {KvType::f32, KvType::f16})
            for (int L : {1, 2, 3, 4, 5, 255, 256, 257, 512, 513}) check_naive(make_workload(shape, type, {L, L + 1}));
    {  // Capacity: the host refuses a length the buffers cannot hold, before anything reaches the GPU.
       // Page 16, 2 blocks per sequence: capacity 32.
        const Workload w = make_workload({1, 8, 2, 128}, KvType::f32, {32, 32});
        const PagedCache cache(w, 16, 5);
        const size_t n = w.out_floats();
        std::vector<float> ref(n);
        w.reference(ref.data());
        Buffer out(n * sizeof(float)), lens(2 * sizeof(int));
        auto refused = [&](int l0, int l1, int max_len, int page = 16) {
            lens.as<int>()[0] = l0, lens.as<int>()[1] = l1;
            PagedArgs a = cache.args(w, 16, out);
            a.seq_lens = &lens, a.max_len = max_len, a.page_size = page;
            std::fill_n(out.as<float>(), n, NAN);
            try {
                run_paged(a);
            } catch (const std::invalid_argument&) {
                return std::all_of(out.as<float>(), out.as<float>() + n, [](float x) { return std::isnan(x); });
            }
            return false;
        };
        expect(!refused(32, 32, 32) && max_abs_diff(out.as<float>(), ref.data(), n) <= kTolKernel, "paged: exactly the capacity runs");
        expect(refused(33, 33, 64), "paged: capacity + 1 is refused, whatever max_len says");
        expect(refused(32, 33, 64), "paged: capacity + 1 in the last sequence alone is refused");
        expect(refused(33, 33, 32), "paged: longer than max_len and capacity is refused");
        expect(refused(32, 32, 31), "paged: longer than max_len is refused");
        expect(refused(-1, 32, 32), "paged: a negative length is refused");
        expect(refused(32, 32, -1), "paged: a negative max_len is refused");
        expect(refused(32, 32, 32, 0), "paged: page size 0 is refused");
        expect(!refused(32, 32, 32) && max_abs_diff(out.as<float>(), ref.data(), n) <= kTolKernel, "paged: works after a refusal");

        auto refused_naive = [&](int l0, int l1, int max_len) {
            lens.as<int>()[0] = l0, lens.as<int>()[1] = l1;
            NaiveArgs a = w.naive_args(out);
            a.seq_lens = &lens, a.max_len = max_len;
            std::fill_n(out.as<float>(), n, NAN);
            try {
                run_naive(a);
            } catch (const std::invalid_argument&) {
                return std::all_of(out.as<float>(), out.as<float>() + n, [](float x) { return std::isnan(x); });
            }
            return false;
        };
        expect(!refused_naive(32, 32, 32) && max_abs_diff(out.as<float>(), ref.data(), n) <= kTolKernel, "naive: exactly max_len runs");
        expect(refused_naive(32, 33, 32), "naive: max_len + 1 is refused");
        expect(refused_naive(-1, 32, 32), "naive: a negative length is refused");
    }
    // Empty sequences next to live ones, first, last and alone. Besides the reference, each kernel
    // must write exactly 0: the reference could be wrong the same way.
    const std::vector<std::vector<int>> batches = {{0, 1}, {1, 0}, {0, 16}, {17, 0}, {0, 65, 0}, {0}, {0, 0}};
    for (const Shape& shape : shapes)
        for (KvType type : kTypes)
            for (const std::vector<int>& lens : batches) {
                const Workload w = make_workload(shape, type, lens);
                for (int part : {0, 64}) check_paged(w, 16, part);
                check_paged(w, 16, 64, true);
                if (type != KvType::i8) check_naive(w);
                const size_t n = w.out_floats(), row = size_t(shape.num_heads) * shape.head_dim;
                const PagedCache cache(w, 16, 5);
                Buffer out(n * sizeof(float));
                std::fill_n(out.as<float>(), n, NAN);
                run_paged(cache.args(w, 0, out));
                const std::vector<float> zero(row, 0.0f);
                for (size_t b = 0; b < lens.size(); ++b)
                    if (lens[b] == 0) expect(max_abs_diff(out.as<float>() + b * row, zero.data(), row) == 0, "paged: an empty sequence gives exactly 0");
            }
    return finish();
}
