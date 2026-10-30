// Properties of the CPU reference and the paged layout that hold exactly (no GPU needed).
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

#include "reference.hpp"

using namespace kvb;

namespace {

int failed = 0, passed = 0;

void expect(bool ok, const char* what) {
    ok ? ++passed : ++failed;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

std::vector<float> random_vec(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (auto& e : x) e = u(rng);
    return x;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

}  // namespace

int main() {
    const Shape s{2, 4, 2, 64};
    const std::vector<int> lens{37, 5};
    const int max_len = 37, D = s.head_dim;
    const float scale = 1.0f / std::sqrt(float(D));
    const size_t nkv = size_t(s.batch) * s.num_kv_heads * max_len * D, nq = size_t(s.batch) * s.num_heads * D;
    const auto k = random_vec(nkv, 1), v = random_vec(nkv, 2);

    {  // q = 0: every score is 0, so the output is the plain mean of V over the sequence.
        std::vector<float> q(nq, 0.0f), out(nq), want(nq);
        attention_reference<float>(s, lens, max_len, q.data(), k.data(), v.data(), out.data(), scale);
        for (int b = 0; b < s.batch; ++b)
            for (int h = 0; h < s.num_heads; ++h)
                for (int d = 0; d < D; ++d) {
                    double sum = 0;
                    for (int t = 0; t < lens[b]; ++t) sum += v[contig_index(s, max_len, b, h / s.group(), t, d)];
                    want[(b * s.num_heads + h) * D + d] = float(sum / lens[b]);
                }
        expect(max_diff(out, want) < 1e-6, "zero query gives the mean of V");
    }
    {  // One key aligned with a huge query dominates: the output is that token's V row.
        std::vector<float> q(nq, 0.0f), kk(nkv, 0.0f), out(nq), want(nq);
        for (size_t i = 0; i < nq; i += D) q[i] = 1000.0f;
        const int hot[] = {11, 4};
        for (int b = 0; b < s.batch; ++b)
            for (int kvh = 0; kvh < s.num_kv_heads; ++kvh) kk[contig_index(s, max_len, b, kvh, hot[b], 0)] = 1.0f;
        attention_reference<float>(s, lens, max_len, q.data(), kk.data(), v.data(), out.data(), scale);
        for (int b = 0; b < s.batch; ++b)
            for (int h = 0; h < s.num_heads; ++h)
                for (int d = 0; d < D; ++d)
                    want[(b * s.num_heads + h) * D + d] = v[contig_index(s, max_len, b, h / s.group(), hot[b], d)];
        expect(max_diff(out, want) == 0.0, "a dominant key selects its V row");
    }
    {  // Scores near +-1000 would overflow exp() without the max shift; the output must stay a
       // convex combination of V rows.
        std::vector<float> q = random_vec(nq, 3), out(nq);
        for (auto& e : q) e *= 8000.0f;
        attention_reference<float>(s, lens, max_len, q.data(), k.data(), v.data(), out.data(), scale);
        bool bounded = true;
        for (float o : out) bounded &= std::isfinite(o) && std::fabs(o) <= 1.0f;
        expect(bounded, "huge scores stay finite (max-shifted softmax)");
    }
    {  // GQA: query heads that share a KV head and a query give identical outputs.
        std::vector<float> q = random_vec(nq, 5), out(nq);
        for (int b = 0; b < s.batch; ++b)
            for (int d = 0; d < D; ++d) q[(b * s.num_heads + 1) * D + d] = q[(b * s.num_heads + 0) * D + d];
        attention_reference<float>(s, lens, max_len, q.data(), k.data(), v.data(), out.data(), scale, 3);
        bool same = true;
        for (int b = 0; b < s.batch; ++b)
            for (int d = 0; d < D; ++d)
                same &= out[(b * s.num_heads + 0) * D + d] == out[(b * s.num_heads + 1) * D + d];
        expect(same, "query heads 0 and 1 share KV head 0");
    }
    {  // The block table is a permutation of the pool, and scatter + indexed read is the identity.
        const std::vector<int> plens{1, 16, 17, 100};
        const Shape ps{4, 2, 2, 64};
        const PagedLayout L = make_paged_layout(plens, 16, 7);
        std::set<int> used;
        int live = 0;
        for (size_t b = 0; b < plens.size(); ++b)
            for (int i = 0; i < (plens[b] + 15) / 16; ++i, ++live) used.insert(L.block_table[b * L.max_blocks + i]);
        expect(L.num_blocks == 1 + 1 + 2 + 7 && int(used.size()) == live && *used.rbegin() == L.num_blocks - 1,
               "block table maps logical blocks onto distinct physical blocks");
        const auto src = random_vec(size_t(4) * 2 * 100 * 64, 8);
        std::vector<float> dst(size_t(L.num_blocks) * 2 * 16 * 64, 0.0f);
        scatter_to_paged<float>(ps, L, plens, 100, src.data(), dst.data());
        bool same = true;
        for (int b = 0; b < 4; ++b)
            for (int kvh = 0; kvh < 2; ++kvh)
                for (int t = 0; t < plens[b]; ++t)
                    for (int d = 0; d < 64; ++d)
                        same &= dst[paged_index(ps, L, b, kvh, t, d)] == src[contig_index(ps, 100, b, kvh, t, d)];
        expect(same, "paged scatter round-trips through paged_index");
    }
    {  // fp16 storage: the reference reads half values exactly; result stays near the fp32 one.
        std::vector<half_t> kh(nkv), vh(nkv);
        for (size_t i = 0; i < nkv; ++i) kh[i] = half_t(k[i]), vh[i] = half_t(v[i]);
        std::vector<float> q = random_vec(nq, 9), o32(nq), o16(nq);
        attention_reference<float>(s, lens, max_len, q.data(), k.data(), v.data(), o32.data(), scale);
        attention_reference<half_t>(s, lens, max_len, q.data(), kh.data(), vh.data(), o16.data(), scale);
        const double d = max_diff(o32, o16);
        expect(d > 0 && d < 5e-3, "fp16 storage reference differs from fp32 only by rounding");
    }

    std::printf("%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
