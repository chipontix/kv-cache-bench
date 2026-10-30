#include "reference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>

namespace kvb {

PagedLayout make_paged_layout(std::span<const int> seq_lens, int page_size, uint64_t seed) {
    PagedLayout L;
    L.page_size = page_size;
    for (int len : seq_lens) {
        const int blocks = (len + page_size - 1) / page_size;
        L.max_blocks = std::max(L.max_blocks, blocks);
        L.num_blocks += blocks;
    }
    std::vector<int32_t> phys(size_t(L.num_blocks));
    std::iota(phys.begin(), phys.end(), 0);
    std::shuffle(phys.begin(), phys.end(), std::mt19937_64(seed));

    L.block_table.assign(seq_lens.size() * size_t(L.max_blocks), -1);
    size_t next = 0;
    for (size_t b = 0; b < seq_lens.size(); ++b) {
        const int blocks = (seq_lens[b] + page_size - 1) / page_size;
        for (int i = 0; i < blocks; ++i) L.block_table[b * L.max_blocks + i] = phys[next++];
    }
    return L;
}

template <class T>
void attention_reference(const Shape& s, std::span<const int> seq_lens, int max_len,
                         const float* q, const T* k, const T* v, float* out, float scale,
                         int threads) {
    const int D = s.head_dim, H = s.num_heads, rows = s.batch * H;
    auto one_row = [&](int row, std::vector<double>& p, std::vector<double>& acc) {
        const int b = row / H, h = row % H, kvh = h / s.group(), len = seq_lens[b];
        const float* qr = q + size_t(row) * D;
        p.resize(size_t(len));
        double mx = -INFINITY;
        for (int t = 0; t < len; ++t) {
            const T* kr = k + contig_index(s, max_len, b, kvh, t, 0);
            double dot = 0;
            for (int d = 0; d < D; ++d) dot += double(qr[d]) * double(kr[d]);
            p[t] = dot * scale;
            mx = std::max(mx, p[t]);
        }
        double sum = 0;
        for (int t = 0; t < len; ++t) sum += (p[t] = std::exp(p[t] - mx));
        acc.assign(size_t(D), 0.0);
        for (int t = 0; t < len; ++t) {
            const T* vr = v + contig_index(s, max_len, b, kvh, t, 0);
            for (int d = 0; d < D; ++d) acc[d] += p[t] * double(vr[d]);
        }
        for (int d = 0; d < D; ++d) out[size_t(row) * D + d] = float(acc[d] / sum);
    };

    threads = std::clamp(threads, 1, rows);
    std::vector<std::thread> pool;
    for (int w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            std::vector<double> p, acc;
            for (int row = w; row < rows; row += threads) one_row(row, p, acc);
        });
    for (auto& th : pool) th.join();
}

template void attention_reference<float>(const Shape&, std::span<const int>, int, const float*,
                                         const float*, const float*, float*, float, int);
template void attention_reference<half_t>(const Shape&, std::span<const int>, int, const float*,
                                          const half_t*, const half_t*, float*, float, int);

}  // namespace kvb
