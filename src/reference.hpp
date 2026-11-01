// CPU side: shapes, the paged block table, and the fp64 reference used as ground truth.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kvb {

using half_t = _Float16;

// One decode step: every sequence has one new query token per query head and attends to
// its whole KV history. Query head h reads KV head h / group() (grouped-query attention).
struct Shape {
    int batch = 1;
    int num_heads = 32;     // query heads, H
    int num_kv_heads = 8;   // KV heads, Hkv; H % Hkv == 0
    int head_dim = 128;     // D
    int group() const { return num_heads / num_kv_heads; }
};

// Contiguous ("naive") cache layout: [B][Hkv][max_len][D].
inline size_t contig_index(const Shape& s, int max_len, int b, int kvh, int t, int d) {
    return ((size_t(b) * s.num_kv_heads + kvh) * max_len + t) * s.head_dim + d;
}

// Paged cache: a pool of physical blocks, each [Hkv][page_size][D]. Logical block i of
// sequence b lives in physical block block_table[b * max_blocks + i].
struct PagedLayout {
    int page_size = 16;
    int max_blocks = 0;   // row stride of block_table
    int num_blocks = 0;   // physical blocks in the pool
    std::vector<int32_t> block_table;
};

// Assigns physical blocks in a shuffled order, so the kernel cannot rely on logically
// adjacent blocks being adjacent in memory (as after a long-running allocator).
PagedLayout make_paged_layout(std::span<const int> seq_lens, int page_size, uint64_t seed);

inline size_t paged_index(const Shape& s, const PagedLayout& L, int b, int kvh, int t, int d) {
    const size_t blk = size_t(L.block_table[size_t(b) * L.max_blocks + t / L.page_size]);
    return ((blk * s.num_kv_heads + kvh) * L.page_size + t % L.page_size) * s.head_dim + d;
}

template <class T>
void scatter_to_paged(const Shape& s, const PagedLayout& L, std::span<const int> seq_lens,
                      int max_len, const T* contig, T* paged) {
    for (int b = 0; b < s.batch; ++b)
        for (int kvh = 0; kvh < s.num_kv_heads; ++kvh)
            for (int t = 0; t < seq_lens[b]; ++t)
                for (int d = 0; d < s.head_dim; ++d)
                    paged[paged_index(s, L, b, kvh, t, d)] = contig[contig_index(s, max_len, b, kvh, t, d)];
}

// int8 storage: the page_size x D values of one (physical block, KV head) are stored as int8
// with one fp32 scale, max|x| / 127, so x ~= int8 * scale with an error of at most scale / 2.
// Quantizes the live tokens of the contiguous fp32 cache into `paged` ([blocks][Hkv][page][D])
// and `scales` ([blocks][Hkv]).
void quantize_to_paged_i8(const Shape& s, const PagedLayout& L, std::span<const int> seq_lens, int max_len,
                          const float* contig, int8_t* paged, float* scales);

// The inverse, for the reference: the fp32 values the kernel computes with (int8 * scale), in
// the contiguous layout. Only live tokens are written.
void dequantize_paged_i8(const Shape& s, const PagedLayout& L, std::span<const int> seq_lens, int max_len,
                         const int8_t* paged, const float* scales, float* contig);

// out[b][h][d] = softmax_t(scale * q.k_t) . v_t over the contiguous cache. Accumulates in
// fp64 with a two-pass softmax (max, then exp-sum), so its own error is far below the
// tolerances the GPU kernels are tested against. q and out are [B][H][D].
template <class T>
void attention_reference(const Shape& s, std::span<const int> seq_lens, int max_len,
                         const float* q, const T* k, const T* v, float* out, float scale,
                         int threads = 1);

}  // namespace kvb
