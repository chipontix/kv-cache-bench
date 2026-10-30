// Decode-step attention kernels. Compiled at run time by the host (newLibraryWithSource) with
// these macros: HEAD_DIM (64 or 128), GROUP (query heads per KV head), PAGE_SIZE (tokens per
// block), KV_T (float or half: the storage type of the K/V cache), NSIMD (simdgroups per
// threadgroup in paged_decode). Everything is accumulated in fp32, whatever KV_T is.
#include <metal_stdlib>
using namespace metal;

constant constexpr int D = HEAD_DIM;
constant constexpr int G = GROUP;
constant constexpr int P = PAGE_SIZE;
constant constexpr int DPL = D / 32;  // head-dim elements per SIMD lane
static_assert(D == 64 || D == 128, "HEAD_DIM must be 64 or 128");

// Fast math is on, which lets the compiler assume no infinities, so "minus infinity" is a
// large finite number: exp(NEG_BIG - m) is exactly 0 for any real score m.
constant constexpr float NEG_BIG = -1e30f;

typedef vec<KV_T, DPL> kv_vec;
typedef vec<float, DPL> f_vec;

// Must match kvb::KernelParams on the host.
struct Params {
    int num_kv_heads;
    int max_blocks;   // block_table row stride
    int max_parts;    // partitions per (sequence, head) in the workspace
    int part_tokens;  // tokens per partition (split-K)
    int max_len;      // row stride of the contiguous cache (naive kernels)
    float scale;      // 1/sqrt(D)
};

// ---------------------------------------------------------------------------------------------
// Paged, fused, split-K decode.
// Grid: (max_parts, Hkv, B) threadgroups of 32*NSIMD threads. A threadgroup owns one KV head
// and one partition of part_tokens tokens, and serves all G query heads of that KV head, so
// each K/V row is read once per group. Simdgroup s takes tokens t0+s, t0+s+NSIMD, ...; lane i
// holds head-dim elements [i*DPL, (i+1)*DPL). Each simdgroup keeps an online-softmax state
// (running max m, running sum l, unnormalized acc) per query head, the threadgroup merges the
// NSIMD states, and writes the partition's normalized output plus (m, l) for paged_reduce.
kernel void paged_decode(device const float* q [[buffer(0)]],       // [B][H][D]
                         device const KV_T* k_cache [[buffer(1)]],   // [blocks][Hkv][P][D]
                         device const KV_T* v_cache [[buffer(2)]],
                         device const int* block_table [[buffer(3)]],
                         device const int* seq_lens [[buffer(4)]],
                         device float* part_out [[buffer(5)]],       // [B][H][max_parts][D]
                         device float2* part_ml [[buffer(6)]],       // [B][H][max_parts]
                         constant Params& p [[buffer(7)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const int part = tg.x, kvh = tg.y, b = tg.z;
    const int len = seq_lens[b];
    const int t0 = part * p.part_tokens;
    if (t0 >= len) return;  // uniform over the threadgroup, before any barrier
    const int t1 = min(len, t0 + p.part_tokens);
    const int H = p.num_kv_heads * G;

    f_vec qv[G], acc[G];
    float m[G], l[G];
    for (int g = 0; g < G; ++g) {
        // Fold the softmax scale into q once instead of scaling every score.
        qv[g] = *(device const f_vec*)(q + (b * H + kvh * G + g) * D + lane * DPL) * p.scale;
        acc[g] = 0.0f;
        m[g] = NEG_BIG;
        l[g] = 0.0f;
    }

    device const int* bt = block_table + b * p.max_blocks;
    for (int t = t0 + int(sg); t < t1; t += NSIMD) {
        const size_t row = ((size_t(bt[t / P]) * p.num_kv_heads + kvh) * P + t % P) * D + lane * DPL;
        const f_vec k = f_vec(*(device const kv_vec*)(k_cache + row));
        const f_vec v = f_vec(*(device const kv_vec*)(v_cache + row));
        for (int g = 0; g < G; ++g) {
            const float s = simd_sum(dot(qv[g], k));  // identical in every lane
            const float m_new = max(m[g], s);
            const float corr = exp(m[g] - m_new);     // rescales the old state to the new max
            const float pr = exp(s - m_new);
            l[g] = l[g] * corr + pr;
            acc[g] = acc[g] * corr + pr * v;
            m[g] = m_new;
        }
    }

    // Merge the NSIMD per-simdgroup states. A simdgroup that got no tokens has m = NEG_BIG,
    // l = 0, acc = 0, and its weight exp(NEG_BIG - M) is 0.
    threadgroup float tg_acc[NSIMD][G][D];
    threadgroup float2 tg_ml[NSIMD][G];
    for (int g = 0; g < G; ++g) {
        for (int i = 0; i < DPL; ++i) tg_acc[sg][g][lane * DPL + i] = acc[g][i];
        if (lane == 0) tg_ml[sg][g] = float2(m[g], l[g]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = int(tid); idx < G * D; idx += 32 * NSIMD) {
        const int g = idx / D, d = idx % D;
        float M = NEG_BIG;
        for (int s = 0; s < NSIMD; ++s) M = max(M, tg_ml[s][g].x);
        float L = 0.0f, o = 0.0f;
        for (int s = 0; s < NSIMD; ++s) {
            const float w = exp(tg_ml[s][g].x - M);
            L += tg_ml[s][g].y * w;
            o += tg_acc[s][g][d] * w;
        }
        const int slot = (b * H + kvh * G + g) * p.max_parts + part;
        part_out[slot * D + d] = o / L;
        if (d == 0) part_ml[slot] = float2(M, L);
    }
}

// Merges the partitions of one (sequence, query head). Grid: (H, B) threadgroups of D threads.
kernel void paged_reduce(device const float* part_out [[buffer(0)]],
                         device const float2* part_ml [[buffer(1)]],
                         device const int* seq_lens [[buffer(2)]],
                         device float* out [[buffer(3)]],            // [B][H][D]
                         constant Params& p [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint d [[thread_index_in_threadgroup]]) {
    const int h = tg.x, b = tg.y, H = p.num_kv_heads * G;
    const int parts = (seq_lens[b] + p.part_tokens - 1) / p.part_tokens;
    const int base = (b * H + h) * p.max_parts;
    float M = NEG_BIG;
    for (int i = 0; i < parts; ++i) M = max(M, part_ml[base + i].x);
    float L = 0.0f, o = 0.0f;
    for (int i = 0; i < parts; ++i) {
        const float w = part_ml[base + i].y * exp(part_ml[base + i].x - M);
        L += w;
        o += w * part_out[(base + i) * D + d];
    }
    out[(b * H + h) * D + d] = o / L;
}

// ---------------------------------------------------------------------------------------------
// Naive baseline: contiguous cache [B][Hkv][max_len][D], three passes, scores in global memory.

// One thread per (t, h, b): scores[b][h][t] = scale * q.k_t.
kernel void naive_scores(device const float* q [[buffer(0)]],
                         device const KV_T* k_cache [[buffer(1)]],
                         device const int* seq_lens [[buffer(2)]],
                         device float* scores [[buffer(3)]],         // [B][H][max_len]
                         constant Params& p [[buffer(4)]],
                         uint3 gid [[thread_position_in_grid]]) {
    const int t = gid.x, h = gid.y, b = gid.z, H = p.num_kv_heads * G;
    if (t >= seq_lens[b]) return;
    device const KV_T* k = k_cache + ((size_t(b) * p.num_kv_heads + h / G) * p.max_len + t) * D;
    device const float* qh = q + (b * H + h) * D;
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += qh[d] * float(k[d]);
    scores[size_t(b * H + h) * p.max_len + t] = s * p.scale;
}

// One threadgroup of 256 threads per (h, b): max, exp and sum, normalize, in place.
kernel void naive_softmax(device float* scores [[buffer(0)]],
                          device const int* seq_lens [[buffer(1)]],
                          constant Params& p [[buffer(2)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    const int h = tg.x, b = tg.y, H = p.num_kv_heads * G, len = seq_lens[b];
    device float* row = scores + size_t(b * H + h) * p.max_len;
    threadgroup float red[256];

    float mx = NEG_BIG;
    for (int t = tid; t < len; t += 256) mx = max(mx, row[t]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (int t = tid; t < len; t += 256) {
        const float e = exp(row[t] - mx);
        row[t] = e;
        sum += e;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / red[0];
    for (int t = tid; t < len; t += 256) row[t] *= inv;
}

// One thread per (d, h, b): out[b][h][d] = sum_t p_t * v_t[d].
kernel void naive_pv(device const float* scores [[buffer(0)]],
                     device const KV_T* v_cache [[buffer(1)]],
                     device const int* seq_lens [[buffer(2)]],
                     device float* out [[buffer(3)]],
                     constant Params& p [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]]) {
    const int d = gid.x, h = gid.y, b = gid.z, H = p.num_kv_heads * G, len = seq_lens[b];
    device const float* pr = scores + size_t(b * H + h) * p.max_len;
    device const KV_T* v = v_cache + (size_t(b) * p.num_kv_heads + h / G) * p.max_len * D + d;
    float o = 0.0f;
    for (int t = 0; t < len; ++t) o += pr[t] * float(v[size_t(t) * D]);
    out[(b * H + h) * D + d] = o;
}
