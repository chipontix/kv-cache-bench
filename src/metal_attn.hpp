// Plain C++ interface to the Metal kernels (implemented in metal_attn.mm), so tests and the
// benchmark stay ordinary C++.
#pragma once

#include <cstddef>
#include <string>

#include "reference.hpp"

namespace kvb {

bool metal_available();
std::string device_name();

// A shared-storage MTLBuffer: on Apple silicon the CPU and GPU see the same memory, so the
// host fills it through data() with no copies.
class Buffer {
public:
    explicit Buffer(size_t bytes);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void* data() const;
    size_t size() const { return bytes_; }
    template <class T> T* as() const { return static_cast<T*>(data()); }
    void* handle() const { return impl_; }

private:
    void* impl_ = nullptr;
    size_t bytes_ = 0;
};

enum class KvType { f32, f16 };
inline const char* name(KvType t) { return t == KvType::f32 ? "f32" : "f16"; }
inline size_t elem_size(KvType t) { return t == KvType::f32 ? 4 : 2; }

constexpr int kSimdGroups = 4;  // simdgroups per threadgroup in paged_decode

// Split-K partition size used when PagedArgs::part_tokens is 0: the largest power of two in
// [64, 512] that still yields at least 512 threadgroups (B * Hkv * partitions). Short contexts
// need small partitions to fill the GPU; long ones prefer fewer, longer partitions (less merge
// work). Picked from the partition sweep in results/.
inline int auto_part_tokens(const Shape& s, int max_len) {
    int part = 512;
    while (part > 64 && size_t(s.batch) * s.num_kv_heads * ((max_len + part - 1) / part) < 512) part /= 2;
    return part;
}

struct PagedArgs {
    Shape shape;
    KvType type = KvType::f32;
    int page_size = 16;
    int max_blocks = 0;      // PagedLayout::max_blocks
    int max_len = 0;         // longest sequence, sizes the split-K workspace
    int part_tokens = 0;     // split-K partition size, 0 = auto_part_tokens()
    const Buffer* q;         // float [B][H][D]
    const Buffer* k;         // type  [blocks][Hkv][page][D]
    const Buffer* v;
    const Buffer* block_table;  // int32 [B][max_blocks]
    const Buffer* seq_lens;     // int32 [B]
    const Buffer* out;          // float [B][H][D]
    bool share_kv = true;       // false: one threadgroup per query head (GQA ablation)
};

struct NaiveArgs {
    Shape shape;
    KvType type = KvType::f32;
    int max_len = 0;   // row stride of the contiguous cache
    const Buffer* q;
    const Buffer* k;   // type [B][Hkv][max_len][D]
    const Buffer* v;
    const Buffer* seq_lens;
    const Buffer* out;
};

// Each call encodes the kernels into one command buffer, waits for it, and returns its GPU
// time in milliseconds (GPUEndTime - GPUStartTime). Shader compilation happens on the first
// call for a configuration and is not part of the returned time.
double run_paged(const PagedArgs&);
double run_naive(const NaiveArgs&);

// Reads `src` once with a plain streaming kernel; returns GPU milliseconds.
double run_read_bandwidth(const Buffer& src);

// Overwrites a 256 MiB scratch buffer on the GPU, to evict the cache from the GPU's last
// level caches before a timed run (best effort: the cache policy is not documented).
void flush_gpu_caches();

}  // namespace kvb
