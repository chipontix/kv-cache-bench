#import <Metal/Metal.h>

#include "metal_attn.hpp"

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "kernels_src.inc"  // kKernelSource, generated from kernels.metal by CMake

namespace kvb {
namespace {

// Must match struct Params in kernels.metal.
struct KernelParams {
    int num_kv_heads;
    int max_blocks;
    int max_parts;
    int part_tokens;
    int max_len;
    float scale;
};

id<MTLDevice> device() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

id<MTLCommandQueue> queue() {
    static id<MTLCommandQueue> q = [device() newCommandQueueWithMaxCommandBufferCount:512];
    return q;
}

// Pipelined mode (begin_pipelined / end_pipelined): command buffers are committed without
// waiting, and each one waits on an event the previous one signals, so they run strictly one
// after another with no CPU round trip in between.
bool g_pipelined = false;
uint64_t g_event_value = 0;
std::vector<id<MTLCommandBuffer>> g_timed;

id<MTLEvent> event() {
    static id<MTLEvent> ev = [device() newEvent];
    return ev;
}

id<MTLCommandBuffer> new_cb() {
    id<MTLCommandBuffer> cb = [queue() commandBuffer];
    if (g_pipelined) [cb encodeWaitForEvent:event() value:g_event_value];
    return cb;
}

id<MTLBuffer> mtl(const Buffer* b) { return (__bridge id<MTLBuffer>)b->handle(); }

[[noreturn]] void fail(const std::string& what, NSError* err) {
    throw std::runtime_error(what + (err ? std::string(": ") + err.localizedDescription.UTF8String : ""));
}

// One library per macro set (head_dim, group, page, type); pipelines cached by name too.
id<MTLComputePipelineState> pipeline(const char* fn, const Shape& s, int page, KvType t, int qg = 0) {
    static std::map<std::string, id<MTLLibrary>> libs;
    static std::map<std::string, id<MTLComputePipelineState>> pipes;
    const std::string lib_key = std::to_string(s.head_dim) + "/" + std::to_string(s.group()) + "/" +
                                std::to_string(page) + "/" + name(t) + "/" + std::to_string(qg);
    const std::string key = lib_key + "/" + fn;
    if (auto it = pipes.find(key); it != pipes.end()) return it->second;

    id<MTLLibrary> lib = libs[lib_key];
    if (!lib) {
        if (s.head_dim != 64 && s.head_dim != 128) throw std::invalid_argument("head_dim must be 64 or 128");
        if (s.num_heads % s.num_kv_heads != 0) throw std::invalid_argument("H % Hkv != 0");
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.preprocessorMacros = @{
            @"HEAD_DIM" : @(s.head_dim),
            @"GROUP" : @(s.group()),
            @"PAGE_SIZE" : @(page),
            @"KV_T" : t == KvType::f32 ? @"float" : @"half",
            @"NSIMD" : @(kSimdGroups),
            @"QG" : @(qg ? qg : s.group()),
        };
        NSError* err = nil;
        lib = [device() newLibraryWithSource:@(kKernelSource) options:opt error:&err];
        if (!lib) fail("Metal compile failed", err);
        libs[lib_key] = lib;
    }
    id<MTLFunction> f = [lib newFunctionWithName:@(fn)];
    NSError* err = nil;
    id<MTLComputePipelineState> ps = [device() newComputePipelineStateWithFunction:f error:&err];
    if (!ps) fail(std::string("pipeline ") + fn, err);
    pipes[key] = ps;
    return ps;
}

// Grow-only scratch buffers for the split-K partials and the naive scores.
const Buffer& workspace(int slot, size_t bytes) {
    static std::unique_ptr<Buffer> bufs[3];
    if (!bufs[slot] || bufs[slot]->size() < bytes) {
        bufs[slot].reset();
        bufs[slot] = std::make_unique<Buffer>(bytes);
    }
    return *bufs[slot];
}

double finish(id<MTLCommandBuffer> cb, bool timed = true) {
    if (g_pipelined) {
        [cb encodeSignalEvent:event() value:++g_event_value];
        [cb commit];
        if (timed) g_timed.push_back(cb);
        return 0;
    }
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) fail("command buffer failed", cb.error);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
}

}  // namespace

bool metal_available() { return device() != nil; }
std::string device_name() { return device() ? device().name.UTF8String : ""; }

// [MTLBuffer contents] retains and autoreleases the buffer, and a plain C++ main() has no pool
// to drain it: calling it on every data() leaked every buffer (8 GB peak footprint for the
// sweep, results/peak-rss.txt). So the pointer is taken once, inside a pool, and cached.
Buffer::Buffer(size_t bytes) : bytes_(bytes) {
    @autoreleasepool {
        id<MTLBuffer> b = [device() newBufferWithLength:std::max<size_t>(bytes, 16)
                                                options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("MTLBuffer allocation failed (" + std::to_string(bytes) + " bytes)");
        impl_ = (__bridge_retained void*)b;
        data_ = b.contents;
    }
}

Buffer::~Buffer() {
    @autoreleasepool {
        if (impl_) CFRelease(impl_);  // balances __bridge_retained in the constructor
    }
}


double run_paged(const PagedArgs& a) {
    @autoreleasepool {  // Metal objects are autoreleased; drain them per call
        const Shape& s = a.shape;
        if (a.part_tokens < 0 || a.max_len <= 0) throw std::invalid_argument("part_tokens < 0 or max_len <= 0");
        const int part_tokens = a.part_tokens ? a.part_tokens : auto_part_tokens(s, a.max_len);
        const int max_parts = (a.max_len + part_tokens - 1) / part_tokens;
        const size_t slots = size_t(s.batch) * s.num_heads * max_parts;
        const Buffer& part_out = workspace(0, slots * s.head_dim * sizeof(float));
        const Buffer& part_ml = workspace(1, slots * 2 * sizeof(float));
        const KernelParams prm{s.num_kv_heads, a.max_blocks, max_parts, part_tokens, a.max_len,
                               1.0f / std::sqrt(float(s.head_dim))};

        const int qg = a.share_kv ? s.group() : 1;
        id<MTLComputePipelineState> decode = pipeline("paged_decode", s, a.page_size, a.type, qg);
        id<MTLComputePipelineState> reduce = pipeline("paged_reduce", s, a.page_size, a.type, qg);
        id<MTLCommandBuffer> cb = new_cb();
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];  // serial: reduce sees decode's writes
        [e setComputePipelineState:decode];
        [e setBuffer:mtl(a.q) offset:0 atIndex:0];
        [e setBuffer:mtl(a.k) offset:0 atIndex:1];
        [e setBuffer:mtl(a.v) offset:0 atIndex:2];
        [e setBuffer:mtl(a.block_table) offset:0 atIndex:3];
        [e setBuffer:mtl(a.seq_lens) offset:0 atIndex:4];
        [e setBuffer:mtl(&part_out) offset:0 atIndex:5];
        [e setBuffer:mtl(&part_ml) offset:0 atIndex:6];
        [e setBytes:&prm length:sizeof prm atIndex:7];
        [e dispatchThreadgroups:MTLSizeMake(max_parts, s.num_heads / qg, s.batch)
            threadsPerThreadgroup:MTLSizeMake(32 * kSimdGroups, 1, 1)];
        [e setComputePipelineState:reduce];
        [e setBuffer:mtl(&part_out) offset:0 atIndex:0];
        [e setBuffer:mtl(&part_ml) offset:0 atIndex:1];
        [e setBuffer:mtl(a.seq_lens) offset:0 atIndex:2];
        [e setBuffer:mtl(a.out) offset:0 atIndex:3];
        [e setBytes:&prm length:sizeof prm atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(s.num_heads, s.batch, 1)
            threadsPerThreadgroup:MTLSizeMake(s.head_dim, 1, 1)];
        [e endEncoding];
        return finish(cb);
    }
}

double run_naive(const NaiveArgs& a) {
    @autoreleasepool {  // Metal objects are autoreleased; drain them per call
        const Shape& s = a.shape;
        const Buffer& scores = workspace(2, size_t(s.batch) * s.num_heads * a.max_len * sizeof(float));
        const KernelParams prm{s.num_kv_heads, 0, 0, 0, a.max_len, 1.0f / std::sqrt(float(s.head_dim))};

        id<MTLComputePipelineState> ps_scores = pipeline("naive_scores", s, 16, a.type);
        id<MTLComputePipelineState> ps_softmax = pipeline("naive_softmax", s, 16, a.type);
        id<MTLComputePipelineState> ps_pv = pipeline("naive_pv", s, 16, a.type);
        id<MTLCommandBuffer> cb = new_cb();
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps_scores];
        [e setBuffer:mtl(a.q) offset:0 atIndex:0];
        [e setBuffer:mtl(a.k) offset:0 atIndex:1];
        [e setBuffer:mtl(a.seq_lens) offset:0 atIndex:2];
        [e setBuffer:mtl(&scores) offset:0 atIndex:3];
        [e setBytes:&prm length:sizeof prm atIndex:4];
        [e dispatchThreads:MTLSizeMake(a.max_len, s.num_heads, s.batch) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:ps_softmax];
        [e setBuffer:mtl(&scores) offset:0 atIndex:0];
        [e setBuffer:mtl(a.seq_lens) offset:0 atIndex:1];
        [e setBytes:&prm length:sizeof prm atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(s.num_heads, s.batch, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e setComputePipelineState:ps_pv];
        [e setBuffer:mtl(&scores) offset:0 atIndex:0];
        [e setBuffer:mtl(a.v) offset:0 atIndex:1];
        [e setBuffer:mtl(a.seq_lens) offset:0 atIndex:2];
        [e setBuffer:mtl(a.out) offset:0 atIndex:3];
        [e setBytes:&prm length:sizeof prm atIndex:4];
        [e dispatchThreads:MTLSizeMake(s.head_dim, s.num_heads, s.batch)
            threadsPerThreadgroup:MTLSizeMake(s.head_dim, 1, 1)];
        [e endEncoding];
        return finish(cb);
    }
}

double run_read_bandwidth(const Buffer& src) {
    @autoreleasepool {  // Metal objects are autoreleased; drain them per call
        constexpr uint32_t kThreads = 1u << 18;
        const Buffer& sink = workspace(2, kThreads * sizeof(float));
        const uint32_t n4 = uint32_t(src.size() / 16);
        id<MTLCommandBuffer> cb = new_cb();
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pipeline("read_bw", Shape{}, 16, KvType::f32)];
        [e setBuffer:mtl(&src) offset:0 atIndex:0];
        [e setBuffer:mtl(&sink) offset:0 atIndex:1];
        [e setBytes:&n4 length:sizeof n4 atIndex:2];
        [e dispatchThreads:MTLSizeMake(kThreads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        return finish(cb);
    }
}

void flush_gpu_caches() {
    @autoreleasepool {  // Metal objects are autoreleased; drain them per call
        static Buffer scratch(256u << 20);
        id<MTLCommandBuffer> cb = new_cb();
        id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
        static uint8_t value = 0;
        [e fillBuffer:mtl(&scratch) range:NSMakeRange(0, scratch.size()) value:++value];
        [e endEncoding];
        finish(cb, false);
    }
}

void begin_pipelined() {
    g_pipelined = true;
    g_timed.clear();
}

std::vector<double> end_pipelined() {
    g_pipelined = false;
    std::vector<double> ms;
    for (id<MTLCommandBuffer> cb : g_timed) {
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) fail("command buffer failed", cb.error);
        ms.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e3);
    }
    g_timed.clear();
    return ms;
}

}  // namespace kvb
