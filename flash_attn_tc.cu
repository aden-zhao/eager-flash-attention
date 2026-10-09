#include "flash_attn_tc.cuh"

#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p)
{
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr(smem)), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Four 8x8 b16 matrices; lane l supplies the row address for row l%8 of matrix l/8.
// Result r_i for lane t: elements (t/4, 2*(t%4)), (t/4, 2*(t%4)+1) of matrix i.
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(smem)));
}

// Same, transposed: lane t receives elements (2*(t%4), t/4), (2*(t%4)+1, t/4).
__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(smem)));
}

// D[16x8] (+)= A[16x16] * B[16x8], fp16 inputs, fp32 accumulate.
// A regs: a0 (row g, cols 2c..2c+1), a1 (row g+8, same), a2 (row g, cols 8+2c..), a3 (row g+8, cols 8+2c..)
// B regs: b0 (rows 2c..2c+1, col g), b1 (rows 8+2c.., col g)          where g = lane/4, c = lane%4
// C regs: c0,c1 (row g, cols 2c..2c+1), c2,c3 (row g+8, same cols)
__device__ __forceinline__ void mma_16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi)
{
    half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int D>
__global__ void __launch_bounds__(128)
flash_attn_tc_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    void*       __restrict__ O_void,
    FlashAttnTCPartials part,
    int q_start, int S, int H_local, int nsplit, int kv_len, int causal, int out_bf16,
    float scale_log2e, int* __restrict__ tile_counters, int tile_rows)
{
    constexpr int BM  = FLASH_TC_BM;
    constexpr int BN  = FLASH_TC_BN;
    constexpr int LDS = D + 8;            // padded smem row stride (halfs): keeps ldmatrix bank-conflict free
    constexpr int KT  = D / 16;           // k-steps for Q K^T
    constexpr int NT  = BN / 8;           // 8-wide n-tiles of S per warp
    constexpr int DT  = D / 8;            // 8-wide n-tiles of O per warp
    constexpr int CHUNK = BN * LDS;       // halfs per K (or V) stage

    extern __shared__ __align__(16) half smem[];
    half* Ks[2] = { smem,             smem + CHUNK     };
    half* Vs[2] = { smem + 2 * CHUNK, smem + 3 * CHUNK };

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane >> 2;           // row group within mma fragment
    const int c    = lane & 3;            // column pair within mma fragment
    // x enumerates (head, split) pairs and is the fastest-varying dispatch index,
    // so all blocks of a query-row block finish before the next row block starts.
    const int head  = blockIdx.x % H_local;
    const int split = blockIdx.x / H_local;
    const int block_row0 = blockIdx.y * BM;                  // first row of this block within the launch
    const int local_row0 = block_row0 + warp * 16;           // first row of this warp within the launch
    const int q_abs0 = q_start + block_row0;                 // absolute index of the block's first row
    const int kv0 = split * kv_len;                          // this block's key range
    int kv_end = kv0 + kv_len;
    if (causal) kv_end = min(kv_end, q_abs0 + BM);           // rows are 64-aligned: no keys past the block
    const int q0 = q_start + local_row0;

    const size_t head_off = (size_t)head * S * D;
    const half* Qh = Q + head_off;
    const half* Kh = K + head_off;
    const half* Vh = V + head_off;

    // Q fragments for all k-steps, loaded once (A operand layout).
    uint32_t qf[KT][4];
    #pragma unroll
    for (int kk = 0; kk < KT; kk++) {
        const half* r0 = Qh + (size_t)(q0 + g) * D + kk * 16 + 2 * c;
        const half* r1 = r0 + 8 * D;
        qf[kk][0] = *reinterpret_cast<const uint32_t*>(r0);
        qf[kk][1] = *reinterpret_cast<const uint32_t*>(r1);
        qf[kk][2] = *reinterpret_cast<const uint32_t*>(r0 + 8);
        qf[kk][3] = *reinterpret_cast<const uint32_t*>(r1 + 8);
    }

    float o[DT][4];
    #pragma unroll
    for (int i = 0; i < DT; i++) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f; }
    float m[2] = { -INFINITY, -INFINITY };   // running max for rows g and g+8 (log2 domain)
    float l[2] = { 0.f, 0.f };               // running denominator

    auto load_chunk = [&](int stage, int kv0) {
        constexpr int CPR   = D / 8;          // 16-byte pieces per row
        constexpr int TOTAL = BN * CPR;
        #pragma unroll
        for (int i = threadIdx.x; i < TOTAL; i += 128) {
            int r   = i / CPR;
            int col = (i % CPR) * 8;
            cp_async16(Ks[stage] + r * LDS + col, Kh + (size_t)(kv0 + r) * D + col);
            cp_async16(Vs[stage] + r * LDS + col, Vh + (size_t)(kv0 + r) * D + col);
        }
        cp_async_commit();
    };

    const int nchunks = (kv_end > kv0) ? (kv_end - kv0) / BN : 0;
    if (nchunks > 0) load_chunk(0, kv0);

    for (int j = 0; j < nchunks; j++) {
        const int st = j & 1;
        const int chunk0 = kv0 + j * BN;
        if (j + 1 < nchunks) {
            load_chunk(st ^ 1, chunk0 + BN);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // ---- S = Q K^T  (16 x BN per warp) ----
        float s[NT][4];
        #pragma unroll
        for (int i = 0; i < NT; i++) { s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f; }

        #pragma unroll
        for (int kk = 0; kk < KT; kk++) {
            #pragma unroll
            for (int nt = 0; nt < NT; nt += 2) {
                // matrices: 0 -> (rows nt*8.., cols kk*16..)   -> b0 of tile nt
                //           1 -> (rows nt*8.., cols kk*16+8..) -> b1 of tile nt
                //           2 -> (rows nt*8+8.., cols kk*16..) -> b0 of tile nt+1
                //           3 -> (rows nt*8+8.., cols kk*16+8) -> b1 of tile nt+1
                const int mi  = lane >> 3, ri = lane & 7;
                const int row = nt * 8 + ri + ((mi & 2) ? 8 : 0);
                const int col = kk * 16 + ((mi & 1) ? 8 : 0);
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, Ks[st] + row * LDS + col);
                mma_16816(s[nt],     qf[kk], b0, b1);
                mma_16816(s[nt + 1], qf[kk], b2, b3);
            }
        }

        // ---- causal mask on the diagonal chunk (key k valid for row i iff k <= i) ----
        if (causal && chunk0 == q_abs0) {
            #pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int k   = nt * 8 + 2 * c + (i & 1);
                    const int row = warp * 16 + g + ((i & 2) ? 8 : 0);
                    if (k > row) s[nt][i] = -INFINITY;
                }
            }
        }

        // ---- online softmax (base-2 exponent; scale folded into scale_log2e) ----
        float mx[2] = { -INFINITY, -INFINITY };
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                s[nt][i] *= scale_log2e;
                mx[i >> 1] = fmaxf(mx[i >> 1], s[nt][i]);
            }
        }
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
        }

        float corr[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            float m_new = fmaxf(m[r], mx[r]);
            corr[r] = exp2f(m[r] - m_new);
            m[r] = m_new;
        }

        float rs[2] = { 0.f, 0.f };
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                s[nt][i] = exp2f(s[nt][i] - m[i >> 1]);
                rs[i >> 1] += s[nt][i];
            }
        }
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            rs[r] += __shfl_xor_sync(0xffffffff, rs[r], 1);
            rs[r] += __shfl_xor_sync(0xffffffff, rs[r], 2);
            l[r] = l[r] * corr[r] + rs[r];
        }
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) {
            o[dt][0] *= corr[0]; o[dt][1] *= corr[0];
            o[dt][2] *= corr[1]; o[dt][3] *= corr[1];
        }

        // ---- O += P V ----
        #pragma unroll
        for (int kk = 0; kk < BN / 16; kk++) {
            // P fragment for K rows kk*16 .. kk*16+15: two adjacent S n-tiles form one A operand.
            uint32_t pa[4] = {
                pack_half2(s[2 * kk][0],     s[2 * kk][1]),
                pack_half2(s[2 * kk][2],     s[2 * kk][3]),
                pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3]),
            };
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                // matrices: 0 -> (V rows kk*16..,   cols dt*8..)   -> b0 of tile dt
                //           1 -> (V rows kk*16+8.., cols dt*8..)   -> b1 of tile dt
                //           2 -> (V rows kk*16..,   cols dt*8+8..) -> b0 of tile dt+1
                //           3 -> (V rows kk*16+8.., cols dt*8+8..) -> b1 of tile dt+1
                const int mi  = lane >> 3, ri = lane & 7;
                const int row = kk * 16 + ri + ((mi & 1) ? 8 : 0);
                const int col = dt * 8 + ((mi & 2) ? 8 : 0);
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4_trans(b0, b1, b2, b3, Vs[st] + row * LDS + col);
                mma_16816(o[dt],     pa, b0, b1);
                mma_16816(o[dt + 1], pa, b2, b3);
            }
        }

        __syncthreads();   // all warps done with stage `st` before it is refilled
    }

    // ---- epilogue ----
    const size_t ld_o = (size_t)H_local * D;
    if (nsplit == 1) {
        // normalize and store (fp32 or bf16)
        const float inv0 = 1.f / l[0];
        const float inv1 = 1.f / l[1];
        const size_t off0 = (size_t)(local_row0 + g) * ld_o + head * D;
        const size_t off1 = off0 + 8 * ld_o;
        if (out_bf16) {
            __nv_bfloat16* Ob = reinterpret_cast<__nv_bfloat16*>(O_void);
            #pragma unroll
            for (int dt = 0; dt < DT; dt++) {
                const int col = dt * 8 + 2 * c;
                *reinterpret_cast<__nv_bfloat162*>(Ob + off0 + col) = __floats2bfloat162_rn(o[dt][0] * inv0, o[dt][1] * inv0);
                *reinterpret_cast<__nv_bfloat162*>(Ob + off1 + col) = __floats2bfloat162_rn(o[dt][2] * inv1, o[dt][3] * inv1);
            }
        } else {
            float* O = reinterpret_cast<float*>(O_void);
            #pragma unroll
            for (int dt = 0; dt < DT; dt++) {
                const int col = dt * 8 + 2 * c;
                *reinterpret_cast<float2*>(O + off0 + col) = make_float2(o[dt][0] * inv0, o[dt][1] * inv0);
                *reinterpret_cast<float2*>(O + off1 + col) = make_float2(o[dt][2] * inv1, o[dt][3] * inv1);
            }
        }
    } else {
        // split-KV: store unnormalized partial + (m, l) for a later combine
        const size_t row = (size_t)local_row0 + g;
        float* O0 = part.O_part + ((size_t)split * part.rows + row) * ld_o + head * D;
        float* O1 = O0 + 8 * ld_o;
        #pragma unroll
        for (int dt = 0; dt < DT; dt++) {
            const int col = dt * 8 + 2 * c;
            *reinterpret_cast<float2*>(O0 + col) = make_float2(o[dt][0], o[dt][1]);
            *reinterpret_cast<float2*>(O1 + col) = make_float2(o[dt][2], o[dt][3]);
        }
        if (c == 0) {
            const size_t s0 = ((size_t)split * part.rows + row) * H_local + head;
            const size_t s1 = s0 + 8 * (size_t)H_local;
            part.m_part[s0] = m[0]; part.l_part[s0] = l[0];
            part.m_part[s1] = m[1]; part.l_part[s1] = l[1];
        }
    }

    // ---- signal tile completion ----
    if (tile_counters != nullptr) {
        __threadfence();          // each thread's O stores are device-visible before the count
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(tile_counters + block_row0 / tile_rows, 1);
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// One thread per (row, column pair): O = sum_i 2^(m_i - M) O_i / sum_i 2^(m_i - M) l_i
static __global__ void combine_kernel(
    const float* __restrict__ O_part, const float* __restrict__ m_part,
    const float* __restrict__ l_part, void* __restrict__ O_void, int out_bf16,
    int row0, int rows, int ws_rows, int d_local, int H_local, int d_h, int split)
{
    const int pairs = d_local / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * pairs) return;
    const int r = idx / pairs;
    const int col = (idx % pairs) * 2;
    const int head = col / d_h;
    const size_t row = (size_t)row0 + r;

    float M = -INFINITY;
    for (int i = 0; i < split; i++)
        M = fmaxf(M, m_part[((size_t)i * ws_rows + row) * H_local + head]);

    float L = 0.f;
    float2 acc = make_float2(0.f, 0.f);
    for (int i = 0; i < split; i++) {
        const size_t s = ((size_t)i * ws_rows + row) * H_local + head;
        const float w = exp2f(m_part[s] - M);
        L += w * l_part[s];
        const float2 v = *reinterpret_cast<const float2*>(
            O_part + ((size_t)i * ws_rows + row) * d_local + col);
        acc.x += w * v.x;
        acc.y += w * v.y;
    }
    const float inv = 1.f / L;
    const size_t off = (size_t)r * d_local + col;
    if (out_bf16)
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(O_void) + off) =
            __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
    else
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(O_void) + off) = make_float2(acc.x * inv, acc.y * inv);
}

void flash_attn_tc_combine(
    const FlashAttnTCPartials& part, void* O, int row0, int rows,
    const FlashAttnTCParams& p, cudaStream_t stream)
{
    const int d_local = p.H_local * p.d_h;
    const int total = rows * (d_local / 2);
    const int blocks = (total + 255) / 256;
    combine_kernel<<<blocks, 256, 0, stream>>>(
        part.O_part, part.m_part, part.l_part, O, p.out_bf16,
        row0, rows, part.rows, d_local, p.H_local, p.d_h, p.split);
}

template <int D>
static void launch_tc(
    const half* Q, const half* K, const half* V,
    void* O, const FlashAttnTCPartials& part, int q_start, int q_rows,
    const FlashAttnTCParams& p, cudaStream_t stream,
    int* tile_counters, int tile_rows)
{
    constexpr size_t smem_needed = 4ull * FLASH_TC_BN * (D + 8) * sizeof(half);
    size_t smem = smem_needed;
    if (p.max_blocks_per_sm > 0) {
        // Ask for enough shared memory that only max_blocks_per_sm blocks fit per SM.
        // (1 KB per block is reserved by the hardware on Ampere.)
        static int sm_smem = 0;
        if (sm_smem == 0) {
            int dev; cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sm_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
        }
        size_t cap = (size_t)sm_smem / (p.max_blocks_per_sm + 1) + 1024;
        if (cap > smem) smem = cap;
    }
    static size_t configured = 0;
    if (configured < smem) {
        cudaFuncSetAttribute(flash_attn_tc_kernel<D>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = smem;
    }
    dim3 grid(p.H_local * p.split, q_rows / FLASH_TC_BM);
    flash_attn_tc_kernel<D><<<grid, 128, smem, stream>>>(
        Q, K, V, O, part, q_start, p.S, p.H_local, p.split, p.S / p.split, p.causal, p.out_bf16,
        p.scale * 1.4426950408889634f, tile_counters, tile_rows);
}

void flash_attn_tc_forward(
    const half* Q, const half* K, const half* V,
    void* O, const FlashAttnTCPartials& part, int q_start, int q_rows,
    const FlashAttnTCParams& p, cudaStream_t stream,
    int* tile_counters, int tile_rows)
{
    switch (p.d_h) {
        case 16:  launch_tc<16> (Q, K, V, O, part, q_start, q_rows, p, stream, tile_counters, tile_rows); break;
        case 32:  launch_tc<32> (Q, K, V, O, part, q_start, q_rows, p, stream, tile_counters, tile_rows); break;
        case 64:  launch_tc<64> (Q, K, V, O, part, q_start, q_rows, p, stream, tile_counters, tile_rows); break;
        case 128: launch_tc<128>(Q, K, V, O, part, q_start, q_rows, p, stream, tile_counters, tile_rows); break;
        default:
            fprintf(stderr, "flash_attn_tc: unsupported d_h=%d\n", p.d_h);
            std::abort();
    }
}

const char* flash_attn_tc_check_params(const FlashAttnTCParams& p, int q_rows)
{
    if (p.d_h != 16 && p.d_h != 32 && p.d_h != 64 && p.d_h != 128)
        return "tc: d_h must be 16, 32, 64 or 128";
    if (p.S % FLASH_TC_BN != 0) return "tc: S must be a multiple of 64";
    if (q_rows % FLASH_TC_BM != 0) return "tc: tile must be a multiple of 64";
    if (p.split < 1 || p.S % (p.split * FLASH_TC_BN) != 0) return "tc: S must be a multiple of split * 64";
    return nullptr;
}

static __global__ void f32_to_f16_kernel(const float* __restrict__ in, half* __restrict__ out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = __float2half(in[i]);
}

void convert_f32_to_f16(const float* in, half* out, size_t n, cudaStream_t stream)
{
    int blocks = (int)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(in, out, n);
}
