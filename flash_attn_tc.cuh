#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Tensor-core FlashAttention forward (fp16 inputs, fp32 accumulate, fp32 output).
//
// Block = 4 warps handles 64 query rows of one head (16 rows per warp).
// K/V are streamed through shared memory in 64-row chunks, double-buffered
// with cp.async. S = Q K^T and O += P V run on mma.sync.m16n8k16; the online
// softmax statistics live in registers in the mma accumulator layout.
//
// Grid: (H_local * split, q_rows / 64)   Block: 128 threads
// Heads are the fast grid index so blocks are dispatched in query-row order;
// that is what lets a per-tile completion counter fire early.
struct FlashAttnTCParams {
    int   S;
    int   d_h;      // 16, 32, 64 or 128
    int   H_local;
    int   split;    // split-KV factor: each block sweeps S / split keys (1 = no split)
    int   causal;   // 1 = query row i attends keys j <= i only
    float scale;
};

// Split-KV partial results (only used when split > 1), indexed by row within a launch.
//   O_part: [split][rows][H_local * d_h]   unnormalized output
//   m_part: [split][rows][H_local]         row max (log2 domain)
//   l_part: [split][rows][H_local]         row denominator
struct FlashAttnTCPartials {
    float* O_part;
    float* m_part;
    float* l_part;
    int    rows;    // rows the workspace was sized for (= launch rows)
};

constexpr int FLASH_TC_BM = 64;   // query rows per block
constexpr int FLASH_TC_BN = 64;   // K/V rows per shared-memory chunk

// Q, K, V: half [H_local][S][d_h] head-major.
// O: [q_rows][H_local * d_h] sequence-major, rows q_start .. q_start + q_rows. Written directly when p.split == 1;
//    otherwise partials go to `part` and O is untouched until flash_attn_tc_combine.
// If tile_counters != nullptr, every block atomically increments
// tile_counters[(row within launch) / tile_rows] after its output rows are
// globally visible, so a consumer can wait for a tile with cuStreamWaitValue32.
void flash_attn_tc_forward(
    const half* Q, const half* K, const half* V,
    float* O, const FlashAttnTCPartials& part, int q_start, int q_rows,
    const FlashAttnTCParams& p, cudaStream_t stream,
    int* tile_counters = nullptr, int tile_rows = 0
);

// Merge the split partials for workspace rows [row0, row0 + rows) into
// O[0 .. rows) (O points at the first of those rows).
void flash_attn_tc_combine(
    const FlashAttnTCPartials& part, float* O, int row0, int rows,
    const FlashAttnTCParams& p, cudaStream_t stream
);

// Returns nullptr if ok, otherwise a static string describing the problem.
const char* flash_attn_tc_check_params(const FlashAttnTCParams& p, int q_rows);

void convert_f32_to_f16(const float* in, half* out, size_t n, cudaStream_t stream);
