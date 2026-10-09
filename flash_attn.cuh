#pragma once

#include <cuda_runtime.h>

// v1: fp32 scalar kernel. One warp per query row, lanes split d_h.
// Grid: (H_local, q_rows / B_M)   Block: (32 * B_M)
struct FlashAttnParams {
    int   S;
    int   d_h;
    int   H_local;
    int   B_M;
    int   B_N;
    int   causal;
    float scale;
};

size_t flash_attn_smem_bytes(const FlashAttnParams& p);

__global__ void flash_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int          q_start,
    FlashAttnParams p,
    int*         __restrict__ tile_counters,
    int          tile_rows
);

// Q, K, V: [H_local][S][d_h] head-major.
// O: [q_rows][H_local * d_h] sequence-major, rows q_start .. q_start + q_rows.
// tile_counters: see flash_attn_tc_forward.
void flash_attn_forward(
    const float* Q, const float* K, const float* V,
    float* O, int q_start, int q_rows,
    const FlashAttnParams& p, cudaStream_t stream,
    int* tile_counters = nullptr, int tile_rows = 0
);

// Returns nullptr if ok, otherwise a static string describing the problem.
const char* flash_attn_check_params(const FlashAttnParams& p, int q_rows, int cuda_device);
