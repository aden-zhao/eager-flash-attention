#pragma once

#include <cuda_runtime.h>

struct FlashAttnParams {
    int   S;
    int   d_h;
    int   H_local;
    int   B_M;
    int   B_N;
    float scale;
};

size_t flash_attn_smem_bytes(const FlashAttnParams& p);

// Grid: (H_local)  Block: (d_h, B_M)
__global__ void flash_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O_tile,
    int          tile_idx,
    FlashAttnParams p
);

void flash_attn_forward_tile(
    const float*           Q,
    const float*           K,
    const float*           V,
    float*                 O_tile,
    int                    tile_idx,
    const FlashAttnParams& p,
    cudaStream_t           stream
);

bool flash_attn_check_params(const FlashAttnParams& p, int cuda_device);
