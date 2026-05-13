#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "flash_attn.cuh"

struct AttnParams {
    int S, d, H, R;
    int H_local, d_h, d_local, num_tiles;
    int B_M, B_N;
    int num_buffers, max_inflight;
    FlashAttnParams flash;
};

struct RunResult {
    double wall_ms;
};
