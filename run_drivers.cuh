#pragma once

#include "attn_defs.cuh"

// All input/output buffers are pre-allocated by the caller.
// Each driver allocates and frees its own temporary buffers internally.
// No temporary buffers are shared or reused across drivers.

// Megatron-style: all tiles computed first, then one blocking allreduce.
RunResult run_megatron(
    const float* Q,
    const float* K,
    const float* V,
    const float* W_O,
    float* output,
    const AttnParams& p,
    cudaStream_t compute_stream,
    cudaStream_t proj_stream,
    cublasHandle_t cublas_handle,
    MPI_Comm comm
);

// Synchronous tiled: flash_attn per tile, then blocking allreduce per tile.
RunResult run_sync(
    const float* Q,
    const float* K,
    const float* V,
    const float* W_O,
    float* output,
    const AttnParams& p,
    cudaStream_t compute_stream,
    cudaStream_t proj_stream,
    cublasHandle_t cublas_handle,
    MPI_Comm comm
);

// Overlap: flash_attn per tile, then non-blocking Iallreduce overlapping
// with the next tile's computation.
RunResult run_overlap(
    const float* Q,
    const float* K,
    const float* V,
    const float* W_O,
    float* output,
    const AttnParams& p,
    cudaStream_t compute_stream,
    cudaStream_t proj_stream,
    cublasHandle_t cublas_handle,
    MPI_Comm comm
);
