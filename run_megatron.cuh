#pragma once

#include "attn_defs.cuh"

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
