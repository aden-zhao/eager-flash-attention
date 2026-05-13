#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Naive single-GPU attention for correctness checking.
// Q, K, V : [H_local, S, d_h]  head-major
// O       : [S, H_local * d_h] sequence-major (matches flash_attn output layout)
// workspace: S * S floats, caller-allocated via baseline_attention_workspace_bytes()
void baseline_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* workspace,
    int H_local,
    int S,
    int d_h,
    cudaStream_t stream,
    cublasHandle_t handle
);

size_t baseline_attention_workspace_bytes(int S);
