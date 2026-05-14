#pragma once

#include "attn_defs.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Weights {
    float* X;
    float* W_q, *W_k, *W_v, *W_O;
    float* Q, *K, *V;
};

void init_mpi(int* argc, char*** argv, int& rank, int& world_size);
int init_cuda(int rank);
ncclComm_t init_nccl(int rank, int world_size);

AttnParams make_params(int S, int d, int H, int B_M, int B_N, int world_size);

// X seeded with `seed`, weights seeded with `seed * 31 + rank + 1`.
// X is identical across all ranks; weights are rank-specific.
Weights allocate_and_fill(const AttnParams& p, int rank, int seed, cudaStream_t stream);

// Runs on whatever stream cublas_handle is currently set to.
void compute_qkv(const Weights& w, const AttnParams& p, cublasHandle_t handle);

void free_weights(const Weights& w);
