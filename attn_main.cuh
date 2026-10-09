#pragma once

#include "attn_defs.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Weights {
    float* X;
    float* W_q, *W_k, *W_v, *W_O;
    float* Q, *K, *V;      // fp32 [H_local][S][d_h]
    half*  Qh, *Kh, *Vh;   // fp16 copies for the tensor-core kernel
    __nv_bfloat16* W_Ob;   // bf16 copy of W_O for the bf16 projection
};

void init_mpi(int* argc, char*** argv, int& rank, int& world_size);
int init_cuda(int rank);
ncclComm_t init_nccl(int rank, int world_size);   // one communicator over MPI_COMM_WORLD

AttnParams make_params(int S, int d, int H, int tile, int B_M, int B_N,
                       int kernel, int nbuf, int split, int causal, int proj, int world_size);

const char* proj_name(int proj);
int parse_proj(const char* s);   // "fp32" | "tf32" | "bf16", -1 if unknown
size_t o_elem_bytes(const AttnParams& p);   // bytes per attention-output element (4, or 2 for bf16)

// nullptr if the configuration is runnable, otherwise a description of the problem.
const char* attn_check_params(const AttnParams& p, int cuda_device);

const char* kernel_name(int kernel);
int parse_kernel(const char* s);   // "v1" | "tc", -1 if unknown

// X seeded with `seed`, weights seeded with `seed * 31 + rank + 1`.
// X is identical across all ranks; weights are rank-specific.
Weights allocate_and_fill(const AttnParams& p, int rank, int seed, cudaStream_t stream);

// Runs on whatever stream cublas_handle is currently set to; also fills the fp16 copies.
void compute_qkv(const Weights& w, const AttnParams& p, cublasHandle_t handle);

void free_weights(const Weights& w);

// ncomm NCCL communicators + comm streams (1..MAX_COMM).
// prioritize: give the projection and communication streams the highest
// stream priority so their blocks are scheduled ahead of pending attention blocks.
// gctx_sms: reserve this many SMs in a green context for the communication
// streams (0 = no partitioning); gctx_proj also moves the projection stream there.
RunCtx make_ctx(MPI_Comm comm, int rank, int world_size, int ncomm = 1, bool prioritize = true,
                int gctx_sms = 0, bool gctx_proj = false);
void destroy_ctx(RunCtx& c);

// Attention over query rows [q_start, q_start + q_rows) for every local head.
// O: [q_rows][d_local] sequence-major.
AttnWorkspace alloc_workspace(const AttnParams& p, int rows);
void free_workspace(AttnWorkspace& ws);

// Attention kernel only. With split == 1 writes O directly; otherwise writes
// partials into ws (rows indexed from 0 within this launch) and O is untouched.
// O holds float, or bf16 when p.proj == PROJ_BF16 (see o_elem_bytes).
// tile_counters (optional): per-tile block-completion counters, see flash_attn_tc.cuh.
void launch_attention_raw(const Weights& w, void* O, const AttnWorkspace& ws,
                          int q_start, int q_rows, const AttnParams& p, cudaStream_t stream,
                          int* tile_counters = nullptr, int tile_rows = 0);

// Combine split partials for workspace rows [row0, row0 + rows) into O (no-op when split == 1).
void launch_combine(const AttnWorkspace& ws, void* O, int row0, int rows,
                    const AttnParams& p, cudaStream_t stream);

// launch_attention_raw followed by launch_combine on the same stream.
void launch_attention(const Weights& w, void* O, const AttnWorkspace& ws,
                      int q_start, int q_rows, const AttnParams& p, cudaStream_t stream,
                      int* tile_counters = nullptr, int tile_rows = 0);

// out[rows x d] = O_tile[rows x d_local] * W_O, in the precision selected by p.proj.
void project(const RunCtx& c, const Weights& w, const void* O_tile, float* out,
             int rows, const AttnParams& p);

// Blocks per pipeline tile for the configured kernel (the counter target).
int blocks_per_tile(const AttnParams& p);
