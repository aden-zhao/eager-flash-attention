#pragma once

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include "flash_attn.cuh"
#include "flash_attn_tc.cuh"

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t e = (call);                                         \
        if (e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(e));         \
            MPI_Abort(MPI_COMM_WORLD, 1);                               \
        }                                                               \
    } while (0)

#define CUBLAS_CHECK(call)                                              \
    do {                                                                \
        cublasStatus_t s = (call);                                      \
        if (s != CUBLAS_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                \
                    __FILE__, __LINE__, (int)s);                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                               \
        }                                                               \
    } while (0)

#define MPI_CHECK(call)                                                 \
    do {                                                                \
        int e = (call);                                                 \
        if (e != MPI_SUCCESS) {                                         \
            fprintf(stderr, "MPI error %s:%d\n", __FILE__, __LINE__);  \
            MPI_Abort(MPI_COMM_WORLD, 1);                               \
        }                                                               \
    } while (0)

#define NCCL_CHECK(call)                                                \
    do {                                                                \
        ncclResult_t r = (call);                                        \
        if (r != ncclSuccess) {                                         \
            fprintf(stderr, "NCCL error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, ncclGetErrorString(r));         \
            MPI_Abort(MPI_COMM_WORLD, 1);                               \
        }                                                               \
    } while (0)

// Which attention kernel to use.
//   KERNEL_V1: fp32 scalar kernel, one warp per query row (original prototype)
//   KERNEL_TC: fp16 tensor-core kernel (mma.sync), fp32 accumulate
enum AttnKernel { KERNEL_V1 = 0, KERNEL_TC = 1 };

// Output-projection GEMM precision. bf16 stores the attention output in bf16
// (written directly by the tensor-core kernel / combine) and uses a bf16
// tensor-core GEMM with fp32 accumulation, the shape production layers use.
enum ProjPrec { PROJ_FP32 = 0, PROJ_TF32 = 1, PROJ_BF16 = 2 };

struct AttnParams {
    int S, d, H, R;
    int H_local, d_h, d_local;
    int tile;        // query rows per pipeline tile (projection + allreduce granularity)
    int num_tiles;   // S / tile
    int B_M, B_N;    // v1 kernel: query rows per block, K/V rows per smem chunk
    int kernel;      // AttnKernel
    int nbuf;        // pipeline depth (buffers in flight) for overlap mode
    int split;       // split-KV factor for the tc kernel (1 = off)
    int causal;      // causal (lower-triangular) attention
    int proj;        // ProjPrec
    int skip_proj;   // diagnostics: omit the projection GEMM
    int skip_comm;   // diagnostics: omit the AllReduce
    FlashAttnParams   flash;
    FlashAttnTCParams flash_tc;
};

// Everything a run driver needs besides the data.
// comm_streams[i] pairs with nccls[i]; tile t uses slot t % ncomm so that up
// to ncomm AllReduces can be in flight concurrently (one NCCL communicator
// can only run one collective at a time).
// Optional SM partitioning with CUDA green contexts (gctx_sms > 0): the
// communication streams (and optionally the projection stream) live in a green
// context holding `gctx_sms` SMs and the compute stream in one holding the rest,
// so NCCL / GEMM blocks never wait for a fully occupied SM to drain.
#define MAX_COMM 8
struct RunCtx {
    cudaStream_t   compute_stream;
    cudaStream_t   proj_stream;
    cudaStream_t   comm_streams[MAX_COMM];
    ncclComm_t     nccls[MAX_COMM];
    int            ncomm;
    cublasHandle_t cublas;
    MPI_Comm       comm;
    CUgreenCtx     gctx_compute;   // null unless partitioned
    CUgreenCtx     gctx_comm;
    int            gctx_sms_comm;  // SMs actually granted to the comm partition
    int            gctx_sms_compute;
};

#define CU_CHECK(call)                                                  \
    do {                                                                \
        CUresult r_ = (call);                                           \
        if (r_ != CUDA_SUCCESS) {                                       \
            const char* msg = nullptr;                                  \
            cuGetErrorString(r_, &msg);                                 \
            fprintf(stderr, "CUDA driver error %s:%d: %s\n",           \
                    __FILE__, __LINE__, msg ? msg : "?");               \
            MPI_Abort(MPI_COMM_WORLD, 1);                               \
        }                                                               \
    } while (0)

struct RunResult {
    double wall_ms;   // max over ranks
};

// Split-KV scratch for one attention launch of `rows` query rows (empty when split == 1).
struct AttnWorkspace {
    FlashAttnTCPartials part;
};
