#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include "flash_attn.cuh"

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

struct AttnParams {
    int S, d, H, R;
    int H_local, d_h, d_local, num_tiles;
    int B_M, B_N;
    FlashAttnParams flash;
};

struct RunResult {
    double wall_ms;
};
