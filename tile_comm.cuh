#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <climits>
#include <cstdio>

#ifdef USE_NVTX
#include <nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st = (call);                                        \
        if (st != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n",            \
                    __FILE__, __LINE__, (int)st);                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

#define MPI_CHECK(call)                                                    \
    do {                                                                   \
        int err = (call);                                                  \
        if (err != MPI_SUCCESS) {                                          \
            fprintf(stderr, "MPI error %s:%d\n", __FILE__, __LINE__);     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

enum class Mode { SERIALIZED_FULL, SERIALIZED, TILED, OVERLAP };
enum class ComputeKind { SYNTHETIC, BASELINE };
enum class CudaAwareStatus { YES, NO, UNKNOWN };

struct TileDesc {
    int tile_id;
    int buf_id;
    float* send_ptr;
    float* recv_ptr;
    int send_elems;
    int group_tiles;
    cudaEvent_t ready_event;
    double producer_wall;
};

struct SharedState {
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<TileDesc> ready_tiles;
    bool done = false;
    std::vector<bool> buffer_free;
};

struct TileTiming {
    double queue_wait_ms;
    double event_sync_ms;
    double comm_ms;
    double d2d_copy_ms;
    double total_ms;
    double comm_start_offset_ms;
    double comm_end_offset_ms;
    double copy_start_offset_ms;
    double copy_end_offset_ms;
};

struct TimingStats {
    std::vector<TileTiming> per_tile;
};

struct AllreduceGroup {
    int first_tile;
    int group_tiles;
    int send_elems;
    float* send_ptr;
    float* recv_ptr;
};

struct InFlight {
    TileDesc desc;
    MPI_Request req;
    double comm_start;
};

Mode parse_mode(int argc, char** argv);
const char* mode_str(Mode m);
ComputeKind parse_compute(int argc, char** argv);
const char* compute_str(ComputeKind c);
CudaAwareStatus detect_cuda_aware_mpi();
const char* cuda_aware_str(CudaAwareStatus s);
int parse_int(int argc, char** argv, const char* name, int def);

AllreduceGroup make_allreduce_group(
    int first_tile, int group_tiles,
    float* send_ptr, float* recv_ptr,
    size_t output_tile_elems
);

void allreduce_group_blocking(const AllreduceGroup& g, MPI_Comm comm);
void iallreduce_group(const AllreduceGroup& g, MPI_Comm comm, MPI_Request* req);

void project_tile(
    cublasHandle_t handle,
    float* pre_proj, float* W_out, float* send_buf,
    int proj_m, int proj_k, int proj_n
);

void comm_thread_fn(
    SharedState* state,
    TimingStats* timing,
    MPI_Comm comm,
    float* output_device,
    int output_tile_elems,
    cudaStream_t comm_stream,
    int max_inflight,
    double t0_wall
);
