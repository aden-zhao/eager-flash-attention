/*
 * Four benchmark modes (--mode):
 *   serialized_full : compute all -> project all -> one full allreduce
 *   serialized      : compute all -> project all -> allreduce per tile
 *   tiled           : compute+project per tile -> sync -> allreduce per tile
 *   overlap         : compute+project+allreduce pipelined per tile
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_NVTX
#include <nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP()      nvtxRangePop()
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif

/* Error-checking macros */

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st = (call);                                        \
        if (st != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n",             \
                    __FILE__, __LINE__, (int)st);                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)

#define MPI_CHECK(call)                                                    \
    do {                                                                   \
        int err = (call);                                                  \
        if (err != MPI_SUCCESS) {                                          \
            fprintf(stderr, "MPI error %s:%d\n", __FILE__, __LINE__);      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                  \
    } while (0)


/* Synthetic compute kernel with tile/rank-dependent values */
/* TODO: replace with actual kernel */

__global__ void fill_tile_kernel(float* ptr, int n, float base_value,
                                 int fma_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = base_value;
        for (int i = 0; i < fma_iters; ++i) {
            v = __fmaf_rn(v, 1.0f, 0.0001f);
        }
        ptr[idx] = v;
    }
}

/*  Expected value computation (host-side, matches kernel exactly) */
/* TODO: replace with serialized_baseline */

static float compute_expected_kernel_output(float base_value, int fma_iters) {
    float v = base_value;
    for (int i = 0; i < fma_iters; ++i) {
        v = std::fma(v, 1.0f, 0.0001f);
    }
    return v;
}

enum class Mode { SERIALIZED_FULL, SERIALIZED, TILED, OVERLAP };

static Mode parse_mode(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0) {
            if (std::strcmp(argv[i + 1], "serialized_full") == 0) return Mode::SERIALIZED_FULL;
            if (std::strcmp(argv[i + 1], "serialized") == 0)      return Mode::SERIALIZED;
            if (std::strcmp(argv[i + 1], "tiled") == 0)           return Mode::TILED;
            if (std::strcmp(argv[i + 1], "overlap") == 0)         return Mode::OVERLAP;
            fprintf(stderr, "Unknown mode: %s (options: serialized_full, serialized, tiled, overlap)\n",
                    argv[i + 1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    return Mode::OVERLAP;
}

static const char* mode_str(Mode m) {
    switch (m) {
        case Mode::SERIALIZED_FULL: return "serialized_full";
        case Mode::SERIALIZED:      return "serialized";
        case Mode::TILED:           return "tiled";
        case Mode::OVERLAP:         return "overlap";
    }
    return "unknown";
}

/*  CUDA-aware MPI detection (3-state)                                 */

enum class CudaAwareStatus { YES, NO, UNKNOWN };

static CudaAwareStatus detect_cuda_aware_mpi() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() ? CudaAwareStatus::YES : CudaAwareStatus::NO;
#else
    return CudaAwareStatus::UNKNOWN;
#endif
}

static const char* cuda_aware_str(CudaAwareStatus s) {
    switch (s) {
        case CudaAwareStatus::YES:     return "detected yes";
        case CudaAwareStatus::NO:      return "detected no";
        case CudaAwareStatus::UNKNOWN: return "unknown (assuming yes)";
    }
    return "?";
}

/* Shared types for overlap mode */

struct TileDesc {
    int   tile_id;
    int   buf_id;
    float* send_ptr;
    float* recv_ptr;
    int   send_elems;       // may span multiple tiles when aggregating
    cudaEvent_t ready_event;
    double producer_wall;
};

struct SharedState {
    std::mutex              mtx;
    std::condition_variable cv;
    std::queue<TileDesc>    ready_tiles;
    bool                    done = false;
    std::vector<bool>       buffer_free;
};

struct TileTiming {
    double queue_wait_ms;
    double event_sync_ms;
    double comm_ms;
    double d2d_copy_ms;
    double total_ms;
    // Absolute offsets from run start (for Gantt charts)
    double comm_start_offset_ms;
    double comm_end_offset_ms;
    double copy_start_offset_ms;
    double copy_end_offset_ms;
};

struct TimingStats {
    std::vector<TileTiming> per_tile;
};

/* Communication thread (overlap mode) — multi-in-flight MPI */

struct InFlight {
    TileDesc    desc;
    MPI_Request req;
    double      comm_start;
};

void comm_thread_fn(
    SharedState*  state,
    TimingStats*  timing,
    MPI_Comm      comm,
    float*        output_device,
    int           output_tile_elems,  // elems per output tile (for offset calc)
    cudaStream_t  comm_stream,
    int           max_inflight,
    double        t0_wall
) {
    std::vector<InFlight> active;
    active.reserve(max_inflight);
    bool producer_done = false;

    auto try_launch = [&]() {
        while ((int)active.size() < max_inflight) {
            TileDesc desc;
            {
                std::unique_lock<std::mutex> lock(state->mtx);
                if (state->ready_tiles.empty()) {
                    producer_done = state->done;
                    break;
                }
                desc = state->ready_tiles.front();
                state->ready_tiles.pop();
                producer_done = state->done && state->ready_tiles.empty();
            }

            double t_dequeue = MPI_Wtime();
            timing->per_tile[desc.tile_id].queue_wait_ms =
                (t_dequeue - desc.producer_wall) * 1e3;

            double t_ev0 = MPI_Wtime();
            CUDA_CHECK(cudaEventSynchronize(desc.ready_event));
            double t_ev1 = MPI_Wtime();
            timing->per_tile[desc.tile_id].event_sync_ms = (t_ev1 - t_ev0) * 1e3;

            InFlight inf;
            inf.desc = desc;
            inf.comm_start = MPI_Wtime();

            timing->per_tile[desc.tile_id].comm_start_offset_ms =
                (inf.comm_start - t0_wall) * 1e3;

            NVTX_PUSH(("MPI tile " + std::to_string(desc.tile_id)).c_str());

            MPI_CHECK(MPI_Iallreduce(
                desc.send_ptr, desc.recv_ptr,
                desc.send_elems, MPI_FLOAT, MPI_SUM, comm,
                &inf.req
            ));

            NVTX_POP();
            active.push_back(inf);
        }
    };

    auto complete_one = [&](int idx) {
        InFlight& inf = active[idx];
        double t_comm_end = MPI_Wtime();

        TileTiming& tt = timing->per_tile[inf.desc.tile_id];
        tt.comm_ms = (t_comm_end - inf.comm_start) * 1e3;
        tt.comm_end_offset_ms = (t_comm_end - t0_wall) * 1e3;

        float* out_ptr = output_device +
                         static_cast<size_t>(inf.desc.tile_id) * output_tile_elems;

        double t_d2d0 = MPI_Wtime();
        tt.copy_start_offset_ms = (t_d2d0 - t0_wall) * 1e3;

        CUDA_CHECK(cudaMemcpyAsync(
            out_ptr, inf.desc.recv_ptr,
            static_cast<size_t>(inf.desc.send_elems) * sizeof(float),
            cudaMemcpyDeviceToDevice, comm_stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(comm_stream));

        double t_d2d1 = MPI_Wtime();
        tt.d2d_copy_ms = (t_d2d1 - t_d2d0) * 1e3;
        tt.copy_end_offset_ms = (t_d2d1 - t0_wall) * 1e3;
        tt.total_ms = (t_d2d1 - inf.desc.producer_wall) * 1e3;

        {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->buffer_free[inf.desc.buf_id] = true;
        }
        state->cv.notify_all();

        active[idx] = active.back();
        active.pop_back();
    };

    while (true) {
        try_launch();

        if (active.empty()) {
            if (producer_done) break;
            {
                std::unique_lock<std::mutex> lock(state->mtx);
                state->cv.wait(lock, [&] {
                    return state->done || !state->ready_tiles.empty();
                });
                producer_done = state->done && state->ready_tiles.empty();
            }
            continue;
        }

        if (active.size() == 1) {
            MPI_CHECK(MPI_Wait(&active[0].req, MPI_STATUS_IGNORE));
            complete_one(0);
        } else {
            std::vector<MPI_Request> reqs(active.size());
            for (size_t i = 0; i < active.size(); ++i)
                reqs[i] = active[i].req;

            int outcount = 0;
            std::vector<int> indices(active.size());
            MPI_CHECK(MPI_Waitsome(
                (int)reqs.size(), reqs.data(),
                &outcount, indices.data(), MPI_STATUSES_IGNORE
            ));

            for (size_t i = 0; i < active.size(); ++i)
                active[i].req = reqs[i];

            if (outcount > 0) {
                std::sort(indices.begin(), indices.begin() + outcount,
                          std::greater<int>());
                for (int i = 0; i < outcount; ++i)
                    complete_one(indices[i]);
            }
        }
    }
}

/* CLI helpers */

static int parse_int(int argc, char** argv, const char* name, int def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
    return def;
}

/* Projection helper */

static void project_tile(
    cublasHandle_t handle,
    float* pre_proj, float* W_out, float* send_buf,
    int proj_m, int proj_k, int proj_n
) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        proj_n, proj_m, proj_k,
        &alpha,
        W_out,    proj_n,
        pre_proj, proj_k,
        &beta,
        send_buf, proj_n
    ));
}

/* main */

int main(int argc, char** argv) {

    /* MPI init */
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv,
                              MPI_THREAD_MULTIPLE, &provided));
    if (provided < MPI_THREAD_MULTIPLE) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cerr << "ERROR: MPI does not provide MPI_THREAD_MULTIPLE.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0, world_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    /* GPU setup */
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) std::cerr << "No CUDA devices found.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));

    /* Parse CLI */
    Mode mode = parse_mode(argc, argv);

    const int num_tiles       = parse_int(argc, argv, "--num_tiles",       8);
    const int tile_elems      = parse_int(argc, argv, "--tile_elems",      1024 * 1024);
    const int num_buffers     = parse_int(argc, argv, "--num_buffers",     2);
    const int fma_iters       = parse_int(argc, argv, "--fma_iters",       200);
    const int max_inflight    = parse_int(argc, argv, "--max_inflight",    1);
    const int aggregate_tiles = parse_int(argc, argv, "--aggregate_tiles", 1);

    const int proj_m = parse_int(argc, argv, "--proj_m", 1024);
    const int proj_k = parse_int(argc, argv, "--proj_k", tile_elems / 1024);
    const int proj_n = parse_int(argc, argv, "--proj_n", proj_k);

    const size_t output_tile_elems = static_cast<size_t>(proj_m) * proj_n;

    /* Dimension validation */
    if (static_cast<size_t>(proj_m) * proj_k != static_cast<size_t>(tile_elems)) {
        if (rank == 0)
            std::cerr << "ERROR: proj_m * proj_k (" << proj_m << " * " << proj_k
                      << " = " << static_cast<size_t>(proj_m) * proj_k
                      << ") must equal tile_elems (" << tile_elems << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (output_tile_elems > static_cast<size_t>(INT_MAX)) {
        if (rank == 0)
            std::cerr << "ERROR: output_tile_elems exceeds INT_MAX; MPI count would overflow.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (mode == Mode::OVERLAP && num_tiles % aggregate_tiles != 0) {
        if (rank == 0)
            std::cerr << "ERROR: num_tiles (" << num_tiles
                      << ") must be divisible by aggregate_tiles ("
                      << aggregate_tiles << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (mode == Mode::OVERLAP &&
        static_cast<size_t>(aggregate_tiles) * output_tile_elems > static_cast<size_t>(INT_MAX)) {
        if (rank == 0)
            std::cerr << "ERROR: aggregated message size exceeds INT_MAX.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (mode == Mode::OVERLAP && max_inflight > num_buffers) {
        if (rank == 0)
            std::cerr << "ERROR: max_inflight (" << max_inflight
                      << ") must be <= num_buffers (" << num_buffers
                      << ") — cannot have more collectives in flight than "
                         "reusable send/recv buffer slots.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    CudaAwareStatus cuda_aware = detect_cuda_aware_mpi();

    if (rank == 0) {
        std::cout << "Tile communication benchmark (v4.1)\n"
                  << "  mode             = " << mode_str(mode) << "\n"
                  << "  world_size       = " << world_size     << "\n"
                  << "  num_tiles        = " << num_tiles      << "\n"
                  << "  tile_elems       = " << tile_elems     << "\n"
                  << "  num_buffers      = " << num_buffers    << "\n"
                  << "  max_inflight     = " << max_inflight   << "\n"
                  << "  aggregate_tiles  = " << aggregate_tiles << "\n"
                  << "  fma_iters        = " << fma_iters      << "\n"
                  << "  proj (MxKxN)     = " << proj_m << "x" << proj_k
                  << "x" << proj_n << "\n"
                  << "  output tile      = " << output_tile_elems << " elems\n"
                  << "  cuda-aware MPI   = " << cuda_aware_str(cuda_aware)
                  << "\n";
    }

    /* Allocate per-buffer resources */
    // Non-overlap modes: one buffer per tile.
    // Overlap mode: num_buffers with reuse.
    const int n_bufs = (mode == Mode::OVERLAP) ? num_buffers : num_tiles;

    // For aggregation, send/recv buffers must hold aggregate_tiles worth of data.
    const size_t agg_elems = static_cast<size_t>(aggregate_tiles) * output_tile_elems;

    std::vector<float*>      send_buffers(n_bufs, nullptr);
    std::vector<float*>      recv_buffers(n_bufs, nullptr);
    std::vector<cudaEvent_t> ready_events(n_bufs);

    // Pre-projection buffers and compute_done events need one slot per tile
    // within an aggregation group to avoid stream races: tile N+1's compute
    // must not overwrite pre_proj while tile N's projection is still reading it.
    // In overlap mode: num_buffers * aggregate_tiles slots.
    // In other modes: num_tiles slots (one per tile, trivially safe).
    const int n_preproj = (mode == Mode::OVERLAP)
        ? num_buffers * aggregate_tiles : num_tiles;

    std::vector<float*>      pre_proj_buffers(n_preproj, nullptr);
    std::vector<cudaEvent_t> compute_done_events(n_preproj);

    for (int b = 0; b < n_bufs; ++b) {
        size_t send_recv_bytes = (mode == Mode::OVERLAP)
            ? agg_elems * sizeof(float)
            : output_tile_elems * sizeof(float);
        CUDA_CHECK(cudaMalloc(&send_buffers[b], send_recv_bytes));
        CUDA_CHECK(cudaMalloc(&recv_buffers[b], send_recv_bytes));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready_events[b],
                                            cudaEventDisableTiming));
    }
    for (int p = 0; p < n_preproj; ++p) {
        CUDA_CHECK(cudaMalloc(&pre_proj_buffers[p],
                              static_cast<size_t>(tile_elems) * sizeof(float)));
        CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_events[p],
                                            cudaEventDisableTiming));
    }

    /* W_ot weight matrix (proj_k x proj_n) */
    float* W_out_device = nullptr;
    CUDA_CHECK(cudaMalloc(&W_out_device,
                          static_cast<size_t>(proj_k) * proj_n * sizeof(float)));
    {
        std::vector<float> W_host(static_cast<size_t>(proj_k) * proj_n, 0.01f);
        CUDA_CHECK(cudaMemcpy(W_out_device, W_host.data(),
                              W_host.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    /* final output buffer */
    float* output_device = nullptr;
    const size_t total_output_elems = static_cast<size_t>(num_tiles) * output_tile_elems;
    CUDA_CHECK(cudaMalloc(&output_device, total_output_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(output_device, 0, total_output_elems * sizeof(float)));

    /* CUDA streams */
    cudaStream_t compute_stream, proj_stream, comm_cuda_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&proj_stream));
    CUDA_CHECK(cudaStreamCreate(&comm_cuda_stream));

    /* cuBLAS */
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    /* Per-tile timing */
    std::vector<float>  compute_ms(num_tiles, 0.0f);
    std::vector<float>  proj_ms(num_tiles, 0.0f);
    std::vector<double> tile_ready_wall(num_tiles, 0.0);

    // CPU enqueue timestamps (not GPU execution times)
    std::vector<double> compute_enqueue_start(num_tiles, 0.0);
    std::vector<double> compute_enqueue_end(num_tiles, 0.0);
    std::vector<double> proj_enqueue_start(num_tiles, 0.0);
    std::vector<double> proj_enqueue_end(num_tiles, 0.0);
    // Comm wall times (meaningful in non-overlap modes)
    std::vector<double> comm_start_wall(num_tiles, 0.0);
    std::vector<double> comm_end_wall(num_tiles, 0.0);

    std::vector<cudaEvent_t> ev_comp_start(num_tiles), ev_comp_stop(num_tiles);
    std::vector<cudaEvent_t> ev_proj_start(num_tiles), ev_proj_stop(num_tiles);
    for (int t = 0; t < num_tiles; ++t) {
        CUDA_CHECK(cudaEventCreate(&ev_comp_start[t]));
        CUDA_CHECK(cudaEventCreate(&ev_comp_stop[t]));
        CUDA_CHECK(cudaEventCreate(&ev_proj_start[t]));
        CUDA_CHECK(cudaEventCreate(&ev_proj_stop[t]));
    }

    /* Comm-thread state (overlap mode only)*/
    SharedState state;
    TimingStats timing;
    timing.per_tile.resize(num_tiles);
    std::thread comm_thread_handle;

    /* Helper: compute + project one tile */

    auto compute_one = [&](int tile, int buf) {
        float base_value = static_cast<float>((rank + 1) * 1000 + (tile + 1));
        int threads = 256;
        int blocks  = (tile_elems + threads - 1) / threads;

        compute_enqueue_start[tile] = MPI_Wtime();
        CUDA_CHECK(cudaEventRecord(ev_comp_start[tile], compute_stream));

        fill_tile_kernel<<<blocks, threads, 0, compute_stream>>>(
            pre_proj_buffers[buf], tile_elems, base_value, fma_iters);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_comp_stop[tile], compute_stream));
        CUDA_CHECK(cudaEventRecord(compute_done_events[buf], compute_stream));
        compute_enqueue_end[tile] = MPI_Wtime();
    };

    auto project_one = [&](int tile, int buf) {
        CUDA_CHECK(cudaStreamWaitEvent(proj_stream,
                                       compute_done_events[buf], 0));
        proj_enqueue_start[tile] = MPI_Wtime();
        CUDA_CHECK(cudaEventRecord(ev_proj_start[tile], proj_stream));

        project_tile(cublas_handle, pre_proj_buffers[buf], W_out_device,
                     send_buffers[buf], proj_m, proj_k, proj_n);

        CUDA_CHECK(cudaEventRecord(ev_proj_stop[tile], proj_stream));
        proj_enqueue_end[tile] = MPI_Wtime();
    };

    // Variant for non-overlap modes where proj writes to a specific offset
    // within the send buffer (used for serialized_full aggregation).
    auto project_one_at_offset = [&](int tile, int buf, float* dest) {
        CUDA_CHECK(cudaStreamWaitEvent(proj_stream,
                                       compute_done_events[buf], 0));
        proj_enqueue_start[tile] = MPI_Wtime();
        CUDA_CHECK(cudaEventRecord(ev_proj_start[tile], proj_stream));

        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            proj_n, proj_m, proj_k,
            &alpha,
            W_out_device, proj_n,
            pre_proj_buffers[buf], proj_k,
            &beta,
            dest, proj_n
        ));

        CUDA_CHECK(cudaEventRecord(ev_proj_stop[tile], proj_stream));
        proj_enqueue_end[tile] = MPI_Wtime();
    };

    /* SERIALIZED_FULL MODE */
    /* compute all -> project all (into contiguous buffer) -> 1 allreduce */

    // Needs a large contiguous send/recv buffer for the full output.
    float* full_send_buf = nullptr;
    float* full_recv_buf = nullptr;

    if (mode == Mode::SERIALIZED_FULL) {
        CUDA_CHECK(cudaMalloc(&full_send_buf, total_output_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&full_recv_buf, total_output_elems * sizeof(float)));
    }

    auto run_serialized_full = [&](double t0_wall) {
        // compute all tiles
        for (int tile = 0; tile < num_tiles; ++tile) {
            compute_one(tile, tile);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // project all tiles into contiguous full_send_buf
        for (int tile = 0; tile < num_tiles; ++tile) {
            float* dest = full_send_buf +
                          static_cast<size_t>(tile) * output_tile_elems;
            project_one_at_offset(tile, tile, dest);
        }
        CUDA_CHECK(cudaStreamSynchronize(proj_stream));

        // one big allreduce
        comm_start_wall[0] = MPI_Wtime();

        if (total_output_elems <= static_cast<size_t>(INT_MAX)) {
            MPI_CHECK(MPI_Allreduce(
                full_send_buf, full_recv_buf,
                (int)total_output_elems, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD
            ));
        } else {
            // Chunked allreduce for very large outputs
            size_t offset = 0;
            while (offset < total_output_elems) {
                int chunk = (int)std::min(total_output_elems - offset,
                                          (size_t)INT_MAX);
                MPI_CHECK(MPI_Allreduce(
                    full_send_buf + offset, full_recv_buf + offset,
                    chunk, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD
                ));
                offset += chunk;
            }
        }

        comm_end_wall[0] = MPI_Wtime();
        // Assign same comm time to all tiles for reporting consistency.
        for (int t = 1; t < num_tiles; ++t) {
            comm_start_wall[t] = comm_start_wall[0];
            comm_end_wall[t]   = comm_end_wall[0];
        }

        // Copy result to output
        CUDA_CHECK(cudaMemcpyAsync(
            output_device, full_recv_buf,
            total_output_elems * sizeof(float),
            cudaMemcpyDeviceToDevice, comm_cuda_stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(comm_cuda_stream));
    };

    /* SERIALIZED MODE */
    /* compute all -> project all -> allreduce per tile */

    auto run_serialized = [&](double t0_wall) {
        for (int tile = 0; tile < num_tiles; ++tile)
            compute_one(tile, tile);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        for (int tile = 0; tile < num_tiles; ++tile)
            project_one(tile, tile);
        CUDA_CHECK(cudaStreamSynchronize(proj_stream));

        for (int tile = 0; tile < num_tiles; ++tile) {
            comm_start_wall[tile] = MPI_Wtime();
            MPI_CHECK(MPI_Allreduce(
                send_buffers[tile], recv_buffers[tile],
                (int)output_tile_elems, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD
            ));
            comm_end_wall[tile] = MPI_Wtime();

            float* out = output_device +
                         static_cast<size_t>(tile) * output_tile_elems;
            CUDA_CHECK(cudaMemcpyAsync(
                out, recv_buffers[tile],
                output_tile_elems * sizeof(float),
                cudaMemcpyDeviceToDevice, comm_cuda_stream
            ));
        }
        CUDA_CHECK(cudaStreamSynchronize(comm_cuda_stream));
    };

    /* TILED MODE */
    /* compute+project per tile -> sync -> allreduce per tile */

    auto run_tiled = [&](double t0_wall) {
        for (int tile = 0; tile < num_tiles; ++tile) {
            compute_one(tile, tile);
            project_one(tile, tile);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(proj_stream));

        for (int tile = 0; tile < num_tiles; ++tile) {
            comm_start_wall[tile] = MPI_Wtime();
            MPI_CHECK(MPI_Allreduce(
                send_buffers[tile], recv_buffers[tile],
                (int)output_tile_elems, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD
            ));
            comm_end_wall[tile] = MPI_Wtime();

            float* out = output_device +
                         static_cast<size_t>(tile) * output_tile_elems;
            CUDA_CHECK(cudaMemcpyAsync(
                out, recv_buffers[tile],
                output_tile_elems * sizeof(float),
                cudaMemcpyDeviceToDevice, comm_cuda_stream
            ));
        }
        CUDA_CHECK(cudaStreamSynchronize(comm_cuda_stream));
    };

    /* OVERLAP MODE */
    /* compute+project per tile -> enqueue for allreduce */
    /* Supports message aggregation: --aggregate_tiles N */

    auto run_overlap = [&](double t0_wall) {
        state.buffer_free.assign(num_buffers, true);
        state.done = false;
        while (!state.ready_tiles.empty()) state.ready_tiles.pop();

        comm_thread_handle = std::thread(
            comm_thread_fn,
            &state, &timing, MPI_COMM_WORLD,
            output_device,
            (int)output_tile_elems,
            comm_cuda_stream,
            max_inflight,
            t0_wall
        );

        const int num_groups = num_tiles / aggregate_tiles;

        for (int grp = 0; grp < num_groups; ++grp) {
            int first_tile = grp * aggregate_tiles;
            int buf = grp % num_buffers;

            // Wait for buffer
            {
                std::unique_lock<std::mutex> lock(state.mtx);
                state.cv.wait(lock, [&] { return state.buffer_free[buf]; });
                state.buffer_free[buf] = false;
            }

            // Compute + project each tile in the aggregation group.
            // Each tile gets its own pre_proj buffer within this group's
            // buffer set to prevent compute/projection stream races.
            for (int t = 0; t < aggregate_tiles; ++t) {
                int tile = first_tile + t;
                int pre_buf = buf * aggregate_tiles + t;  // per-tile pre_proj slot

                // Compute
                float base_value = static_cast<float>((rank + 1) * 1000 + (tile + 1));
                int threads = 256;
                int blocks  = (tile_elems + threads - 1) / threads;

                compute_enqueue_start[tile] = MPI_Wtime();
                CUDA_CHECK(cudaEventRecord(ev_comp_start[tile], compute_stream));

                fill_tile_kernel<<<blocks, threads, 0, compute_stream>>>(
                    pre_proj_buffers[pre_buf], tile_elems, base_value, fma_iters);
                CUDA_CHECK(cudaGetLastError());

                CUDA_CHECK(cudaEventRecord(ev_comp_stop[tile], compute_stream));
                CUDA_CHECK(cudaEventRecord(compute_done_events[pre_buf], compute_stream));
                compute_enqueue_end[tile] = MPI_Wtime();

                // Project into aggregated send buffer at correct offset
                float* proj_dest = send_buffers[buf] +
                                   static_cast<size_t>(t) * output_tile_elems;
                CUDA_CHECK(cudaStreamWaitEvent(proj_stream,
                                               compute_done_events[pre_buf], 0));

                proj_enqueue_start[tile] = MPI_Wtime();
                CUDA_CHECK(cudaEventRecord(ev_proj_start[tile], proj_stream));

                const float alpha = 1.0f, beta = 0.0f;
                CUBLAS_CHECK(cublasSgemm(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    proj_n, proj_m, proj_k,
                    &alpha,
                    W_out_device, proj_n,
                    pre_proj_buffers[pre_buf], proj_k,
                    &beta,
                    proj_dest, proj_n
                ));

                CUDA_CHECK(cudaEventRecord(ev_proj_stop[tile], proj_stream));
                proj_enqueue_end[tile] = MPI_Wtime();
            }

            // Record ready event after all tiles in group are projected
            CUDA_CHECK(cudaEventRecord(ready_events[buf], proj_stream));

            // Enqueue aggregated group for communication.
            // We tag it with first_tile as the tile_id for timing/output offset.
            TileDesc desc;
            desc.tile_id      = first_tile;
            desc.buf_id       = buf;
            desc.send_ptr     = send_buffers[buf];
            desc.recv_ptr     = recv_buffers[buf];
            desc.send_elems   = (int)(aggregate_tiles * output_tile_elems);
            desc.ready_event  = ready_events[buf];
            desc.producer_wall = MPI_Wtime();
            tile_ready_wall[first_tile] = desc.producer_wall;

            {
                std::lock_guard<std::mutex> lock(state.mtx);
                state.ready_tiles.push(desc);
            }
            state.cv.notify_all();
        }

        {
            std::lock_guard<std::mutex> lock(state.mtx);
            state.done = true;
        }
        state.cv.notify_all();
        comm_thread_handle.join();
    };

    /* Run the selected mode */

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double total_t0 = MPI_Wtime();

    switch (mode) {
        case Mode::SERIALIZED_FULL: run_serialized_full(total_t0); break;
        case Mode::SERIALIZED:      run_serialized(total_t0);      break;
        case Mode::TILED:           run_tiled(total_t0);           break;
        case Mode::OVERLAP:         run_overlap(total_t0);         break;
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double total_t1 = MPI_Wtime();

    /* Collect GPU-side timing */
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int t = 0; t < num_tiles; ++t) {
        CUDA_CHECK(cudaEventElapsedTime(&compute_ms[t],
                                        ev_comp_start[t], ev_comp_stop[t]));
        CUDA_CHECK(cudaEventElapsedTime(&proj_ms[t],
                                        ev_proj_start[t], ev_proj_stop[t]));
    }

    /* Verification */

    std::vector<float> output_host(total_output_elems);
    CUDA_CHECK(cudaMemcpy(output_host.data(), output_device,
                          total_output_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        float expected_sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            float base = static_cast<float>((r + 1) * 1000 + (tile + 1));
            float kernel_out = compute_expected_kernel_output(base, fma_iters);
            expected_sum += kernel_out * 0.01f * static_cast<float>(proj_k);
        }

        float tol = std::fabs(expected_sum) * 1e-3f + 1e-4f;

        size_t offset = static_cast<size_t>(tile) * output_tile_elems;
        for (size_t i = 0; i < output_tile_elems; ++i) {
            if (std::fabs(output_host[offset + i] - expected_sum) > tol) {
                bad++;
                if (bad < 10) {
                    std::cerr << "Rank " << rank
                              << " tile " << tile
                              << " elem " << i
                              << ": got " << output_host[offset + i]
                              << ", expected " << expected_sum
                              << " (diff " << std::fabs(output_host[offset + i] - expected_sum)
                              << ", tol " << tol << ")\n";
                }
            }
        }
    }

    int global_bad = 0;
    MPI_CHECK(MPI_Allreduce(&bad, &global_bad, 1, MPI_INT, MPI_SUM,
                            MPI_COMM_WORLD));

    /* Aggregate timing */

    double local_total_ms   = (total_t1 - total_t0) * 1e3;
    double local_compute_ms = 0, local_proj_ms = 0, local_comm_ms = 0;
    for (int t = 0; t < num_tiles; ++t) {
        local_compute_ms += compute_ms[t];
        local_proj_ms    += proj_ms[t];
    }

    if (mode == Mode::OVERLAP) {
        // Comm timing from the comm thread
        int num_groups = num_tiles / aggregate_tiles;
        for (int g = 0; g < num_groups; ++g) {
            int first = g * aggregate_tiles;
            local_comm_ms += timing.per_tile[first].comm_ms;
        }
    } else if (mode == Mode::SERIALIZED_FULL) {
        local_comm_ms = (comm_end_wall[0] - comm_start_wall[0]) * 1e3;
    } else {
        for (int t = 0; t < num_tiles; ++t)
            local_comm_ms += (comm_end_wall[t] - comm_start_wall[t]) * 1e3;
    }

    double max_total_ms = 0, max_compute_ms = 0;
    double max_proj_ms = 0, max_comm_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_total_ms,   &max_total_ms,   1,
                         MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_compute_ms, &max_compute_ms, 1,
                         MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_proj_ms,    &max_proj_ms,    1,
                         MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&local_comm_ms,    &max_comm_ms,    1,
                         MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        double serial_sum = max_compute_ms + max_proj_ms + max_comm_ms;

        std::cout << "\nVerification: "
                  << (global_bad == 0 ? "PASS" : "FAIL")
                  << "  bad_count=" << global_bad << "\n";

        std::cout << "\nMode: " << mode_str(mode) << "\n";
        std::cout << "Timing summary (max over ranks):\n"
                  << "  total wall time               : " << max_total_ms   << " ms\n"
                  << "  sum compute time (GPU events) : " << max_compute_ms << " ms\n"
                  << "  sum projection time (GPU evt) : " << max_proj_ms    << " ms\n"
                  << "  sum comm time (wall clock)    : " << max_comm_ms    << " ms\n"
                  << "  compute+proj+comm             : " << serial_sum     << " ms\n";

        double overlap_eff = 0.0;
        if (serial_sum > 0)
            overlap_eff = 1.0 - max_total_ms / serial_sum;
        std::cout << "  estimated overlap efficiency  : "
                  << (overlap_eff * 100.0) << " %\n";

        size_t total_bytes = total_output_elems * sizeof(float);

        size_t bytes_per_mpi_msg = 0;
        if (mode == Mode::SERIALIZED_FULL) {
            bytes_per_mpi_msg = total_bytes;
        } else if (mode == Mode::OVERLAP) {
            bytes_per_mpi_msg = static_cast<size_t>(aggregate_tiles)
                              * output_tile_elems * sizeof(float);
        } else {
            bytes_per_mpi_msg = output_tile_elems * sizeof(float);
        }

        std::cout << "\n  bytes per MPI message         : " << bytes_per_mpi_msg << "\n"
                  << "  total bytes allreduced        : " << total_bytes << "\n";

        if (max_comm_ms > 0) {
            double bw = static_cast<double>(total_bytes) / (max_comm_ms * 1e-3) / 1e9;
            std::cout << "  effective payload BW          : " << bw << " GB/s\n";
        }

        std::cout << "\nNote: 'estimated overlap efficiency' is a synthetic metric.\n"
                  << "For true 'fraction of comm hidden', compare wall times across modes:\n"
                  << "  comm_hidden_ms   = tiled_wall - overlap_wall\n"
                  << "  comm_hidden_frac = comm_hidden_ms / tiled_comm_ms\n";
    }

    /* Per-tile CSV timeline */
    {
        // Create results directory (rank 0 creates, barrier ensures it exists).
        if (rank == 0) {
            (void)system("mkdir -p results");
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        std::string fname = "results/tile_timeline_"
            + std::string(mode_str(mode))
            + "_rank" + std::to_string(rank)
            + "_ngpu" + std::to_string(world_size)
            + "_tiles" + std::to_string(num_tiles)
            + "_buf" + std::to_string(num_buffers)
            + "_inf" + std::to_string(max_inflight)
            + "_agg" + std::to_string(aggregate_tiles)
            + ".csv";
        std::ofstream csv(fname);

        // Header
        csv << "tile_id"
            << ",group_id,group_first_tile,is_group_leader"
            << ",compute_gpu_ms,proj_gpu_ms"
            << ",compute_enqueue_start_ms,compute_enqueue_end_ms"
            << ",proj_enqueue_start_ms,proj_enqueue_end_ms"
            << ",comm_start_ms,comm_end_ms";
        if (mode == Mode::OVERLAP) {
            csv << ",queue_wait_ms,event_sync_ms,mpi_comm_ms"
                << ",d2d_copy_ms,comm_total_ms"
                << ",copy_start_ms,copy_end_ms"
                << ",tile_ready_offset_ms";
        }
        csv << "\n";

        for (int t = 0; t < num_tiles; ++t) {
            int group_id = t / aggregate_tiles;
            int grp_first = group_id * aggregate_tiles;
            int is_leader = (t == grp_first) ? 1 : 0;

            csv << t
                << "," << group_id
                << "," << grp_first
                << "," << is_leader
                << "," << compute_ms[t]
                << "," << proj_ms[t]
                << "," << (compute_enqueue_start[t] - total_t0) * 1e3
                << "," << (compute_enqueue_end[t] - total_t0) * 1e3
                << "," << (proj_enqueue_start[t] - total_t0) * 1e3
                << "," << (proj_enqueue_end[t] - total_t0) * 1e3;

            if (mode == Mode::OVERLAP) {
                // Use the comm thread's actual timestamps for comm columns.
                // For aggregated groups, timing lives at first_tile of the group.
                int grp_first = (t / aggregate_tiles) * aggregate_tiles;
                const auto& tt = timing.per_tile[grp_first];
                csv << "," << tt.comm_start_offset_ms
                    << "," << tt.comm_end_offset_ms
                    << "," << tt.queue_wait_ms
                    << "," << tt.event_sync_ms
                    << "," << tt.comm_ms
                    << "," << tt.d2d_copy_ms
                    << "," << tt.total_ms
                    << "," << tt.copy_start_offset_ms
                    << "," << tt.copy_end_offset_ms
                    << "," << (tile_ready_wall[grp_first] - total_t0) * 1e3;
            } else {
                // Non-overlap modes: use direct wall timestamps
                csv << "," << (comm_start_wall[t] - total_t0) * 1e3
                    << "," << (comm_end_wall[t] - total_t0) * 1e3;
            }
            csv << "\n";
        }
        csv.close();
        if (rank == 0)
            std::cout << "\nTimeline CSV: " << fname << "\n";
    }

    /* Cleanup */

    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    if (full_send_buf) CUDA_CHECK(cudaFree(full_send_buf));
    if (full_recv_buf) CUDA_CHECK(cudaFree(full_recv_buf));

    for (int t = 0; t < num_tiles; ++t) {
        CUDA_CHECK(cudaEventDestroy(ev_comp_start[t]));
        CUDA_CHECK(cudaEventDestroy(ev_comp_stop[t]));
        CUDA_CHECK(cudaEventDestroy(ev_proj_start[t]));
        CUDA_CHECK(cudaEventDestroy(ev_proj_stop[t]));
    }
    for (int b = 0; b < n_bufs; ++b) {
        CUDA_CHECK(cudaEventDestroy(ready_events[b]));
        CUDA_CHECK(cudaFree(send_buffers[b]));
        CUDA_CHECK(cudaFree(recv_buffers[b]));
    }
    for (int p = 0; p < n_preproj; ++p) {
        CUDA_CHECK(cudaEventDestroy(compute_done_events[p]));
        CUDA_CHECK(cudaFree(pre_proj_buffers[p]));
    }
    CUDA_CHECK(cudaFree(W_out_device));
    CUDA_CHECK(cudaFree(output_device));

    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(proj_stream));
    CUDA_CHECK(cudaStreamDestroy(comm_cuda_stream));

    MPI_CHECK(MPI_Finalize());
    return 0;
}
