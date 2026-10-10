#include "run_drivers.cuh"

#include <cuda.h>

#include <cstdlib>
#include <cstring>
#include <vector>

static const char* MODE_NAMES[MODE_COUNT] = {
    "attn", "attn_tiled", "megatron", "sync", "overlap", "stream", "comm"
};

const char* mode_name(int mode)
{
    return (mode >= 0 && mode < MODE_COUNT) ? MODE_NAMES[mode] : "?";
}

int parse_mode(const char* s)
{
    for (int i = 0; i < MODE_COUNT; i++)
        if (std::strcmp(s, MODE_NAMES[i]) == 0) return i;
    return -1;
}

// AllReduce on communicator slot `slot` (tile t uses t % ncomm).
static void allreduce(const RunCtx& c, int slot, const float* send, float* recv, size_t n, const AttnParams& p)
{
    if (p.skip_comm) return;
    NCCL_CHECK(ncclAllReduce(send, recv, n, ncclFloat, ncclSum, c.nccls[slot], c.comm_streams[slot]));
}

static void sync_comm_streams(const RunCtx& c)
{
    for (int i = 0; i < c.ncomm; i++)
        CUDA_CHECK(cudaStreamSynchronize(c.comm_streams[i]));
}

static double finish_timing(double t0, MPI_Comm comm)
{
    double local = (MPI_Wtime() - t0) * 1e3;
    double global = local;
    MPI_CHECK(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm));
    return global;
}

// ---------------------------------------------------------------------------

static double run_attn_only(const Weights& w, const AttnParams& p, const RunCtx& c, bool tiled)
{
    float* O_all = nullptr;
    CUDA_CHECK(cudaMalloc(&O_all, (size_t)p.S * p.d_local * sizeof(float)));
    FlashAttnTCPartials ws = alloc_workspace(p, tiled ? p.tile : p.S);

    MPI_CHECK(MPI_Barrier(c.comm));
    double t0 = MPI_Wtime();

    if (tiled) {
        for (int t = 0; t < p.num_tiles; t++)
            launch_attention(w, O_all + (size_t)t * p.tile * p.d_local, ws, t * p.tile, p.tile, p, c.compute_stream);
    } else {
        launch_attention(w, O_all, ws, 0, p.S, p, c.compute_stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(c.compute_stream));

    double ms = finish_timing(t0, c.comm);
    free_workspace(ws);
    CUDA_CHECK(cudaFree(O_all));
    return ms;
}

static double run_megatron(const Weights& w, float* output, const AttnParams& p, const RunCtx& c)
{
    float* O_all = nullptr;
    float* send_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&O_all,    (size_t)p.S * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.S * p.d * sizeof(float)));
    FlashAttnTCPartials ws = alloc_workspace(p, p.S);
    cudaEvent_t attn_done, proj_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&attn_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&proj_done, cudaEventDisableTiming));

    MPI_CHECK(MPI_Barrier(c.comm));
    double t0 = MPI_Wtime();

    launch_attention(w, O_all, ws, 0, p.S, p, c.compute_stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(attn_done, c.compute_stream));

    CUDA_CHECK(cudaStreamWaitEvent(c.proj_stream, attn_done, 0));
    project(c, w, O_all, send_buf, p.S, p);
    CUDA_CHECK(cudaEventRecord(proj_done, c.proj_stream));

    CUDA_CHECK(cudaStreamWaitEvent(c.comm_streams[0], proj_done, 0));
    allreduce(c, 0, send_buf, output, (size_t)p.S * p.d, p);
    CUDA_CHECK(cudaStreamSynchronize(c.comm_streams[0]));

    double ms = finish_timing(t0, c.comm);

    CUDA_CHECK(cudaEventDestroy(attn_done));
    CUDA_CHECK(cudaEventDestroy(proj_done));
    free_workspace(ws);
    CUDA_CHECK(cudaFree(O_all));
    CUDA_CHECK(cudaFree(send_buf));
    return ms;
}

static double run_sync(const Weights& w, float* output, const AttnParams& p, const RunCtx& c)
{
    float* O_tile = nullptr;
    float* send_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&O_tile,   (size_t)p.tile * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.tile * p.d * sizeof(float)));
    FlashAttnTCPartials ws = alloc_workspace(p, p.tile);

    MPI_CHECK(MPI_Barrier(c.comm));
    double t0 = MPI_Wtime();

    for (int t = 0; t < p.num_tiles; t++) {
        launch_attention(w, O_tile, ws, t * p.tile, p.tile, p, c.compute_stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(c.compute_stream));

        project(c, w, O_tile, send_buf, p.tile, p);
        CUDA_CHECK(cudaStreamSynchronize(c.proj_stream));

        allreduce(c, 0, send_buf, output + (size_t)t * p.tile * p.d, (size_t)p.tile * p.d, p);
        CUDA_CHECK(cudaStreamSynchronize(c.comm_streams[0]));
    }

    double ms = finish_timing(t0, c.comm);
    free_workspace(ws);
    CUDA_CHECK(cudaFree(O_tile));
    CUDA_CHECK(cudaFree(send_buf));
    return ms;
}

// Fully asynchronous pipeline. The host enqueues every tile up front; the
// three streams are ordered only by events:
//   compute:  attn(t)  [waits proj_done(t - nbuf): O_tile[b] free]
//   proj:     gemm(t)  [waits attn_done(t), comm_done(t - nbuf): send_buf[b] free]
//   comm[t % ncomm]: allreduce(t) [waits proj_done(t)]
static double run_overlap(const Weights& w, float* output, const AttnParams& p, const RunCtx& c)
{
    const int nbuf = p.nbuf;
    std::vector<float*> O_tile(nbuf);
    std::vector<float*> send_buf(nbuf);
    std::vector<cudaEvent_t> attn_done(nbuf), proj_done(nbuf), comm_done(nbuf);
    for (int b = 0; b < nbuf; b++) {
        CUDA_CHECK(cudaMalloc(&O_tile[b],   (size_t)p.tile * p.d_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&send_buf[b], (size_t)p.tile * p.d * sizeof(float)));
        CUDA_CHECK(cudaEventCreateWithFlags(&attn_done[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&proj_done[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&comm_done[b], cudaEventDisableTiming));
    }
    FlashAttnTCPartials ws = alloc_workspace(p, p.tile);   // shared: attention + combine are stream-ordered

    MPI_CHECK(MPI_Barrier(c.comm));
    double t0 = MPI_Wtime();

    for (int t = 0; t < p.num_tiles; t++) {
        const int b = t % nbuf;

        if (t >= nbuf)
            CUDA_CHECK(cudaStreamWaitEvent(c.compute_stream, proj_done[b], 0));
        launch_attention(w, O_tile[b], ws, t * p.tile, p.tile, p, c.compute_stream);
        CUDA_CHECK(cudaEventRecord(attn_done[b], c.compute_stream));

        CUDA_CHECK(cudaStreamWaitEvent(c.proj_stream, attn_done[b], 0));
        if (t >= nbuf)
            CUDA_CHECK(cudaStreamWaitEvent(c.proj_stream, comm_done[b], 0));
        project(c, w, O_tile[b], send_buf[b], p.tile, p);
        CUDA_CHECK(cudaEventRecord(proj_done[b], c.proj_stream));

        const int slot = t % c.ncomm;
        CUDA_CHECK(cudaStreamWaitEvent(c.comm_streams[slot], proj_done[b], 0));
        allreduce(c, slot, send_buf[b], output + (size_t)t * p.tile * p.d, (size_t)p.tile * p.d, p);
        CUDA_CHECK(cudaEventRecord(comm_done[b], c.comm_streams[slot]));
    }
    CUDA_CHECK(cudaGetLastError());
    sync_comm_streams(c);

    double ms = finish_timing(t0, c.comm);

    for (int b = 0; b < nbuf; b++) {
        CUDA_CHECK(cudaEventDestroy(attn_done[b]));
        CUDA_CHECK(cudaEventDestroy(proj_done[b]));
        CUDA_CHECK(cudaEventDestroy(comm_done[b]));
        CUDA_CHECK(cudaFree(O_tile[b]));
        CUDA_CHECK(cudaFree(send_buf[b]));
    }
    free_workspace(ws);
    return ms;
}

// Single attention launch over all S; the kernel bumps tile_counters[t] once
// per finished block. The projection stream blocks on each counter with a
// stream memory op (no host involvement, no SM spinning), so projection and
// AllReduce of tile t start as soon as its rows exist while later tiles are
// still computing at full-GPU efficiency.
static double run_stream(const Weights& w, float* output, const AttnParams& p, const RunCtx& c)
{
    float* O_all = nullptr;
    float* send_buf = nullptr;
    int* counters = nullptr;
    CUDA_CHECK(cudaMalloc(&O_all,    (size_t)p.S * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.S * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&counters, (size_t)p.num_tiles * sizeof(int)));
    FlashAttnTCPartials ws = alloc_workspace(p, p.S);
    const int target = blocks_per_tile(p);

    cudaEvent_t reset_done;
    std::vector<cudaEvent_t> proj_done(p.num_tiles);
    CUDA_CHECK(cudaEventCreateWithFlags(&reset_done, cudaEventDisableTiming));
    for (int t = 0; t < p.num_tiles; t++)
        CUDA_CHECK(cudaEventCreateWithFlags(&proj_done[t], cudaEventDisableTiming));

    MPI_CHECK(MPI_Barrier(c.comm));
    double t0 = MPI_Wtime();

    CUDA_CHECK(cudaMemsetAsync(counters, 0, (size_t)p.num_tiles * sizeof(int), c.compute_stream));
    CUDA_CHECK(cudaEventRecord(reset_done, c.compute_stream));
    launch_attention_raw(w, O_all, ws, 0, p.S, p, c.compute_stream, counters, p.tile);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamWaitEvent(c.proj_stream, reset_done, 0));   // never observe a stale counter
    for (int t = 0; t < p.num_tiles; t++) {
        CU_CHECK(cuStreamWaitValue32((CUstream)c.proj_stream,
                                     (CUdeviceptr)(counters + t), (cuuint32_t)target,
                                     CU_STREAM_WAIT_VALUE_GEQ));
        float* O_t = O_all + (size_t)t * p.tile * p.d_local;
        launch_combine(ws, O_t, t * p.tile, p.tile, p, c.proj_stream);
        project(c, w, O_t, send_buf + (size_t)t * p.tile * p.d, p.tile, p);
        CUDA_CHECK(cudaEventRecord(proj_done[t], c.proj_stream));

        const int slot = t % c.ncomm;
        CUDA_CHECK(cudaStreamWaitEvent(c.comm_streams[slot], proj_done[t], 0));
        allreduce(c, slot, send_buf + (size_t)t * p.tile * p.d, output + (size_t)t * p.tile * p.d,
                  (size_t)p.tile * p.d, p);
    }
    CUDA_CHECK(cudaStreamSynchronize(c.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(c.proj_stream));
    sync_comm_streams(c);

    double ms = finish_timing(t0, c.comm);

    CUDA_CHECK(cudaEventDestroy(reset_done));
    for (int t = 0; t < p.num_tiles; t++)
        CUDA_CHECK(cudaEventDestroy(proj_done[t]));
    free_workspace(ws);
    CUDA_CHECK(cudaFree(O_all));
    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(counters));
    return ms;
}

// Communication only: the per-tile AllReduces of a full pass, back to back on
// communicator 0, nothing else in flight. Sweeping --tile gives t_allreduce(bytes).
static double run_comm_only(float* output, const AttnParams& p, const RunCtx& c)
{
    float* send_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.S * p.d * sizeof(float)));
    CUDA_CHECK(cudaMemset(send_buf, 0, (size_t)p.S * p.d * sizeof(float)));

    MPI_CHECK(MPI_Barrier(c.comm));
    double t0 = MPI_Wtime();
    for (int t = 0; t < p.num_tiles; t++)
        allreduce(c, 0, send_buf + (size_t)t * p.tile * p.d, output + (size_t)t * p.tile * p.d,
                  (size_t)p.tile * p.d, p);
    CUDA_CHECK(cudaStreamSynchronize(c.comm_streams[0]));
    double ms = finish_timing(t0, c.comm);

    CUDA_CHECK(cudaFree(send_buf));
    return ms;
}

double run_mode(int mode, const Weights& w, float* output,
                   const AttnParams& p, const RunCtx& c)
{
    switch (mode) {
        case MODE_ATTN:       return run_attn_only(w, p, c, false);
        case MODE_ATTN_TILED: return run_attn_only(w, p, c, true);
        case MODE_MEGATRON:   return run_megatron(w, output, p, c);
        case MODE_SYNC:       return run_sync(w, output, p, c);
        case MODE_OVERLAP:    return run_overlap(w, output, p, c);
        case MODE_STREAM:     return run_stream(w, output, p, c);
        case MODE_COMM:       return run_comm_only(output, p, c);
    }
    fprintf(stderr, "unknown mode %d\n", mode);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 0.0;
}
