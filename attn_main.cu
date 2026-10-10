#include "attn_main.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

void init_mpi(int* argc, char*** argv, int& rank, int& world_size)
{
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided));
    if (provided < MPI_THREAD_MULTIPLE) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cerr << "ERROR: MPI does not provide MPI_THREAD_MULTIPLE.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
}

int init_cuda(int rank)
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) std::cerr << "No CUDA devices found.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

AttnParams make_params(int S, int d, int H, int tile, int B_M, int B_N,
                       int kernel, int nbuf, int split, int causal, int world_size)
{
    AttnParams p;
    p.S = S;
    p.d = d;
    p.H = H;
    p.R = world_size;
    p.H_local = (world_size > 0) ? H / world_size : 0;
    p.d_h = (H > 0) ? d / H : 0;
    p.d_local = p.H_local * p.d_h;
    p.tile = tile;
    p.num_tiles = (tile > 0) ? S / tile : 0;
    p.B_M = B_M;
    p.B_N = B_N;
    p.kernel = kernel;
    p.nbuf = nbuf;
    p.split = split;
    p.causal = causal;
    p.skip_proj = 0;
    p.skip_comm = 0;
    float scale = 1.0f / sqrtf((float)p.d_h);
    p.flash    = { S, p.d_h, p.H_local, B_M, B_N, causal, scale };
    p.flash_tc = { S, p.d_h, p.H_local, split, causal, scale };
    return p;
}

const char* attn_check_params(const AttnParams& p, int cuda_device)
{
    if (p.H % p.R != 0) return "H must be divisible by the number of ranks";
    if (p.d % p.H != 0) return "d must be divisible by H";
    if (p.tile <= 0 || p.S % p.tile != 0) return "S must be a multiple of tile";
    if (p.nbuf < 1) return "nbuf must be >= 1";
    if (p.kernel == KERNEL_TC) return flash_attn_tc_check_params(p.flash_tc, p.tile);
    if (p.kernel == KERNEL_V1) {
        if (p.split != 1) return "v1: split-KV not supported";
        return flash_attn_check_params(p.flash, p.tile, cuda_device);
    }
    return "unknown kernel";
}

const char* kernel_name(int kernel)
{
    return kernel == KERNEL_TC ? "tc" : "v1";
}

int parse_kernel(const char* s)
{
    if (std::strcmp(s, "v1") == 0) return KERNEL_V1;
    if (std::strcmp(s, "tc") == 0) return KERNEL_TC;
    return -1;
}

static void rand_fill(float* d_ptr, size_t n, float scale, cudaStream_t stream)
{
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++)
        h[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
    // Pageable source: cudaMemcpyAsync completes the host-side read before returning.
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, h.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
}

Weights allocate_and_fill(const AttnParams& p, int rank, int seed, cudaStream_t stream)
{
    Weights w;
    const size_t qkv_n = (size_t)p.H_local * p.S * p.d_h;
    CUDA_CHECK(cudaMalloc(&w.X,   (size_t)p.S * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_q, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_k, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_v, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_O, (size_t)p.d_local * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.Q,  qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.K,  qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.V,  qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.Qh, qkv_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w.Kh, qkv_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w.Vh, qkv_n * sizeof(half)));

    // Scales chosen so attention logits have O(1) spread: a flat softmax would
    // hide kernel bugs behind a uniform average.
    srand(seed);
    rand_fill(w.X, (size_t)p.S * p.d, 1.0f, stream);

    srand(seed * 31 + rank + 1);
    rand_fill(w.W_q, (size_t)p.H_local * p.d * p.d_h, 0.15f, stream);
    rand_fill(w.W_k, (size_t)p.H_local * p.d * p.d_h, 0.15f, stream);
    rand_fill(w.W_v, (size_t)p.H_local * p.d * p.d_h, 0.10f, stream);
    rand_fill(w.W_O, (size_t)p.d_local * p.d, 0.05f, stream);
    return w;
}

void compute_qkv(const Weights& w, const AttnParams& p, cublasHandle_t handle)
{
    const float alpha = 1.0f, beta = 0.0f;
    for (int h = 0; h < p.H_local; h++) {
        float* Q_h = w.Q + (size_t)h * p.S * p.d_h;
        float* K_h = w.K + (size_t)h * p.S * p.d_h;
        float* V_h = w.V + (size_t)h * p.S * p.d_h;
        const float* W_q_h = w.W_q + (size_t)h * p.d * p.d_h;
        const float* W_k_h = w.W_k + (size_t)h * p.d * p.d_h;
        const float* W_v_h = w.W_v + (size_t)h * p.d * p.d_h;

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            p.d_h, p.S, p.d, &alpha, W_q_h, p.d_h, w.X, p.d, &beta, Q_h, p.d_h));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            p.d_h, p.S, p.d, &alpha, W_k_h, p.d_h, w.X, p.d, &beta, K_h, p.d_h));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            p.d_h, p.S, p.d, &alpha, W_v_h, p.d_h, w.X, p.d, &beta, V_h, p.d_h));
    }

    cudaStream_t stream;
    CUBLAS_CHECK(cublasGetStream(handle, &stream));
    const size_t qkv_n = (size_t)p.H_local * p.S * p.d_h;
    convert_f32_to_f16(w.Q, w.Qh, qkv_n, stream);
    convert_f32_to_f16(w.K, w.Kh, qkv_n, stream);
    convert_f32_to_f16(w.V, w.Vh, qkv_n, stream);
    CUDA_CHECK(cudaGetLastError());
}

ncclComm_t init_nccl(int rank, int world_size)
{
    ncclUniqueId id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
    return comm;
}

void free_weights(const Weights& w)
{
    CUDA_CHECK(cudaFree(w.X));
    CUDA_CHECK(cudaFree(w.W_q));
    CUDA_CHECK(cudaFree(w.W_k));
    CUDA_CHECK(cudaFree(w.W_v));
    CUDA_CHECK(cudaFree(w.W_O));
    CUDA_CHECK(cudaFree(w.Q));
    CUDA_CHECK(cudaFree(w.K));
    CUDA_CHECK(cudaFree(w.V));
    CUDA_CHECK(cudaFree(w.Qh));
    CUDA_CHECK(cudaFree(w.Kh));
    CUDA_CHECK(cudaFree(w.Vh));
}

RunCtx make_ctx(MPI_Comm comm, int rank, int world_size, int ncomm, bool prioritize)
{
    if (ncomm < 1 || ncomm > MAX_COMM) {
        if (rank == 0) fprintf(stderr, "ncomm must be in [1, %d]\n", MAX_COMM);
        MPI_Abort(comm, 1);
    }
    RunCtx c;
    int lo = 0, hi = 0;   // numerically lower = higher priority
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    const int p_side = prioritize ? hi : lo;

    CUDA_CHECK(cudaStreamCreateWithPriority(&c.compute_stream, cudaStreamNonBlocking, lo));
    CUDA_CHECK(cudaStreamCreateWithPriority(&c.proj_stream,    cudaStreamNonBlocking, p_side));
    c.ncomm = ncomm;
    for (int i = 0; i < ncomm; i++) {
        CUDA_CHECK(cudaStreamCreateWithPriority(&c.comm_streams[i], cudaStreamNonBlocking, p_side));
        c.nccls[i] = init_nccl(rank, world_size);
    }
    CUBLAS_CHECK(cublasCreate(&c.cublas));
    CUBLAS_CHECK(cublasSetStream(c.cublas, c.proj_stream));
    c.comm = comm;
    return c;
}

void destroy_ctx(RunCtx& c)
{
    CUBLAS_CHECK(cublasDestroy(c.cublas));
    CUDA_CHECK(cudaStreamDestroy(c.compute_stream));
    CUDA_CHECK(cudaStreamDestroy(c.proj_stream));
    for (int i = 0; i < c.ncomm; i++) {
        CUDA_CHECK(cudaStreamDestroy(c.comm_streams[i]));
        NCCL_CHECK(ncclCommDestroy(c.nccls[i]));
    }
}

FlashAttnTCPartials alloc_workspace(const AttnParams& p, int rows)
{
    FlashAttnTCPartials ws = { nullptr, nullptr, nullptr, rows };
    if (p.kernel == KERNEL_TC && p.split > 1) {
        CUDA_CHECK(cudaMalloc(&ws.O_part, (size_t)p.split * rows * p.d_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.m_part, (size_t)p.split * rows * p.H_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.l_part, (size_t)p.split * rows * p.H_local * sizeof(float)));
    }
    return ws;
}

void free_workspace(FlashAttnTCPartials& ws)
{
    CUDA_CHECK(cudaFree(ws.O_part));   // cudaFree(nullptr) is a no-op
    CUDA_CHECK(cudaFree(ws.m_part));
    CUDA_CHECK(cudaFree(ws.l_part));
    ws = { nullptr, nullptr, nullptr, 0 };
}

void launch_attention_raw(const Weights& w, float* O, const FlashAttnTCPartials& ws,
                          int q_start, int q_rows, const AttnParams& p, cudaStream_t stream,
                          int* tile_counters, int tile_rows)
{
    if (p.kernel == KERNEL_TC)
        flash_attn_tc_forward(w.Qh, w.Kh, w.Vh, O, ws, q_start, q_rows, p.flash_tc, stream,
                              tile_counters, tile_rows);
    else
        flash_attn_forward(w.Q, w.K, w.V, O, q_start, q_rows, p.flash, stream,
                           tile_counters, tile_rows);
}

void launch_combine(const FlashAttnTCPartials& ws, float* O, int row0, int rows,
                    const AttnParams& p, cudaStream_t stream)
{
    if (p.kernel == KERNEL_TC && p.split > 1)
        flash_attn_tc_combine(ws, O, row0, rows, p.flash_tc, stream);
}

void launch_attention(const Weights& w, float* O, const FlashAttnTCPartials& ws,
                      int q_start, int q_rows, const AttnParams& p, cudaStream_t stream)
{
    launch_attention_raw(w, O, ws, q_start, q_rows, p, stream);
    launch_combine(ws, O, 0, q_rows, p, stream);
}

// out[rows x d] = O_tile[rows x d_local] * W_O[d_local x d]  (row-major, via the column-major swap).
// TF32 is selected on the handle with cublasSetMathMode (attn_bench --proj tf32).
void project(const RunCtx& c, const Weights& w, const float* O_tile, float* out,
             int rows, const AttnParams& p)
{
    if (p.skip_proj) return;
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(c.cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        p.d, rows, p.d_local,
        &alpha, w.W_O, p.d,
                O_tile, p.d_local,
        &beta,  out, p.d));
}

int blocks_per_tile(const AttnParams& p)
{
    int rows_per_block = (p.kernel == KERNEL_TC) ? FLASH_TC_BM : p.B_M;
    int splits = (p.kernel == KERNEL_TC) ? p.split : 1;
    return (p.tile / rows_per_block) * p.H_local * splits;
}
