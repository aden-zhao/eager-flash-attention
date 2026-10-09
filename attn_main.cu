#include "attn_main.cuh"

#include <cuda_bf16.h>
#include <algorithm>
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
                       int kernel, int nbuf, int split, int causal, int proj, int world_size)
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
    p.proj = proj;
    p.skip_proj = 0;
    p.skip_comm = 0;
    float scale = 1.0f / sqrtf((float)p.d_h);
    p.flash    = { S, p.d_h, p.H_local, B_M, B_N, causal, scale };
    p.flash_tc = { S, p.d_h, p.H_local, split, causal, proj == PROJ_BF16 ? 1 : 0, 0, scale };
    return p;
}

const char* attn_check_params(const AttnParams& p, int cuda_device)
{
    if (p.H % p.R != 0) return "H must be divisible by the number of ranks";
    if (p.d % p.H != 0) return "d must be divisible by H";
    if (p.tile <= 0 || p.S % p.tile != 0) return "S must be a multiple of tile";
    if (p.nbuf < 1) return "nbuf must be >= 1";
    if (p.proj < PROJ_FP32 || p.proj > PROJ_BF16) return "unknown projection precision";
    if (p.kernel == KERNEL_TC) return flash_attn_tc_check_params(p.flash_tc, p.tile);
    if (p.kernel == KERNEL_V1) {
        if (p.split != 1) return "v1: split-KV not supported";
        if (p.proj == PROJ_BF16) return "v1: bf16 output not supported";
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

const char* proj_name(int proj)
{
    switch (proj) {
        case PROJ_FP32: return "fp32";
        case PROJ_TF32: return "tf32";
        case PROJ_BF16: return "bf16";
    }
    return "?";
}

int parse_proj(const char* s)
{
    if (std::strcmp(s, "fp32") == 0) return PROJ_FP32;
    if (std::strcmp(s, "tf32") == 0) return PROJ_TF32;
    if (std::strcmp(s, "bf16") == 0) return PROJ_BF16;
    return -1;
}

size_t o_elem_bytes(const AttnParams& p)
{
    return p.proj == PROJ_BF16 ? sizeof(__nv_bfloat16) : sizeof(float);
}

static __global__ void f32_to_bf16_kernel(const float* __restrict__ in, __nv_bfloat16* __restrict__ out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        out[i] = __float2bfloat16(in[i]);
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
    CUDA_CHECK(cudaMalloc(&w.W_Ob, (size_t)p.d_local * p.d * sizeof(__nv_bfloat16)));

    // Scales chosen so attention logits have O(1) spread: a flat softmax would
    // hide kernel bugs behind a uniform average.
    srand(seed);
    rand_fill(w.X, (size_t)p.S * p.d, 1.0f, stream);

    srand(seed * 31 + rank + 1);
    rand_fill(w.W_q, (size_t)p.H_local * p.d * p.d_h, 0.15f, stream);
    rand_fill(w.W_k, (size_t)p.H_local * p.d * p.d_h, 0.15f, stream);
    rand_fill(w.W_v, (size_t)p.H_local * p.d * p.d_h, 0.10f, stream);
    rand_fill(w.W_O, (size_t)p.d_local * p.d, 0.05f, stream);
    {
        const size_t n = (size_t)p.d_local * p.d;
        int blocks = (int)std::min<size_t>((n + 255) / 256, 4096);
        f32_to_bf16_kernel<<<blocks, 256, 0, stream>>>(w.W_O, w.W_Ob, n);
        CUDA_CHECK(cudaGetLastError());
    }
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
    CUDA_CHECK(cudaFree(w.W_Ob));
}

RunCtx make_ctx(MPI_Comm comm, int rank, int world_size, int ncomm, bool prioritize,
                int gctx_sms, bool gctx_proj)
{
    if (ncomm < 1 || ncomm > MAX_COMM) {
        if (rank == 0) fprintf(stderr, "ncomm must be in [1, %d]\n", MAX_COMM);
        MPI_Abort(comm, 1);
    }
    RunCtx c;
    c.gctx_compute = nullptr;
    c.gctx_comm = nullptr;
    c.gctx_sms_comm = c.gctx_sms_compute = 0;
    int lo = 0, hi = 0;   // numerically lower = higher priority
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    const int p_side = prioritize ? hi : lo;

    if (gctx_sms > 0) {
        // Split the device's SMs into a communication partition of >= gctx_sms
        // SMs (rounded up to the hardware's co-scheduling granularity) and a
        // compute partition holding the rest. Streams created in a green
        // context only run on its SMs; memory is shared with the primary context.
        int dev_ordinal = 0;
        CUDA_CHECK(cudaGetDevice(&dev_ordinal));
        CUDA_CHECK(cudaFree(nullptr));   // make sure the primary context exists
        CUdevice dev;
        CU_CHECK(cuDeviceGet(&dev, dev_ordinal));
        CUdevResource all_sms, comm_res, rest_res;
        CU_CHECK(cuDeviceGetDevResource(dev, &all_sms, CU_DEV_RESOURCE_TYPE_SM));
        unsigned int ngroups = 1;
        CU_CHECK(cuDevSmResourceSplitByCount(&comm_res, &ngroups, &all_sms, &rest_res,
                                             0, (unsigned int)gctx_sms));
        CUdevResourceDesc comm_desc, rest_desc;
        CU_CHECK(cuDevResourceGenerateDesc(&comm_desc, &comm_res, 1));
        CU_CHECK(cuDevResourceGenerateDesc(&rest_desc, &rest_res, 1));
        CU_CHECK(cuGreenCtxCreate(&c.gctx_comm,    comm_desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        CU_CHECK(cuGreenCtxCreate(&c.gctx_compute, rest_desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        c.gctx_sms_comm = (int)comm_res.sm.smCount;
        c.gctx_sms_compute = (int)rest_res.sm.smCount;
        if (rank == 0)
            fprintf(stderr, "green contexts: %d SMs for communication%s, %d SMs for attention\n",
                    c.gctx_sms_comm, gctx_proj ? " + projection" : "", c.gctx_sms_compute);

        CU_CHECK(cuGreenCtxStreamCreate((CUstream*)&c.compute_stream, c.gctx_compute,
                                        CU_STREAM_NON_BLOCKING, lo));
        CU_CHECK(cuGreenCtxStreamCreate((CUstream*)&c.proj_stream,
                                        gctx_proj ? c.gctx_comm : c.gctx_compute,
                                        CU_STREAM_NON_BLOCKING, p_side));
        c.ncomm = ncomm;
        for (int i = 0; i < ncomm; i++) {
            CU_CHECK(cuGreenCtxStreamCreate((CUstream*)&c.comm_streams[i], c.gctx_comm,
                                            CU_STREAM_NON_BLOCKING, p_side));
            c.nccls[i] = init_nccl(rank, world_size);
        }
    } else {
        CUDA_CHECK(cudaStreamCreateWithPriority(&c.compute_stream, cudaStreamNonBlocking, lo));
        CUDA_CHECK(cudaStreamCreateWithPriority(&c.proj_stream,    cudaStreamNonBlocking, p_side));
        c.ncomm = ncomm;
        for (int i = 0; i < ncomm; i++) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&c.comm_streams[i], cudaStreamNonBlocking, p_side));
            c.nccls[i] = init_nccl(rank, world_size);
        }
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
    if (c.gctx_comm)    CU_CHECK(cuGreenCtxDestroy(c.gctx_comm));
    if (c.gctx_compute) CU_CHECK(cuGreenCtxDestroy(c.gctx_compute));
}

AttnWorkspace alloc_workspace(const AttnParams& p, int rows)
{
    AttnWorkspace ws;
    ws.part = { nullptr, nullptr, nullptr, rows };
    if (p.kernel == KERNEL_TC && p.split > 1) {
        CUDA_CHECK(cudaMalloc(&ws.part.O_part, (size_t)p.split * rows * p.d_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.part.m_part, (size_t)p.split * rows * p.H_local * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws.part.l_part, (size_t)p.split * rows * p.H_local * sizeof(float)));
    }
    return ws;
}

void free_workspace(AttnWorkspace& ws)
{
    if (ws.part.O_part) CUDA_CHECK(cudaFree(ws.part.O_part));
    if (ws.part.m_part) CUDA_CHECK(cudaFree(ws.part.m_part));
    if (ws.part.l_part) CUDA_CHECK(cudaFree(ws.part.l_part));
    ws.part = { nullptr, nullptr, nullptr, 0 };
}

void launch_attention_raw(const Weights& w, void* O, const AttnWorkspace& ws,
                          int q_start, int q_rows, const AttnParams& p, cudaStream_t stream,
                          int* tile_counters, int tile_rows)
{
    if (p.kernel == KERNEL_TC)
        flash_attn_tc_forward(w.Qh, w.Kh, w.Vh, O, ws.part, q_start, q_rows, p.flash_tc, stream,
                              tile_counters, tile_rows);
    else
        flash_attn_forward(w.Q, w.K, w.V, static_cast<float*>(O), q_start, q_rows, p.flash, stream,
                           tile_counters, tile_rows);
}

void launch_combine(const AttnWorkspace& ws, void* O, int row0, int rows,
                    const AttnParams& p, cudaStream_t stream)
{
    if (p.kernel == KERNEL_TC && p.split > 1)
        flash_attn_tc_combine(ws.part, O, row0, rows, p.flash_tc, stream);
}

void launch_attention(const Weights& w, void* O, const AttnWorkspace& ws,
                      int q_start, int q_rows, const AttnParams& p, cudaStream_t stream,
                      int* tile_counters, int tile_rows)
{
    launch_attention_raw(w, O, ws, q_start, q_rows, p, stream, tile_counters, tile_rows);
    launch_combine(ws, O, 0, q_rows, p, stream);
}

// out[rows x d] = O_tile[rows x d_local] * W_O[d_local x d]  (row-major, via the column-major swap)
void project(const RunCtx& c, const Weights& w, const void* O_tile, float* out,
             int rows, const AttnParams& p)
{
    if (p.skip_proj) return;
    const float alpha = 1.f, beta = 0.f;
    if (p.proj == PROJ_BF16) {
        CUBLAS_CHECK(cublasGemmEx(c.cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.d, rows, p.d_local,
            &alpha, w.W_Ob,  CUDA_R_16BF, p.d,
                    O_tile,  CUDA_R_16BF, p.d_local,
            &beta,  out,     CUDA_R_32F,  p.d,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // TF32 is selected on the handle via cublasSetMathMode (make_ctx / bench flag).
        CUBLAS_CHECK(cublasSgemm(c.cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.d, rows, p.d_local,
            &alpha, w.W_O, p.d,
                    static_cast<const float*>(O_tile), p.d_local,
            &beta,  out, p.d));
    }
}

int blocks_per_tile(const AttnParams& p)
{
    int rows_per_block = (p.kernel == KERNEL_TC) ? FLASH_TC_BM : p.B_M;
    int splits = (p.kernel == KERNEL_TC) ? p.split : 1;
    return (p.tile / rows_per_block) * p.H_local * splits;
}
