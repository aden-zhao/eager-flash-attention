#include "tile_comm.cuh"
#include "flash_attn.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

struct AttnParams {
    int S, d, H, R;
    int H_local, d_h, d_local, num_tiles;
    int B_M, B_N;
    int num_buffers, max_inflight;
    Mode mode;
    FlashAttnParams flash;
};

struct Weights {
    float* X;
    float* W_q, *W_k, *W_v, *W_O;
    float* Q, *K, *V;
};

static void init_mpi(int* argc, char*** argv, int& rank, int& world_size)
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

static int init_cuda(int rank)
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

static AttnParams parse_params(int argc, char** argv, int rank, int world_size)
{
    AttnParams p;
    p.S   = parse_int(argc, argv, "--S",   4096);
    p.d   = parse_int(argc, argv, "--d",   1024);
    p.H   = parse_int(argc, argv, "--H",   16);
    p.B_M = parse_int(argc, argv, "--B_M", 32);
    p.B_N = parse_int(argc, argv, "--B_N", 32);
    p.num_buffers  = parse_int(argc, argv, "--num_buffers",  2);
    p.max_inflight = parse_int(argc, argv, "--max_inflight", 1);
    p.mode = parse_mode(argc, argv);

    p.R       = world_size;
    p.H_local = p.H / p.R;
    p.d_h     = p.d / p.H;
    p.d_local = p.d / p.R;
    p.num_tiles = p.S / p.B_M;

    if (p.H % p.R != 0) {
        if (rank == 0) std::cerr << "ERROR: H must be divisible by R.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (p.d % p.H != 0) {
        if (rank == 0) std::cerr << "ERROR: d must be divisible by H.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    p.flash = { p.S, p.d_h, p.H_local, p.B_M, p.B_N, 1.0f / sqrtf((float)p.d_h) };
    return p;
}

static void print_params(const AttnParams& p)
{
    std::cout << "attn_main\n"
              << "  mode      = " << mode_str(p.mode) << "\n"
              << "  S         = " << p.S        << "\n"
              << "  d         = " << p.d        << "\n"
              << "  H         = " << p.H        << "\n"
              << "  R         = " << p.R        << "\n"
              << "  H_local   = " << p.H_local  << "\n"
              << "  d_h       = " << p.d_h      << "\n"
              << "  d_local   = " << p.d_local  << "\n"
              << "  B_M       = " << p.B_M      << "\n"
              << "  B_N       = " << p.B_N      << "\n"
              << "  num_tiles = " << p.num_tiles << "\n";
}

static void rand_fill(float* d_ptr, size_t n, float scale, cudaStream_t stream)
{
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++)
        h[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, h.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
}

static Weights allocate_and_fill(const AttnParams& p, int rank, cudaStream_t stream)
{
    Weights w;
    CUDA_CHECK(cudaMalloc(&w.X,   (size_t)p.S * p.d               * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_q, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_k, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_v, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_O, (size_t)p.d_local * p.d          * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.Q,   (size_t)p.H_local * p.S * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.K,   (size_t)p.H_local * p.S * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.V,   (size_t)p.H_local * p.S * p.d_h * sizeof(float)));

    srand(42 + rank);
    rand_fill(w.X,   (size_t)p.S * p.d,               0.1f,  stream);
    rand_fill(w.W_q, (size_t)p.H_local * p.d * p.d_h, 0.02f, stream);
    rand_fill(w.W_k, (size_t)p.H_local * p.d * p.d_h, 0.02f, stream);
    rand_fill(w.W_v, (size_t)p.H_local * p.d * p.d_h, 0.02f, stream);
    rand_fill(w.W_O, (size_t)p.d_local * p.d,          0.02f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return w;
}

static void compute_qkv(const Weights& w, const AttnParams& p, cublasHandle_t handle)
{
    const float alpha = 1.0f, beta = 0.0f;
    for (int h = 0; h < p.H_local; h++) {
        float* Q_h       = w.Q   + (size_t)h * p.S * p.d_h;
        float* K_h       = w.K   + (size_t)h * p.S * p.d_h;
        float* V_h       = w.V   + (size_t)h * p.S * p.d_h;
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
}

static void free_weights(const Weights& w)
{
    CUDA_CHECK(cudaFree(w.X));
    CUDA_CHECK(cudaFree(w.W_q));
    CUDA_CHECK(cudaFree(w.W_k));
    CUDA_CHECK(cudaFree(w.W_v));
    CUDA_CHECK(cudaFree(w.W_O));
    CUDA_CHECK(cudaFree(w.Q));
    CUDA_CHECK(cudaFree(w.K));
    CUDA_CHECK(cudaFree(w.V));
}

int main(int argc, char** argv)
{
    int rank, world_size;
    init_mpi(&argc, &argv, rank, world_size);
    int device = init_cuda(rank);

    AttnParams p = parse_params(argc, argv, rank, world_size);

    if (!flash_attn_check_params(p.flash, device)) {
        if (rank == 0) std::cerr << "ERROR: invalid flash attention params.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) print_params(p);

    cudaStream_t compute_stream, proj_stream, comm_cuda_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&proj_stream));
    CUDA_CHECK(cudaStreamCreate(&comm_cuda_stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    Weights w = allocate_and_fill(p, rank, compute_stream);
    compute_qkv(w, p, cublas_handle);

    float* output_device = nullptr;
    CUDA_CHECK(cudaMalloc(&output_device, (size_t)p.S * p.d * sizeof(float)));
    CUDA_CHECK(cudaMemset(output_device, 0, (size_t)p.S * p.d * sizeof(float)));

    // --- main pipeline loop goes here ---

    free_weights(w);
    CUDA_CHECK(cudaFree(output_device));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(proj_stream));
    CUDA_CHECK(cudaStreamDestroy(comm_cuda_stream));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
