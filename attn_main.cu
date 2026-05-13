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

AttnParams make_params(int S, int d, int H, int B_M, int B_N, int world_size)
{
    AttnParams p;
    p.S = S;
    p.d = d;
    p.H = H;
    p.B_M = B_M;
    p.B_N = B_N;
    p.R = world_size;
    p.H_local = H / world_size;
    p.d_h = d / H;
    p.d_local = d / world_size;
    p.num_tiles = S / B_M;
    p.flash = { S, p.d_h, p.H_local, B_M, B_N, 1.0f / sqrtf((float)p.d_h) };
    return p;
}

static void rand_fill(float* d_ptr, size_t n, float scale, cudaStream_t stream)
{
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++)
        h[i] = scale * (2.0f * (float)rand() / RAND_MAX - 1.0f);
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, h.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
}

Weights allocate_and_fill(const AttnParams& p, int rank, int seed, cudaStream_t stream)
{
    Weights w;
    CUDA_CHECK(cudaMalloc(&w.X, (size_t)p.S * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_q, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_k, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_v, (size_t)p.H_local * p.d * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.W_O, (size_t)p.d_local * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.Q, (size_t)p.H_local * p.S * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.K, (size_t)p.H_local * p.S * p.d_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w.V, (size_t)p.H_local * p.S * p.d_h * sizeof(float)));

    srand(seed);
    rand_fill(w.X, (size_t)p.S * p.d, 0.1f, stream);

    srand(seed * 31 + rank + 1);
    rand_fill(w.W_q, (size_t)p.H_local * p.d * p.d_h, 0.02f, stream);
    rand_fill(w.W_k, (size_t)p.H_local * p.d * p.d_h, 0.02f, stream);
    rand_fill(w.W_v, (size_t)p.H_local * p.d * p.d_h, 0.02f, stream);
    rand_fill(w.W_O, (size_t)p.d_local * p.d, 0.02f, stream);
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
}
