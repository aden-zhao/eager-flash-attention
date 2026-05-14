#include "attn_main.cuh"
#include "run_drivers.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static const int D_H = 64;
static const int H = 16;
static const int D_MODEL = H * D_H;
static const int N_WARMUP = 3;
static const int N_TRIALS = 5;

static int parse_int_arg(int argc, char** argv, const char* flag, int def)
{
    for (int i = 1; i + 1 < argc; i++)
        if (std::strcmp(argv[i], flag) == 0)
            return std::atoi(argv[i + 1]);
    return def;
}

static int derive_bn(int d_h)
{
    const int max_smem = 49152;
    int max_bn = max_smem / (2 * d_h * (int)sizeof(float));
    int bn = 1;
    while (bn * 2 <= max_bn) bn *= 2;
    return bn;
}

struct Stats {
    double min, max, mean, median, std;
};

static Stats compute_stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - mean) * (x - mean);
    Stats s;
    s.min = v.front();
    s.max = v.back();
    s.mean = mean;
    s.median = v[v.size() / 2];
    s.std = std::sqrt(sq_sum / v.size());
    return s;
}

static std::vector<double> time_mode(
    int mode,
    const Weights& w,
    const AttnParams& p,
    float* output,
    cudaStream_t compute_stream,
    cudaStream_t proj_stream,
    cublasHandle_t cublas_handle,
    MPI_Comm comm,
    ncclComm_t nccl_comm)
{
    std::vector<double> times;
    for (int trial = 0; trial < N_WARMUP + N_TRIALS; trial++) {
        RunResult r;
        if (mode == 0)
            r = run_megatron(w.Q, w.K, w.V, w.W_O, output, p, compute_stream, proj_stream, cublas_handle, comm, nccl_comm);
        else if (mode == 1)
            r = run_sync(w.Q, w.K, w.V, w.W_O, output, p, compute_stream, proj_stream, cublas_handle, comm, nccl_comm);
        else
            r = run_overlap(w.Q, w.K, w.V, w.W_O, output, p, compute_stream, proj_stream, cublas_handle, comm, nccl_comm);

        if (trial >= N_WARMUP)
            times.push_back(r.wall_ms);
    }
    return times;
}

static void print_row(int S, int D, int B_M, const char* mode, const Stats& s)
{
    printf("%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n",
           S, D, B_M, mode, s.min, s.max, s.mean, s.median, s.std);
}

int main(int argc, char** argv)
{
    int rank, world_size;
    init_mpi(&argc, &argv, rank, world_size);
    int device = init_cuda(rank);

    const int S = parse_int_arg(argc, argv, "--S", 4096);
    const int B_M = parse_int_arg(argc, argv, "--B_M", 32);
    const int B_N = parse_int_arg(argc, argv, "--B_N", derive_bn(D_H));

    AttnParams p = make_params(S, D_MODEL, H, B_M, B_N, world_size);

    if (H % world_size != 0) {
        if (rank == 0)
            fprintf(stderr, "ERROR: H=%d not divisible by D=%d\n", H, world_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!flash_attn_check_params(p.flash, device)) {
        if (rank == 0)
            fprintf(stderr, "ERROR: invalid params S=%d B_M=%d\n", S, B_M);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cudaStream_t compute_stream, proj_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&proj_stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    ncclComm_t nccl_comm = init_nccl(rank, world_size);

    Weights w = allocate_and_fill(p, rank, 42, compute_stream);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    compute_qkv(w, p, cublas_handle);
    CUDA_CHECK(cudaStreamSynchronize(proj_stream));

    float* output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, (size_t)p.S * p.d * sizeof(float)));

    std::vector<double> meg_times = time_mode(0, w, p, output, compute_stream, proj_stream, cublas_handle, MPI_COMM_WORLD, nccl_comm);
    std::vector<double> sync_times = time_mode(1, w, p, output, compute_stream, proj_stream, cublas_handle, MPI_COMM_WORLD, nccl_comm);
    std::vector<double> overlap_times = time_mode(2, w, p, output, compute_stream, proj_stream, cublas_handle, MPI_COMM_WORLD, nccl_comm);

    if (rank == 0) {
        printf("S,D,B_M,mode,min_ms,max_ms,mean_ms,median_ms,std_ms\n");
        print_row(S, world_size, B_M, "megatron", compute_stats(meg_times));
        print_row(S, world_size, B_M, "sync",     compute_stats(sync_times));
        print_row(S, world_size, B_M, "overlap",  compute_stats(overlap_times));
        fflush(stdout);
    }

    free_weights(w);
    CUDA_CHECK(cudaFree(output));
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(proj_stream));
    MPI_CHECK(MPI_Finalize());
    return 0;
}
