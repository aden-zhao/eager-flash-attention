// Benchmark driver. Prints one CSV row per mode (rank 0 only):
//   S,H,d_h,causal,R,kernel,tile,nbuf,split,ncomm,reserve,gctx,proj,mode,min_ms,max_ms,mean_ms,median_ms,std_ms
#include "attn_main.cuh"
#include "run_drivers.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static const char* arg_str(int argc, char** argv, const char* flag, const char* def)
{
    for (int i = 1; i + 1 < argc; i++)
        if (std::strcmp(argv[i], flag) == 0)
            return argv[i + 1];
    return def;
}

static int arg_int(int argc, char** argv, const char* flag, int def)
{
    const char* s = arg_str(argc, argv, flag, nullptr);
    return s ? std::atoi(s) : def;
}

static bool has_flag(int argc, char** argv, const char* flag)
{
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

struct Stats { double min, max, mean, median, std; };

static Stats compute_stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return { v.front(), v.back(), mean, v[v.size() / 2], std::sqrt(sq / v.size()) };
}

static void usage(const char* prog)
{
    fprintf(stderr,
        "usage: %s [--S 4096] [--H 16] [--d_h 64] [--tile 256] [--kernel tc|v1]\n"
        "          [--nbuf 2] [--split 1] [--ncomm 1] [--reserve 0] [--B_M 32] [--B_N 64]\n"
        "          [--warmup 3] [--trials 10]\n"
        "          [--modes attn,attn_tiled,megatron,sync,overlap,stream,comm] [--no-header]\n"
        "          [--causal] [--proj fp32|tf32|bf16] (projection GEMM precision; --tf32 = --proj tf32)\n"
        "          [--no_prio] (equal stream priorities)\n"
        "          [--gctx_sms N] [--gctx_proj]  (green-context SM partition for comm [and proj]; experimental)\n"
        "          [--no_proj] [--no_comm]   (diagnostics: drop a pipeline stage)\n", prog);
}

int main(int argc, char** argv)
{
    int rank, world_size;
    init_mpi(&argc, &argv, rank, world_size);
    int device = init_cuda(rank);

    if (has_flag(argc, argv, "--help")) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    const int S       = arg_int(argc, argv, "--S", 4096);
    const int H       = arg_int(argc, argv, "--H", 16);
    const int d_h     = arg_int(argc, argv, "--d_h", 64);
    const int tile    = arg_int(argc, argv, "--tile", 256);
    const int B_M     = arg_int(argc, argv, "--B_M", 32);
    const int B_N     = arg_int(argc, argv, "--B_N", 64);
    const int nbuf    = arg_int(argc, argv, "--nbuf", 2);
    const int split   = arg_int(argc, argv, "--split", 1);
    const int ncomm   = arg_int(argc, argv, "--ncomm", 1);
    const int reserve = arg_int(argc, argv, "--reserve", 0);   // max attention blocks per SM
    const int causal  = has_flag(argc, argv, "--causal") ? 1 : 0;
    const int proj    = has_flag(argc, argv, "--tf32") ? PROJ_TF32
                                                       : parse_proj(arg_str(argc, argv, "--proj", "fp32"));
    const int warmup  = arg_int(argc, argv, "--warmup", 3);
    const int trials  = arg_int(argc, argv, "--trials", 10);
    const int kernel  = parse_kernel(arg_str(argc, argv, "--kernel", "tc"));
    const bool header = !has_flag(argc, argv, "--no-header");
    const int d = H * d_h;

    std::vector<int> modes;
    {
        std::string list = arg_str(argc, argv, "--modes", "attn,megatron,sync,overlap,stream");
        size_t start = 0;
        while (start <= list.size()) {
            size_t end = list.find(',', start);
            if (end == std::string::npos) end = list.size();
            std::string tok = list.substr(start, end - start);
            if (!tok.empty()) {
                int m = parse_mode(tok.c_str());
                if (m < 0) {
                    if (rank == 0) fprintf(stderr, "ERROR: unknown mode '%s'\n", tok.c_str());
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                modes.push_back(m);
            }
            start = end + 1;
        }
    }

    if (kernel < 0) {
        if (rank == 0) fprintf(stderr, "ERROR: --kernel must be v1 or tc\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (proj < 0) {
        if (rank == 0) fprintf(stderr, "ERROR: --proj must be fp32, tf32 or bf16\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    AttnParams p = make_params(S, d, H, tile, B_M, B_N, kernel, nbuf, split, causal, proj, world_size);
    p.flash_tc.max_blocks_per_sm = reserve;
    p.skip_proj = has_flag(argc, argv, "--no_proj");
    p.skip_comm = has_flag(argc, argv, "--no_comm");
    if (const char* why = attn_check_params(p, device)) {
        if (rank == 0) fprintf(stderr, "ERROR: invalid params: %s\n", why);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int gctx_sms = arg_int(argc, argv, "--gctx_sms", 0);
    RunCtx ctx = make_ctx(MPI_COMM_WORLD, rank, world_size, ncomm, !has_flag(argc, argv, "--no_prio"),
                          gctx_sms, has_flag(argc, argv, "--gctx_proj"));
    if (proj == PROJ_TF32) CUBLAS_CHECK(cublasSetMathMode(ctx.cublas, CUBLAS_TF32_TENSOR_OP_MATH));

    Weights w = allocate_and_fill(p, rank, 42, ctx.compute_stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
    compute_qkv(w, p, ctx.cublas);
    CUDA_CHECK(cudaStreamSynchronize(ctx.proj_stream));

    float* output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, (size_t)p.S * p.d * sizeof(float)));

    if (rank == 0 && header)
        printf("S,H,d_h,causal,R,kernel,tile,nbuf,split,ncomm,reserve,gctx,proj,mode,min_ms,max_ms,mean_ms,median_ms,std_ms\n");

    for (int mode : modes) {
        std::vector<double> times;
        for (int t = 0; t < warmup + trials; t++) {
            RunResult r = run_mode(mode, w, output, p, ctx);
            if (t >= warmup) times.push_back(r.wall_ms);
        }
        if (rank == 0) {
            Stats s = compute_stats(times);
            printf("%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   S, H, d_h, causal, world_size, kernel_name(kernel), tile, nbuf, split, ncomm, reserve,
                   ctx.gctx_sms_comm, proj_name(proj), mode_name(mode),
                   s.min, s.max, s.mean, s.median, s.std);
            fflush(stdout);
        }
    }

    free_weights(w);
    CUDA_CHECK(cudaFree(output));
    destroy_ctx(ctx);
    MPI_CHECK(MPI_Finalize());
    return 0;
}
