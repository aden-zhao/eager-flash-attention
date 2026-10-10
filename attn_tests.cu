// Correctness: every (kernel, mode) pair against a cuBLAS/naive-softmax
// reference with the same weights, judged on relative error.
#include "attn_main.cuh"
#include "run_drivers.cuh"
#include "baseline_attention.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct Diff { float max_abs; float max_ref; };

static Diff compare(const float* a, const float* ref, size_t n)
{
    Diff d = { 0.f, 0.f };
    for (size_t i = 0; i < n; i++) {
        d.max_abs = std::max(d.max_abs, std::fabs(a[i] - ref[i]));
        d.max_ref = std::max(d.max_ref, std::fabs(ref[i]));
    }
    return d;
}

struct TestCase { int S, d, H, tile, B_M, B_N, nbuf, split, causal; };

// Reference: per-head materialized softmax(QK^T)V (fp32), projection, MPI allreduce.
static void reference(const Weights& w, const AttnParams& p, const RunCtx& c, float* out_base)
{
    const size_t out_bytes = (size_t)p.S * p.d * sizeof(float);
    float* O_attn = nullptr;
    float* P_local = nullptr;
    float* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&O_attn, (size_t)p.S * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&P_local, out_bytes));
    CUDA_CHECK(cudaMalloc(&workspace, baseline_attention_workspace_bytes(p.S)));

    baseline_attention(w.Q, w.K, w.V, O_attn, workspace,
                       p.H_local, p.S, p.d_h, p.causal, c.compute_stream, c.cublas);
    CUDA_CHECK(cudaStreamSynchronize(c.compute_stream));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(c.cublas, c.proj_stream));
    CUBLAS_CHECK(cublasSgemm(c.cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        p.d, p.S, p.d_local,
        &alpha, w.W_O, p.d,
                O_attn, p.d_local,
        &beta, P_local, p.d));
    CUDA_CHECK(cudaStreamSynchronize(c.proj_stream));

    // Reference reduction: one plain NCCL AllReduce on communicator 0, fully synchronized.
    CUDA_CHECK(cudaDeviceSynchronize());
    NCCL_CHECK(ncclAllReduce(P_local, out_base, (size_t)p.S * p.d, ncclFloat, ncclSum,
                             c.nccls[0], c.comm_streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(c.comm_streams[0]));

    CUDA_CHECK(cudaFree(O_attn));
    CUDA_CHECK(cudaFree(P_local));
    CUDA_CHECK(cudaFree(workspace));
}

int main(int argc, char** argv)
{
    int rank, world_size;
    init_mpi(&argc, &argv, rank, world_size);
    int device = init_cuda(rank);
    RunCtx ctx = make_ctx(MPI_COMM_WORLD, rank, world_size, /*ncomm=*/3);

    //                 S     d    H  tile B_M B_N nbuf split causal
    const TestCase cases[] = {
        {  128,   64,  4,   64, 16, 32, 2, 1, 0 },   // d_h=16: tc only
        {  256,  128,  4,   64, 32, 32, 2, 1, 0 },   // d_h=32
        {  512,  256,  8,  128, 32, 64, 2, 1, 0 },   // d_h=32
        { 1024,  512,  8,  256, 16, 64, 2, 1, 0 },   // d_h=64
        { 2048, 1024,  8,  512, 32, 64, 3, 1, 0 },   // d_h=128
        { 1024, 1024,  8,   64, 32, 64, 3, 1, 0 },   // d_h=128, 16 tiles deep pipeline
        { 1024,  512,  8,  256, 16, 64, 2, 4, 0 },   // split-KV (tc only)
        { 2048,  512,  8,  128, 16, 64, 2, 8, 0 },   // split-KV, 16 tiles
        {  512,  256,  8,   64, 32, 64, 2, 2, 0 },   // split-KV, d_h=32
        { 1024,  512,  8,  256, 16, 64, 2, 1, 1 },   // causal
        { 2048,  512,  8,  128, 32, 64, 2, 8, 1 },   // causal + split-KV (some blocks idle)
        {  512,  256,  8,   64, 32, 32, 2, 1, 1 },   // causal, B_N=32 (v1 partial chunks)
        { 2048, 1024,  8,  512, 32, 64, 3, 4, 1 },   // causal, d_h=128
    };
    const int seeds[] = { 42, 123, 777 };
    const int kernels[] = { KERNEL_V1, KERNEL_TC };
    const int modes[] = { MODE_MEGATRON, MODE_SYNC, MODE_OVERLAP, MODE_STREAM };

    int total_pass = 0, total_fail = 0, total_skip = 0;

    for (const auto& tc : cases) {
        for (int kernel : kernels) {
            AttnParams p = make_params(tc.S, tc.d, tc.H, tc.tile, tc.B_M, tc.B_N,
                                       kernel, tc.nbuf, tc.split, tc.causal, world_size);
            const char* why = attn_check_params(p, device);
            if (why) {
                if (rank == 0)
                    printf("SKIP  kernel=%s S=%d d=%d H=%d tile=%d split=%d causal=%d: %s\n",
                           kernel_name(kernel), tc.S, tc.d, tc.H, tc.tile, tc.split, tc.causal, why);
                total_skip++;
                continue;
            }
            // fp16 inputs and fp16 P: ~1e-3 relative; fp32 fast-math: ~1e-5.
            const float rtol = (kernel == KERNEL_TC) ? 2e-2f : 1e-3f;

            for (int seed : seeds) {
                if (rank == 0)
                    printf("\nkernel=%s S=%d d=%d H=%d d_h=%d tile=%d B_M=%d B_N=%d nbuf=%d split=%d causal=%d seed=%d R=%d\n",
                           kernel_name(kernel), tc.S, tc.d, tc.H, p.d_h, tc.tile,
                           tc.B_M, tc.B_N, tc.nbuf, tc.split, tc.causal, seed, world_size);

                Weights w = allocate_and_fill(p, rank, seed, ctx.compute_stream);
                CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
                CUBLAS_CHECK(cublasSetStream(ctx.cublas, ctx.proj_stream));
                compute_qkv(w, p, ctx.cublas);
                CUDA_CHECK(cudaStreamSynchronize(ctx.proj_stream));

                const size_t n = (size_t)p.S * p.d;
                const size_t out_bytes = n * sizeof(float);
                float *out_base, *out_mode;
                CUDA_CHECK(cudaMalloc(&out_base, out_bytes));
                CUDA_CHECK(cudaMalloc(&out_mode, out_bytes));

                reference(w, p, ctx, out_base);
                CUBLAS_CHECK(cublasSetStream(ctx.cublas, ctx.proj_stream));
                std::vector<float> h_base(n), h_mode(n);
                CUDA_CHECK(cudaMemcpy(h_base.data(), out_base, out_bytes, cudaMemcpyDeviceToHost));

                for (int mode : modes) {
                    CUDA_CHECK(cudaMemset(out_mode, 0, out_bytes));
                    run_mode(mode, w, out_mode, p, ctx);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(h_mode.data(), out_mode, out_bytes, cudaMemcpyDeviceToHost));

                    Diff d = compare(h_mode.data(), h_base.data(), n);
                    float rel = d.max_abs / std::max(d.max_ref, 1e-20f);
                    int pass = (rel <= rtol) && std::isfinite(d.max_abs);
                    int all_pass = pass;
                    MPI_CHECK(MPI_Allreduce(&pass, &all_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
                    if (rank == 0) {
                        printf("  %-10s max_abs=%.3e  max_ref=%.3e  rel=%.3e  %s\n",
                               mode_name(mode), d.max_abs, d.max_ref, rel, all_pass ? "PASS" : "FAIL");
                        if (all_pass) total_pass++; else total_fail++;
                    }
                }

                CUDA_CHECK(cudaFree(out_base));
                CUDA_CHECK(cudaFree(out_mode));
                free_weights(w);
            }
        }
    }

    if (rank == 0)
        printf("\n%d passed, %d failed, %d skipped configurations\n", total_pass, total_fail, total_skip);

    destroy_ctx(ctx);
    MPI_CHECK(MPI_Finalize());
    return total_fail > 0 ? 1 : 0;
}
