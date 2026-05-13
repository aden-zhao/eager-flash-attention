#include "attn_main.cuh"
#include "run_drivers.cuh"
#include "baseline_attention.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static float max_abs_diff(const float* a, const float* b, size_t n)
{
    float worst = 0.f;
    for (size_t i = 0; i < n; i++)
        worst = std::max(worst, std::fabsf(a[i] - b[i]));
    return worst;
}

static bool check(const char* label, const float* a, const float* b, size_t n, float atol)
{
    float diff = max_abs_diff(a, b, n);
    bool pass = diff <= atol;
    printf("  %-30s  max_diff=%.2e  %s\n", label, diff, pass ? "PASS" : "FAIL");
    return pass;
}

struct TestCase { int S, d, H, B_M, B_N; };

int main(int argc, char** argv)
{
    int rank, world_size;
    init_mpi(&argc, &argv, rank, world_size);
    int device = init_cuda(rank);

    cudaStream_t compute_stream, proj_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&proj_stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    // H divisible by 4 so cases run on 1, 2, or 4 GPUs without change.
    // B_M <= 32 (thread limit: B_M * 32 <= 1024).
    // S % B_M == 0 and S % B_N == 0.
    const TestCase cases[] = {
        {  128,  64, 4, 16, 16 },  // tiny, baseline smoke test
        {  256, 128, 8, 32, 32 },  // standard config
        {  512, 128, 8, 32, 64 },  // B_N > B_M: larger KV tiles
        {  512, 256, 8, 16, 32 },  // small B_M: many tiles (32 tiles)
        { 1024, 256, 8, 32, 64 },  // larger S
    };
    const int seeds[] = { 42, 123, 777 };

    int total_pass = 0, total_fail = 0;

    for (const auto& tc : cases) {
        AttnParams p = make_params(tc.S, tc.d, tc.H, tc.B_M, tc.B_N, world_size);

        if (!flash_attn_check_params(p.flash, device)) {
            if (rank == 0)
                printf("SKIP S=%d d=%d H=%d B_M=%d: invalid params\n",
                       tc.S, tc.d, tc.H, tc.B_M);
            continue;
        }

        for (int seed : seeds) {
            if (rank == 0)
                printf("\nS=%d d=%d H=%d B_M=%d B_N=%d seed=%d R=%d\n",
                       tc.S, tc.d, tc.H, tc.B_M, tc.B_N, seed, world_size);

            Weights w = allocate_and_fill(p, rank, seed, compute_stream);
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            compute_qkv(w, p, cublas_handle);
            CUDA_CHECK(cudaStreamSynchronize(proj_stream));

            const size_t out_bytes = (size_t)p.S * p.d * sizeof(float);
            float *out_meg, *out_sync, *out_overlap, *out_base;
            CUDA_CHECK(cudaMalloc(&out_meg,     out_bytes));
            CUDA_CHECK(cudaMalloc(&out_sync,    out_bytes));
            CUDA_CHECK(cudaMalloc(&out_overlap, out_bytes));
            CUDA_CHECK(cudaMalloc(&out_base,    out_bytes));

            // --- three pipeline modes ---
            run_megatron(w.Q, w.K, w.V, w.W_O, out_meg,     p, compute_stream, proj_stream, cublas_handle, MPI_COMM_WORLD);
            run_sync    (w.Q, w.K, w.V, w.W_O, out_sync,    p, compute_stream, proj_stream, cublas_handle, MPI_COMM_WORLD);
            run_overlap (w.Q, w.K, w.V, w.W_O, out_overlap, p, compute_stream, proj_stream, cublas_handle, MPI_COMM_WORLD);

            // --- baseline: per-rank attention + project + allreduce ---
            {
                float* O_attn  = nullptr;
                float* P_local = nullptr;
                float* workspace = nullptr;
                CUDA_CHECK(cudaMalloc(&O_attn,    (size_t)p.S * p.d_local * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&P_local,   out_bytes));
                CUDA_CHECK(cudaMalloc(&workspace, baseline_attention_workspace_bytes(p.S)));

                baseline_attention(w.Q, w.K, w.V, O_attn, workspace,
                                   p.H_local, p.S, p.d_h, compute_stream, cublas_handle);
                CUDA_CHECK(cudaStreamSynchronize(compute_stream));

                const float alpha = 1.f, beta = 0.f;
                CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));
                CUBLAS_CHECK(cublasSgemm(cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    p.d, p.S, p.d_local,
                    &alpha, w.W_O,   p.d,
                            O_attn,  p.d_local,
                    &beta,  P_local, p.d));
                CUDA_CHECK(cudaStreamSynchronize(proj_stream));

                MPI_CHECK(MPI_Allreduce(P_local, out_base,
                                        p.S * p.d, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

                CUDA_CHECK(cudaFree(O_attn));
                CUDA_CHECK(cudaFree(P_local));
                CUDA_CHECK(cudaFree(workspace));
            }

            // --- compare on rank 0 ---
            if (rank == 0) {
                const size_t n = (size_t)p.S * p.d;
                std::vector<float> h_meg(n), h_sync(n), h_overlap(n), h_base(n);
                CUDA_CHECK(cudaMemcpy(h_meg.data(),     out_meg,     out_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_sync.data(),    out_sync,    out_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_overlap.data(), out_overlap, out_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_base.data(),    out_base,    out_bytes, cudaMemcpyDeviceToHost));

                const float atol = 1e-3f;
                bool ok = true;
                ok &= check("megatron vs baseline",  h_meg.data(),     h_base.data(),    n, atol);
                ok &= check("sync vs baseline",      h_sync.data(),    h_base.data(),    n, atol);
                ok &= check("overlap vs baseline",   h_overlap.data(), h_base.data(),    n, atol);
                ok &= check("sync vs megatron",      h_sync.data(),    h_meg.data(),     n, atol);
                ok &= check("overlap vs megatron",   h_overlap.data(), h_meg.data(),     n, atol);

                if (ok) total_pass++; else total_fail++;
            }

            CUDA_CHECK(cudaFree(out_meg));
            CUDA_CHECK(cudaFree(out_sync));
            CUDA_CHECK(cudaFree(out_overlap));
            CUDA_CHECK(cudaFree(out_base));
            free_weights(w);
        }
    }

    if (rank == 0)
        printf("\n%d passed, %d failed\n", total_pass, total_fail);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(proj_stream));
    MPI_CHECK(MPI_Finalize());
    return total_fail > 0 ? 1 : 0;
}
