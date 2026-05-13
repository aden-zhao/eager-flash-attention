#include "run_drivers.cuh"
#include "flash_attn.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t e = (call);                                         \
        if (e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(e));         \
            std::abort();                                               \
        }                                                               \
    } while (0)

#define CUBLAS_CHECK(call)                                              \
    do {                                                                \
        cublasStatus_t s = (call);                                      \
        if (s != CUBLAS_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                \
                    __FILE__, __LINE__, (int)s);                        \
            std::abort();                                               \
        }                                                               \
    } while (0)

#define MPI_CHECK(call)                                                 \
    do {                                                                \
        int e = (call);                                                 \
        if (e != MPI_SUCCESS) {                                         \
            fprintf(stderr, "MPI error %s:%d\n", __FILE__, __LINE__);  \
            std::abort();                                               \
        }                                                               \
    } while (0)

RunResult run_megatron(
    const float* Q,
    const float* K,
    const float* V,
    const float* W_O,
    float* output,
    const AttnParams& p,
    cudaStream_t compute_stream,
    cudaStream_t proj_stream,
    cublasHandle_t cublas_handle,
    MPI_Comm comm
)
{
    int rank;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));

    float* O_tiles  = nullptr;
    float* send_buf = nullptr;
    float* recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&O_tiles,  (size_t)p.S * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.S * p.d       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buf, (size_t)p.S * p.d       * sizeof(float)));

    MPI_CHECK(MPI_Barrier(comm));
    double t0 = MPI_Wtime();

    // All tiles launched sequentially on compute_stream.
    // CUDA stream ordering ensures tile i+1 waits for tile i.
    for (int tile = 0; tile < p.num_tiles; tile++)
        flash_attn_forward_tile(Q, K, V,
                                O_tiles + (size_t)tile * p.B_M * p.d_local,
                                tile, p.flash, compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // Project: send_buf[S, d] = O_tiles[S, d_local] * W_O[d_local, d]
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        p.d, p.S, p.d_local,
        &alpha, W_O,    p.d,
                O_tiles, p.d_local,
        &beta,  send_buf, p.d));
    CUDA_CHECK(cudaStreamSynchronize(proj_stream));

    // One blocking allreduce over the full projected output
    MPI_CHECK(MPI_Allreduce(send_buf, recv_buf,
                            p.S * p.d, MPI_FLOAT, MPI_SUM, comm));

    CUDA_CHECK(cudaMemcpy(output, recv_buf,
                          (size_t)p.S * p.d * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    double t1 = MPI_Wtime();

    CUDA_CHECK(cudaFree(O_tiles));
    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));

    RunResult result;
    result.wall_ms = (t1 - t0) * 1e3;
    return result;
}

RunResult run_sync(
    const float* Q,
    const float* K,
    const float* V,
    const float* W_O,
    float* output,
    const AttnParams& p,
    cudaStream_t compute_stream,
    cudaStream_t proj_stream,
    cublasHandle_t cublas_handle,
    MPI_Comm comm
)
{
    float* O_tile   = nullptr;
    float* send_buf = nullptr;
    float* recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&O_tile,   (size_t)p.B_M * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.B_M * p.d       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buf, (size_t)p.B_M * p.d       * sizeof(float)));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    MPI_CHECK(MPI_Barrier(comm));
    double t0 = MPI_Wtime();

    for (int tile = 0; tile < p.num_tiles; tile++) {
        flash_attn_forward_tile(Q, K, V, O_tile, tile, p.flash, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // send_buf[B_M, d] = O_tile[B_M, d_local] * W_O[d_local, d]
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.d, p.B_M, p.d_local,
            &alpha, W_O,    p.d,
                    O_tile, p.d_local,
            &beta,  send_buf, p.d));
        CUDA_CHECK(cudaStreamSynchronize(proj_stream));

        MPI_CHECK(MPI_Allreduce(send_buf, recv_buf,
                                p.B_M * p.d, MPI_FLOAT, MPI_SUM, comm));

        CUDA_CHECK(cudaMemcpy(output + (size_t)tile * p.B_M * p.d,
                              recv_buf,
                              (size_t)p.B_M * p.d * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    double t1 = MPI_Wtime();

    CUDA_CHECK(cudaFree(O_tile));
    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));

    RunResult result;
    result.wall_ms = (t1 - t0) * 1e3;
    return result;
}

RunResult run_overlap(
    const float* Q,
    const float* K,
    const float* V,
    const float* W_O,
    float* output,
    const AttnParams& p,
    cudaStream_t compute_stream,
    cudaStream_t proj_stream,
    cublasHandle_t cublas_handle,
    MPI_Comm comm
)
{
    float* O_tile      = nullptr;
    float* send_buf[2] = {nullptr, nullptr};
    float* recv_buf    = nullptr;
    CUDA_CHECK(cudaMalloc(&O_tile,      (size_t)p.B_M * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf[0], (size_t)p.B_M * p.d       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf[1], (size_t)p.B_M * p.d       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buf,    (size_t)p.B_M * p.d       * sizeof(float)));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    MPI_Request req = MPI_REQUEST_NULL;

    MPI_CHECK(MPI_Barrier(comm));
    double t0 = MPI_Wtime();

    for (int tile = 0; tile < p.num_tiles; tile++) {
        int buf = tile % 2;

        flash_attn_forward_tile(Q, K, V, O_tile, tile, p.flash, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // send_buf[buf][B_M, d] = O_tile[B_M, d_local] * W_O[d_local, d]
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.d, p.B_M, p.d_local,
            &alpha, W_O,        p.d,
                    O_tile,     p.d_local,
            &beta,  send_buf[buf], p.d));
        CUDA_CHECK(cudaStreamSynchronize(proj_stream));

        // Wait for the previous tile's allreduce before reusing recv_buf
        // and before overwriting send_buf[buf] in the next cycle.
        if (req != MPI_REQUEST_NULL) {
            MPI_CHECK(MPI_Wait(&req, MPI_STATUS_IGNORE));
            CUDA_CHECK(cudaMemcpy(output + (size_t)(tile - 1) * p.B_M * p.d,
                                  recv_buf,
                                  (size_t)p.B_M * p.d * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        MPI_CHECK(MPI_Iallreduce(send_buf[buf], recv_buf,
                                 p.B_M * p.d, MPI_FLOAT, MPI_SUM, comm, &req));
    }

    // Wait for the final tile's allreduce
    MPI_CHECK(MPI_Wait(&req, MPI_STATUS_IGNORE));
    CUDA_CHECK(cudaMemcpy(output + (size_t)(p.num_tiles - 1) * p.B_M * p.d,
                          recv_buf,
                          (size_t)p.B_M * p.d * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    double t1 = MPI_Wtime();

    CUDA_CHECK(cudaFree(O_tile));
    CUDA_CHECK(cudaFree(send_buf[0]));
    CUDA_CHECK(cudaFree(send_buf[1]));
    CUDA_CHECK(cudaFree(recv_buf));

    RunResult result;
    result.wall_ms = (t1 - t0) * 1e3;
    return result;
}
