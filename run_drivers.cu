#include "run_drivers.cuh"
#include "flash_attn.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>

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

    float* O_tiles = nullptr;
    float* send_buf = nullptr;
    float* recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&O_tiles, (size_t)p.S * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.S * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buf, (size_t)p.S * p.d * sizeof(float)));

    MPI_CHECK(MPI_Barrier(comm));
    double t0 = MPI_Wtime();

    for (int tile = 0; tile < p.num_tiles; tile++)
        flash_attn_forward_tile(Q, K, V,
                                O_tiles + (size_t)tile * p.B_M * p.d_local,
                                tile, p.flash, compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        p.d, p.S, p.d_local,
        &alpha, W_O, p.d,
                O_tiles, p.d_local,
        &beta, send_buf, p.d));
    CUDA_CHECK(cudaStreamSynchronize(proj_stream));

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
    float* O_tile = nullptr;
    float* send_buf = nullptr;
    float* recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&O_tile, (size_t)p.B_M * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.B_M * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buf, (size_t)p.B_M * p.d * sizeof(float)));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    MPI_CHECK(MPI_Barrier(comm));
    double t0 = MPI_Wtime();

    for (int tile = 0; tile < p.num_tiles; tile++) {
        flash_attn_forward_tile(Q, K, V, O_tile, tile, p.flash, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        CUBLAS_CHECK(cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.d, p.B_M, p.d_local,
            &alpha, W_O, p.d,
                    O_tile, p.d_local,
            &beta, send_buf, p.d));
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
    float* O_tile = nullptr;
    float* send_buf[2] = {nullptr, nullptr};
    float* recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&O_tile, (size_t)p.B_M * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf[0], (size_t)p.B_M * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf[1], (size_t)p.B_M * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&recv_buf, (size_t)p.B_M * p.d * sizeof(float)));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    MPI_Request req = MPI_REQUEST_NULL;

    MPI_CHECK(MPI_Barrier(comm));
    double t0 = MPI_Wtime();

    for (int tile = 0; tile < p.num_tiles; tile++) {
        int buf = tile % 2;

        flash_attn_forward_tile(Q, K, V, O_tile, tile, p.flash, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        CUBLAS_CHECK(cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.d, p.B_M, p.d_local,
            &alpha, W_O, p.d,
                    O_tile, p.d_local,
            &beta, send_buf[buf], p.d));
        CUDA_CHECK(cudaStreamSynchronize(proj_stream));

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
