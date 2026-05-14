#include "run_drivers.cuh"
#include "flash_attn.cuh"

#include <cstdlib>

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
    MPI_Comm comm,
    ncclComm_t nccl_comm
)
{
    float* O_tiles = nullptr;
    float* send_buf = nullptr;
    cudaStream_t comm_stream;
    CUDA_CHECK(cudaMalloc(&O_tiles, (size_t)p.S * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.S * p.d * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&comm_stream));

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

    NCCL_CHECK(ncclAllReduce(send_buf, output,
                             (size_t)p.S * p.d, ncclFloat, ncclSum, nccl_comm, comm_stream));
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));

    double t1 = MPI_Wtime();

    CUDA_CHECK(cudaFree(O_tiles));
    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaStreamDestroy(comm_stream));

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
    MPI_Comm comm,
    ncclComm_t nccl_comm
)
{
    float* O_tile = nullptr;
    float* send_buf = nullptr;
    cudaStream_t comm_stream;
    CUDA_CHECK(cudaMalloc(&O_tile, (size_t)p.B_M * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf, (size_t)p.B_M * p.d * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&comm_stream));

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

        NCCL_CHECK(ncclAllReduce(send_buf,
                                 output + (size_t)tile * p.B_M * p.d,
                                 p.B_M * p.d, ncclFloat, ncclSum, nccl_comm, comm_stream));
        CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    }

    double t1 = MPI_Wtime();

    CUDA_CHECK(cudaFree(O_tile));
    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaStreamDestroy(comm_stream));

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
    MPI_Comm comm,
    ncclComm_t nccl_comm
)
{
    float* O_tile = nullptr;
    float* send_buf[2] = {nullptr, nullptr};
    cudaStream_t comm_stream;
    CUDA_CHECK(cudaMalloc(&O_tile, (size_t)p.B_M * p.d_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf[0], (size_t)p.B_M * p.d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&send_buf[1], (size_t)p.B_M * p.d * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&comm_stream));

    // comm_done[b] signals when the allreduce using send_buf[b] is complete,
    // allowing send_buf[b] to be reused by the projection 2 tiles later.
    cudaEvent_t comm_done[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&comm_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&comm_done[1], cudaEventDisableTiming));

    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSetStream(cublas_handle, proj_stream));

    MPI_CHECK(MPI_Barrier(comm));
    double t0 = MPI_Wtime();

    for (int tile = 0; tile < p.num_tiles; tile++) {
        int buf = tile % 2;

        if (tile >= 2)
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, comm_done[buf]));

        flash_attn_forward_tile(Q, K, V, O_tile, tile, p.flash, compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        CUBLAS_CHECK(cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            p.d, p.B_M, p.d_local,
            &alpha, W_O, p.d,
                    O_tile, p.d_local,
            &beta, send_buf[buf], p.d));
        CUDA_CHECK(cudaStreamSynchronize(proj_stream));

        // Enqueue allreduce on comm_stream — returns immediately.
        // Next tile's attention launches on compute_stream in parallel.
        NCCL_CHECK(ncclAllReduce(send_buf[buf],
                                 output + (size_t)tile * p.B_M * p.d,
                                 p.B_M * p.d, ncclFloat, ncclSum, nccl_comm, comm_stream));
        CUDA_CHECK(cudaEventRecord(comm_done[buf], comm_stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(comm_stream));

    double t1 = MPI_Wtime();

    CUDA_CHECK(cudaEventDestroy(comm_done[0]));
    CUDA_CHECK(cudaEventDestroy(comm_done[1]));
    CUDA_CHECK(cudaFree(O_tile));
    CUDA_CHECK(cudaFree(send_buf[0]));
    CUDA_CHECK(cudaFree(send_buf[1]));
    CUDA_CHECK(cudaStreamDestroy(comm_stream));

    RunResult result;
    result.wall_ms = (t1 - t0) * 1e3;
    return result;
}
