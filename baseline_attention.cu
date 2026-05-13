#include "baseline_attention.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

// One block per row. Shared memory holds per-thread partial values.
// Computes softmax in-place over ncols elements.
static __global__ void softmax_rows(float* A, int ncols)
{
    extern __shared__ float smem[];
    float* row = A + blockIdx.x * ncols;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < ncols; j += blockDim.x)
        local_max = fmaxf(local_max, row[j]);
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int j = threadIdx.x; j < ncols; j += blockDim.x)
        local_sum += expf(row[j] - row_max);
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    for (int j = threadIdx.x; j < ncols; j += blockDim.x)
        row[j] = expf(row[j] - row_max) / row_sum;
}

// Copy head h's result O_h[S, d_h] into O[S, d_local] at column offset h*d_h.
static __global__ void scatter_head(
    float* O, const float* O_h,
    int S, int d_h, int d_local, int col_offset)
{
    int s = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (s < S && c < d_h)
        O[s * d_local + col_offset + c] = O_h[s * d_h + c];
}

size_t baseline_attention_workspace_bytes(int S)
{
    return (size_t)S * S * sizeof(float);
}

void baseline_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* workspace,
    int H_local,
    int S,
    int d_h,
    cudaStream_t stream,
    cublasHandle_t handle)
{
    const int d_local = H_local * d_h;
    const float scale = 1.f / sqrtf((float)d_h);
    const float one = 1.f, zero = 0.f;

    // Temporary buffer for one head's attention output before scattering.
    float* O_h = nullptr;
    CUDA_CHECK(cudaMalloc(&O_h, (size_t)S * d_h * sizeof(float)));

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    for (int h = 0; h < H_local; h++) {
        const float* Q_h = Q + (size_t)h * S * d_h;
        const float* K_h = K + (size_t)h * S * d_h;
        const float* V_h = V + (size_t)h * S * d_h;

        // scores[S, S] = (1/sqrt(d_h)) * Q_h[S, d_h] * K_h[S, d_h]^T
        // cuBLAS row-major trick: swap operands, use CUBLAS_OP_T on K_h.
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            S, S, d_h,
            &scale,
            K_h, d_h,
            Q_h, d_h,
            &zero,
            workspace, S));

        // softmax each row of scores in-place
        softmax_rows<<<S, 256, 256 * sizeof(float), stream>>>(workspace, S);
        CUDA_CHECK(cudaGetLastError());

        // O_h[S, d_h] = scores[S, S] * V_h[S, d_h]
        // cuBLAS row-major: swap operands.
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_h, S, S,
            &one,
            V_h, d_h,
            workspace, S,
            &zero,
            O_h, d_h));

        // scatter O_h into O at column offset h*d_h
        dim3 grid(S, (d_h + 255) / 256);
        scatter_head<<<grid, 256, 0, stream>>>(O, O_h, S, d_h, d_local, h * d_h);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(O_h));
}
