/*
 * Baseline attention: naive GPU softmax attention for correctness / timing
 * scaffolding. Not tuned — teammate's kernel should replace this path.
 *
 * Optional: -DBASELINE_ATTENTION_SYNC_DEBUG syncs the stream after each kernel
 * (correctness/debug only; do not use when measuring overlap).
 */

#include "baseline_attention.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "baseline_attention CUDA error %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#ifdef BASELINE_ATTENTION_SYNC_DEBUG
#define BASELINE_ATTENTION_SYNC_STREAM(s) CUDA_CHECK(cudaStreamSynchronize(s))
#else
#define BASELINE_ATTENTION_SYNC_STREAM(s) ((void)0)
#endif

namespace {

constexpr int kSoftmaxThreads = 256;

__global__ void attention_scores_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int M,
    int Kdim,
    float inv_sqrt_d
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= M)
        return;

    float sum = 0.f;
    const float* qi = Q + static_cast<size_t>(i) * Kdim;
    const float* kj = K + static_cast<size_t>(j) * Kdim;
    for (int d = 0; d < Kdim; ++d)
        sum = __fmaf_rn(qi[d], kj[d], sum);

    scores[static_cast<size_t>(i) * M + j] = sum * inv_sqrt_d;
}

/*
 * One CUDA block per row of scores; softmax across columns (keys). Writes
 * probabilities in-place over scores[row * M : row * M + M].
 *
 * Requires blockDim.x == kSoftmaxThreads (fixed shared memory layout).
 */
__global__ void softmax_rows_kernel(float* scores, int M) {
    const int row = blockIdx.x;
    float* row_scores = scores + static_cast<size_t>(row) * M;

    __shared__ float sreduce[kSoftmaxThreads];

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < M; j += blockDim.x)
        local_max = fmaxf(local_max, row_scores[j]);

    sreduce[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] =
                fmaxf(sreduce[threadIdx.x], sreduce[threadIdx.x + stride]);
        __syncthreads();
    }

    const float row_max = sreduce[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int j = threadIdx.x; j < M; j += blockDim.x)
        local_sum += expf(row_scores[j] - row_max);

    sreduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] += sreduce[threadIdx.x + stride];
        __syncthreads();
    }

    const float row_sum = sreduce[0];
    __syncthreads();

    for (int j = threadIdx.x; j < M; j += blockDim.x)
        row_scores[j] = expf(row_scores[j] - row_max) / row_sum;
}

__global__ void attention_output_kernel(
    const float* attn_weights,
    const float* V,
    float* O,
    int M,
    int Kdim
) {
    const int i = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= M || d >= Kdim)
        return;

    float sum = 0.f;
    const float* row = attn_weights + static_cast<size_t>(i) * M;
    for (int j = 0; j < M; ++j)
        sum = __fmaf_rn(row[j], V[static_cast<size_t>(j) * Kdim + d], sum);

    O[static_cast<size_t>(i) * Kdim + d] = sum;
}

__global__ void fill_sample_qkv_kernel(
    float* Q,
    float* K,
    float* V,
    int M,
    int Kdim,
    int mpi_rank,
    int tile_id
) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total =
        static_cast<size_t>(M) * static_cast<size_t>(Kdim);
    if (idx >= total)
        return;

    const int token = static_cast<int>(idx / static_cast<size_t>(Kdim));
    const int dim = static_cast<int>(idx % static_cast<size_t>(Kdim));

    const float base =
        static_cast<float>((mpi_rank + 1) * 10000 + (tile_id + 1) * 1000);

    /* Simple separated patterns so Q,K,V are not identical. */
    Q[idx] = sinf(base * 0.001f + static_cast<float>(token) * 0.1f +
                  static_cast<float>(dim) * 0.01f);
    K[idx] = cosf(base * 0.0013f + static_cast<float>(token) * 0.11f +
                  static_cast<float>(dim) * 0.011f);
    V[idx] = tanhf(base * 0.0007f + static_cast<float>(token) * 0.09f +
                   static_cast<float>(dim) * 0.009f);
}

}  // namespace

size_t baseline_attention_workspace_bytes(int proj_m) {
    if (proj_m <= 0)
        return 0;

    const size_t m = static_cast<size_t>(proj_m);
    const size_t max_square_floats = SIZE_MAX / sizeof(float);
    if (m > max_square_floats / m) {
        fprintf(stderr,
                "baseline_attention: workspace element count overflows size_t\n");
        std::abort();
    }

    return m * m * sizeof(float);
}

void launch_baseline_attention_fill_sample_qkv(
    cudaStream_t stream,
    float* Q,
    float* K,
    float* V,
    int proj_m,
    int proj_k,
    int mpi_rank,
    int tile_id
) {
    if (proj_m <= 0 || proj_k <= 0)
        return;

    const size_t total =
        static_cast<size_t>(proj_m) * static_cast<size_t>(proj_k);

    constexpr int threads = 256;
    const size_t blocks64 = (total + static_cast<size_t>(threads) - 1U) /
                            static_cast<size_t>(threads);

    if (blocks64 > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr,
                "baseline_attention_fill_sample_qkv: grid dimension too large\n");
        std::abort();
    }

    fill_sample_qkv_kernel<<<static_cast<unsigned int>(blocks64), threads, 0,
                               stream>>>(
        Q, K, V, proj_m, proj_k, mpi_rank, tile_id);
    CUDA_CHECK(cudaGetLastError());
    BASELINE_ATTENTION_SYNC_STREAM(stream);
}

void launch_baseline_attention(
    cudaStream_t stream,
    const float* Q,
    const float* K,
    const float* V,
    float* out_pre_proj,
    float* workspace,
    size_t workspace_bytes,
    int proj_m,
    int proj_k
) {
    if (proj_m <= 0 || proj_k <= 0)
        return;

    const size_t need = baseline_attention_workspace_bytes(proj_m);
    if (workspace_bytes < need) {
        fprintf(stderr,
                "baseline_attention: workspace too small (have %zu need %zu)\n",
                workspace_bytes, need);
        std::abort();
    }

    const float inv_sqrt_d = 1.f / sqrtf(static_cast<float>(proj_k));

    dim3 block_scores(16, 16);
    dim3 grid_scores((proj_m + 15) / 16, (proj_m + 15) / 16);
    attention_scores_kernel<<<grid_scores, block_scores, 0, stream>>>(
        Q, K, workspace, proj_m, proj_k, inv_sqrt_d);
    CUDA_CHECK(cudaGetLastError());
    BASELINE_ATTENTION_SYNC_STREAM(stream);

    softmax_rows_kernel<<<proj_m, kSoftmaxThreads, 0, stream>>>(
        workspace, proj_m);
    CUDA_CHECK(cudaGetLastError());
    BASELINE_ATTENTION_SYNC_STREAM(stream);

    const int out_threads = 256;
    dim3 block_out(out_threads, 1);
    dim3 grid_out(proj_m, (proj_k + out_threads - 1) / out_threads);
    attention_output_kernel<<<grid_out, block_out, 0, stream>>>(
        workspace, V, out_pre_proj, proj_m, proj_k);
    CUDA_CHECK(cudaGetLastError());
    BASELINE_ATTENTION_SYNC_STREAM(stream);
}

static void host_fill_qkv(
    std::vector<float>& Q,
    std::vector<float>& K,
    std::vector<float>& V,
    int M,
    int Kdim,
    int mpi_rank,
    int tile_id
) {
    const size_t n = static_cast<size_t>(M) * static_cast<size_t>(Kdim);
    Q.resize(n);
    K.resize(n);
    V.resize(n);
    const float base =
        static_cast<float>((mpi_rank + 1) * 10000 + (tile_id + 1) * 1000);

    for (int tok = 0; tok < M; ++tok) {
        for (int d = 0; d < Kdim; ++d) {
            const size_t idx =
                static_cast<size_t>(tok) * static_cast<size_t>(Kdim) +
                static_cast<size_t>(d);
            Q[idx] =
                sinf(base * 0.001f + static_cast<float>(tok) * 0.1f +
                     static_cast<float>(d) * 0.01f);
            K[idx] =
                cosf(base * 0.0013f + static_cast<float>(tok) * 0.11f +
                     static_cast<float>(d) * 0.011f);
            V[idx] =
                tanhf(base * 0.0007f + static_cast<float>(tok) * 0.09f +
                      static_cast<float>(d) * 0.009f);
        }
    }
}

void baseline_attention_host_rank_projected_tile(
    int mpi_rank,
    int tile_id,
    int proj_m,
    int proj_k,
    int proj_n,
    float w_elem,
    float* out,
    size_t out_elems
) {
    const size_t need =
        static_cast<size_t>(proj_m) * static_cast<size_t>(proj_n);
    if (proj_m <= 0 || proj_k <= 0 || proj_n <= 0 ||
        out_elems < need) {
        fprintf(stderr,
                "baseline_attention_host_rank_projected_tile: invalid args "
                "(m=%d k=%d n=%d out_elems=%zu need=%zu)\n",
                proj_m, proj_k, proj_n, out_elems, need);
        std::abort();
    }

    std::vector<float> Q, K, V;
    host_fill_qkv(Q, K, V, proj_m, proj_k, mpi_rank, tile_id);

    const float inv_sqrt_d = 1.f / sqrtf(static_cast<float>(proj_k));

    std::vector<float> scores(static_cast<size_t>(proj_m) * proj_m);
    for (int i = 0; i < proj_m; ++i) {
        for (int j = 0; j < proj_m; ++j) {
            float s = 0.f;
            const size_t row_q =
                static_cast<size_t>(i) * static_cast<size_t>(proj_k);
            const size_t row_k =
                static_cast<size_t>(j) * static_cast<size_t>(proj_k);
            for (int d = 0; d < proj_k; ++d)
                s = std::fma(Q[row_q + d], K[row_k + d], s);
            scores[static_cast<size_t>(i) * proj_m + j] = s * inv_sqrt_d;
        }
    }

    for (int i = 0; i < proj_m; ++i) {
        float row_max = -INFINITY;
        float* row = scores.data() + static_cast<size_t>(i) * proj_m;
        for (int j = 0; j < proj_m; ++j)
            row_max = std::max(row_max, row[j]);

        float sum_exp = 0.f;
        for (int j = 0; j < proj_m; ++j)
            sum_exp += expf(row[j] - row_max);

        for (int j = 0; j < proj_m; ++j)
            row[j] = expf(row[j] - row_max) / sum_exp;
    }

    std::vector<float> O(static_cast<size_t>(proj_m) * proj_k);
    for (int i = 0; i < proj_m; ++i) {
        for (int d = 0; d < proj_k; ++d) {
            float sum = 0.f;
            for (int j = 0; j < proj_m; ++j) {
                const float w =
                    scores[static_cast<size_t>(i) * proj_m + j];
                sum = std::fma(
                    w,
                    V[static_cast<size_t>(j) * proj_k + d],
                    sum);
            }
            O[static_cast<size_t>(i) * proj_k + d] = sum;
        }
    }

    /* Column-major C (proj_n × proj_m): C[r,c] = w_elem * sum_k B[k,c]. */
    for (int c = 0; c < proj_m; ++c) {
        float dot_o = 0.f;
        for (int k = 0; k < proj_k; ++k)
            dot_o += O[static_cast<size_t>(c) * proj_k + k];
        const float col_val = w_elem * dot_o;
        for (int r = 0; r < proj_n; ++r)
            out[r + c * proj_n] = col_val;
    }
}

void baseline_attention_host_expected_after_allreduce(
    int tile_id,
    int world_size,
    int proj_m,
    int proj_k,
    int proj_n,
    float w_elem,
    float* out_reduced,
    size_t out_elems
) {
    const size_t need =
        static_cast<size_t>(proj_m) * static_cast<size_t>(proj_n);
    if (world_size <= 0 || out_elems < need) {
        fprintf(stderr,
                "baseline_attention_host_expected_after_allreduce: invalid args\n");
        std::abort();
    }

    std::fill(out_reduced, out_reduced + need, 0.f);
    std::vector<float> rank_out(need);
    for (int r = 0; r < world_size; ++r) {
        baseline_attention_host_rank_projected_tile(
            r, tile_id, proj_m, proj_k, proj_n, w_elem,
            rank_out.data(), need);
        for (size_t i = 0; i < need; ++i)
            out_reduced[i] += rank_out[i];
    }
}
