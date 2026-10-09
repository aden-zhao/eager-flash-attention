#define MAX_B_N 256
#define MAX_D_H 256

#include "flash_attn.cuh"

size_t flash_attn_smem_bytes(const FlashAttnParams& p)
{
    return 2 * p.B_N * p.d_h * sizeof(float);
}

void flash_attn_forward(
    const float* Q, const float* K, const float* V,
    float* O, int q_start, int q_rows,
    const FlashAttnParams& p, cudaStream_t stream,
    int* tile_counters, int tile_rows
)
{
    dim3 grid(p.H_local, q_rows / p.B_M);
    flash_attn_kernel<<<grid, p.B_M << 5, flash_attn_smem_bytes(p), stream>>>(
        Q, K, V, O, q_start, p, tile_counters, tile_rows);
}

__global__ void flash_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int q_start,
    FlashAttnParams p,
    int* __restrict__ tile_counters,
    int tile_rows
)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int cols_per = p.d_h / 32;
    int head = blockIdx.x;
    int block_row0 = blockIdx.y * p.B_M;
    int local_row = block_row0 + warp_id;   // row within this launch
    int q_row = q_start + local_row;

    int scores_per = p.B_N / 32;
    int kv_elems = p.B_N * p.d_h;
    size_t head_off = (size_t)head * p.S * p.d_h;

    const float* Q_lane = Q + head_off + (size_t)q_row * p.d_h + lane_id * cols_per;

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + kv_elems;

    float scores[MAX_B_N / 32];
    float O_acc[MAX_D_H / 32] = {0};
    float m = -INFINITY;
    float ell = 0.0f;

    for (int j = 0; j < p.S; j += p.B_N)
    {
        // Block-uniform early exit: no row of this block needs keys past the block's last row.
        if (p.causal && j > q_start + block_row0 + p.B_M - 1) break;
        const float* K_base = K + head_off + (size_t)j * p.d_h;
        const float* V_base = V + head_off + (size_t)j * p.d_h;

        for (int e = threadIdx.x; e < kv_elems; e += blockDim.x) {
            K_smem[e] = K_base[e];
            V_smem[e] = V_base[e];
        }
        __syncthreads();

        for (int q = 0; q < p.B_N; q++) {
            float partial = 0.0f;
            const float* K_lane = K_smem + q * p.d_h + lane_id * cols_per;

            for (int f = 0; f < cols_per; f++)
                partial += Q_lane[f] * K_lane[f];

            partial *= p.scale;

            for (int offset = 16; offset >= 1; offset >>= 1)
                partial += __shfl_xor_sync(0xffffffff, partial, offset);

            if (p.causal && j + q > q_row) partial = -INFINITY;

            if (lane_id == q % 32)
                scores[q / 32] = partial;
        }

        float local_max = scores[0];
        for (int i = 1; i < scores_per; i++)
            local_max = fmaxf(local_max, scores[i]);

        for (int offset = 16; offset >= 1; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));

        float m_new = fmaxf(m, local_max);
        float correction = expf(m - m_new);
        m = m_new;

        for (int f = 0; f < cols_per; f++)
            O_acc[f] *= correction;

        float local_sum = 0.0f;
        for (int i = 0; i < scores_per; i++) {
            scores[i] = expf(scores[i] - m_new);
            local_sum += scores[i];
        }
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        ell = correction * ell + local_sum;

        for (int q = 0; q < p.B_N; q++) {
            float P_q = __shfl_sync(0xffffffff, scores[q / 32], q % 32);
            const float* V_lane = V_smem + q * p.d_h + lane_id * cols_per;
            for (int f = 0; f < cols_per; f++)
                O_acc[f] += P_q * V_lane[f];
        }

        __syncthreads();
    }

    float* O_lane = O + (size_t)local_row * p.H_local * p.d_h + head * p.d_h + lane_id * cols_per;
    for (int f = 0; f < cols_per; f++)
        O_lane[f] = O_acc[f] / ell;

    if (tile_counters != nullptr) {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(tile_counters + block_row0 / tile_rows, 1);
    }
}

const char* flash_attn_check_params(const FlashAttnParams& p, int q_rows, int cuda_device)
{
    if (p.B_M < 1 || (p.B_M << 5) > 1024) return "v1: B_M must be in [1, 32]";
    if (p.d_h % 32 != 0) return "v1: d_h must be a multiple of 32";
    if (p.d_h > MAX_D_H) return "v1: d_h too large";
    if (p.B_N % 32 != 0 || p.B_N > MAX_B_N) return "v1: B_N must be a multiple of 32 and <= 256";
    if (p.S % p.B_N != 0) return "v1: S must be a multiple of B_N";
    if (q_rows % p.B_M != 0) return "v1: tile must be a multiple of B_M";

    int maxSmem;
    cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlock, cuda_device);
    if (flash_attn_smem_bytes(p) > (size_t)maxSmem) return "v1: B_N * d_h too large for shared memory";
    return nullptr;
}
