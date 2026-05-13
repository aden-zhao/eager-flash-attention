#define MAX_B_N 256
#define MAX_D_H 256

#include "flash_attn.cuh"

size_t flash_attn_smem_bytes(const FlashAttnParams& p)
{
    return 2 * p.B_N * p.d_h * sizeof(float);
}

void flash_attn_forward_tile(
    const float* Q, const float* K, const float* V,
    float* O_tile, int tile_idx,
    const FlashAttnParams& p, cudaStream_t stream
)
{
    flash_attn_kernel<<<p.H_local, p.B_M * 32, flash_attn_smem_bytes(p), stream>>>(
        Q, K, V, O_tile, tile_idx, p);
}

__global__ void flash_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_tile,
    int tile_idx,
    FlashAttnParams p
)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int cols_per = p.d_h / 32;
    int head = blockIdx.x;

    int scores_per = p.B_N / 32;
    int kv_elems = p.B_N * p.d_h;
    int head_off = head * p.S * p.d_h;

    const float* Q_lane = Q + head_off + (tile_idx * p.B_M + warp_id) * p.d_h + lane_id * cols_per;

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + kv_elems;

    float scores[MAX_B_N / 32];
    float O_acc[MAX_D_H / 32] = {0};
    float m = -INFINITY;
    float ell = 0.0f;

    for (int j = 0; j < p.S; j += p.B_N)
    {
        const float* K_base = K + head_off + j * p.d_h;
        const float* V_base = V + head_off + j * p.d_h;

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

    float* O_lane = O_tile + warp_id * p.H_local * p.d_h + head * p.d_h + lane_id * cols_per;
    for (int f = 0; f < cols_per; f++)
        O_lane[f] = O_acc[f] / ell;
}

bool flash_attn_check_params(const FlashAttnParams& p, int cuda_device)
{
    if (p.S <= 0 || p.d_h <= 0 || p.H_local <= 0 || p.B_M <= 0 || p.B_N <= 0) return false;
    if (p.d_h > MAX_D_H || p.B_N > MAX_B_N) return false;
    if (p.d_h % 32 != 0) return false;
    if (p.B_N % 32 != 0) return false;
    if (p.B_M * 32 > 1024) return false;

    int maxSmem;
    cudaError_t attr_status = cudaDeviceGetAttribute(
        &maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda_device);
    if (attr_status != cudaSuccess) return false;

    size_t smem_bytes = flash_attn_smem_bytes(p);
    if (smem_bytes > static_cast<size_t>(maxSmem)) return false;
    if (p.S % p.B_N != 0) return false;
    if (p.S % p.B_M != 0) return false;

    attr_status = cudaFuncSetAttribute(
        flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (attr_status != cudaSuccess) return false;

    return true;
}
