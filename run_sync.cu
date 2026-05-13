#include "run_sync.cuh"

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
    // Per tile: flash_attn -> project -> blocking MPI_Allreduce.
    // No overlap. Isolates tiling overhead from async benefit.
    RunResult result = {};
    return result;
}
