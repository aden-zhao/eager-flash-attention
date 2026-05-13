#include "run_overlap.cuh"

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
    // Per tile: flash_attn -> project -> MPI_Iallreduce (non-blocking).
    // Iallreduce for tile i overlaps with flash_attn for tile i+1.
    RunResult result = {};
    return result;
}
