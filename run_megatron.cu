#include "run_megatron.cuh"

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
    // All T_M tiles computed first, then one blocking allreduce.
    // No per-tile communication.
    RunResult result = {};
    return result;
}
