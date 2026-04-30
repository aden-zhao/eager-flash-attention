#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

/*
 * Baseline (reference-style) single-head scaled dot-product attention for the
 * tile_comm harness.
 *
 * Tensor layout matches pre_proj in tile_comm.cu:
 *   - Logical shape: proj_m tokens × proj_k floats per token
 *   - Storage: row-major, linear index (i, d) -> i * proj_k + d
 *
 * Computes O = softmax(Q K^T / sqrt(proj_k)) V into out_pre_proj.
 *
 * Workspace: one proj_m × proj_m scratch matrix (scores, then softmax in-place).
 *
 * Non-positive proj_m or proj_k: launch helpers return immediately (no kernels).
 */

size_t baseline_attention_workspace_bytes(int proj_m);

/* Deterministic toy Q/K/V for benchmarking before real weights exist. */
void launch_baseline_attention_fill_sample_qkv(
    cudaStream_t stream,
    float* Q,
    float* K,
    float* V,
    int proj_m,
    int proj_k,
    int mpi_rank,
    int tile_id);

void launch_baseline_attention(
    cudaStream_t stream,
    const float* Q,
    const float* K,
    const float* V,
    float* out_pre_proj,
    float* workspace,
    size_t workspace_bytes,
    int proj_m,
    int proj_k);

/*
 * Host reference for verification: same QKV sampling + attention + projection as
 * tile_comm (constant W = w_elem everywhere). Writes one rank's projected tile in
 * column-major cuBLAS layout: linear index i maps to r=i%proj_n, c=i/proj_n,
 * value = w_elem * sum_k O[c,k].
 */
void baseline_attention_host_rank_projected_tile(
    int mpi_rank,
    int tile_id,
    int proj_m,
    int proj_k,
    int proj_n,
    float w_elem,
    float* out,
    size_t out_elems);

/* Sum projected tiles over all ranks (MPI allreduce expectation). */
void baseline_attention_host_expected_after_allreduce(
    int tile_id,
    int world_size,
    int proj_m,
    int proj_k,
    int proj_n,
    float w_elem,
    float* out_reduced,
    size_t out_elems);
