#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "flash_attn.cuh"

// Error checking macros
#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if(e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at line " << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

#define CUBLAS_CHECK(cmd) do { \
    cublasStatus_t s = cmd; \
    if(s != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error Code: " << s << " at line " << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

#define MPI_CHECK(cmd) do { \
    int e = cmd; \
    if(e != MPI_SUCCESS) { \
        char err_string[MPI_MAX_ERROR_STRING]; \
        int err_len = 0; \
        MPI_Error_string(e, err_string, &err_len); \
        std::cerr << "MPI Error: " << err_string << " at line " << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

enum RunMode {
    SERIALIZED = 1,
    TILED_UNOVERLAPPED = 2,
    OVERLAPPED = 3
};

int main(int argc, char* argv[]) {
    // 1. Initialize MPI
    int required_thread_support = MPI_THREAD_MULTIPLE;
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, required_thread_support, &provided_thread_support);

    int rank, num_ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    // Parse command line arguments
    if (argc < 6) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <S> <d_h> <H_local> <B_M> <B_N> [mode (1:Serial, 2:Tiled, 3:Overlapped)] [--validate]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    FlashAttnParams p;
    p.S = std::atoi(argv[1]);
    p.d_h = std::atoi(argv[2]);
    p.H_local = std::atoi(argv[3]);
    p.B_M = std::atoi(argv[4]);
    p.B_N = std::atoi(argv[5]);
    p.scale = 1.0f / std::sqrt(static_cast<float>(p.d_h));
    
    int mode = (argc > 6) ? std::atoi(argv[6]) : OVERLAPPED;
    bool validate = false;
    for (int argi = 7; argi < argc; ++argi) {
        std::string arg = argv[argi];
        if (arg == "--validate" || arg == "validate") {
            validate = true;
        }
    }
    if (mode < SERIALIZED || mode > OVERLAPPED) {
        if (rank == 0) std::cerr << "Invalid mode. Use 1, 2, or 3." << std::endl;
        MPI_Finalize();
        return 1;
    }

    // 2. Bind MPI Rank to Local GPU Device
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    const char* slurm_localid = std::getenv("SLURM_LOCALID");
    int local_rank = slurm_localid ? std::atoi(slurm_localid) : rank;
    int local_device = local_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(local_device));

    const char* mpich_gpu_support = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
    if (rank == 0 && (!mpich_gpu_support || std::atoi(mpich_gpu_support) == 0)) {
        std::cerr << "Warning: MPICH_GPU_SUPPORT_ENABLED is not set to 1. "
                  << "Cray MPICH may not accept CUDA device buffers directly." << std::endl;
    }

    if (!flash_attn_check_params(p, local_device)) {
        if (rank == 0) std::cerr << "Parameter check failed for GPU configuration." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int T_M = p.S / p.B_M; // Number of outer query rows tiles
    int d_model = p.d_h * p.H_local * num_ranks; // Assumed hidden dimension sizing

    // Size Calculations
    size_t qkv_bytes = static_cast<size_t>(p.S) * p.d_h * p.H_local * sizeof(float);
    size_t tile_O_bytes = static_cast<size_t>(p.B_M) * p.d_h * p.H_local * sizeof(float);
    size_t tile_P_bytes = static_cast<size_t>(p.B_M) * d_model * sizeof(float);
    size_t total_O_bytes = static_cast<size_t>(p.S) * d_model * sizeof(float);
    size_t weight_bytes = static_cast<size_t>(p.d_h * p.H_local) * d_model * sizeof(float);

    // 3. Allocate Memory
    float *d_Q, *d_K, *d_V, *d_W_O;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_O, weight_bytes));

    // For pipelining, we allocate an array of buffers or reuse buffers depending on execution mode
    float *d_O_tile;
    CUDA_CHECK(cudaMalloc(&d_O_tile, tile_O_bytes));
    
    // In the non-blocking pipeline, distinct iterations need separate buffers
    // to avoid races while kernels, GEMMs, and MPI requests are still active.
    float **d_O_tile_array = new float*[T_M];
    float **d_P_tile_array = new float*[T_M];
    float **d_P_reduced_array = new float*[T_M];
    for(int i = 0; i < T_M; ++i) {
        CUDA_CHECK(cudaMalloc(&d_O_tile_array[i], tile_O_bytes));
        CUDA_CHECK(cudaMalloc(&d_P_tile_array[i], tile_P_bytes));
        CUDA_CHECK(cudaMalloc(&d_P_reduced_array[i], tile_P_bytes));
    }
    
    float *d_O_final;
    CUDA_CHECK(cudaMalloc(&d_O_final, total_O_bytes));

    // Initialize deterministic mock data. cudaMemset is byte-oriented, so use
    // host initialization to avoid accidentally benchmarking all-zero floats.
    size_t qkv_elems = qkv_bytes / sizeof(float);
    size_t weight_elems = weight_bytes / sizeof(float);
    std::vector<float> h_Q(qkv_elems), h_K(qkv_elems), h_V(qkv_elems), h_W_O(weight_elems);
    for (size_t i = 0; i < qkv_elems; ++i) {
        h_Q[i] = 0.01f * static_cast<float>((i % 17) + 1);
        h_K[i] = 0.01f * static_cast<float>((i % 13) + 1);
        h_V[i] = 0.01f * static_cast<float>((i % 11) + 1);
    }
    for (size_t i = 0; i < weight_elems; ++i) {
        h_W_O[i] = 0.001f * static_cast<float>((i % 19) + 1);
    }
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_O, h_W_O.data(), weight_bytes, cudaMemcpyHostToDevice));

    auto compute_cpu_reference = [&]() {
        int local_width = p.d_h * p.H_local;
        std::vector<float> h_O_full(static_cast<size_t>(p.S) * local_width, 0.0f);
        std::vector<float> h_O_projected(total_O_bytes / sizeof(float), 0.0f);

        for (int head = 0; head < p.H_local; ++head) {
            int head_off = head * p.S * p.d_h;
            for (int q = 0; q < p.S; ++q) {
                std::vector<float> scores(p.S);
                float row_max = -std::numeric_limits<float>::infinity();

                for (int k = 0; k < p.S; ++k) {
                    float score = 0.0f;
                    for (int f = 0; f < p.d_h; ++f) {
                        score += h_Q[head_off + q * p.d_h + f] * h_K[head_off + k * p.d_h + f];
                    }
                    score *= p.scale;
                    scores[k] = score;
                    row_max = std::max(row_max, score);
                }

                float row_sum = 0.0f;
                for (int k = 0; k < p.S; ++k) {
                    scores[k] = std::exp(scores[k] - row_max);
                    row_sum += scores[k];
                }

                for (int f = 0; f < p.d_h; ++f) {
                    float value = 0.0f;
                    for (int k = 0; k < p.S; ++k) {
                        value += (scores[k] / row_sum) * h_V[head_off + k * p.d_h + f];
                    }
                    h_O_full[static_cast<size_t>(q) * local_width + head * p.d_h + f] = value;
                }
            }
        }

        for (int token = 0; token < p.S; ++token) {
            for (int row = 0; row < d_model; ++row) {
                float acc = 0.0f;
                for (int col = 0; col < local_width; ++col) {
                    acc += h_W_O[static_cast<size_t>(col) * d_model + row] *
                           h_O_full[static_cast<size_t>(token) * local_width + col];
                }
                h_O_projected[static_cast<size_t>(token) * d_model + row] =
                    acc * static_cast<float>(num_ranks);
            }
        }

        return h_O_projected;
    };

    // 4. Initialize Streams and Handles
    cudaStream_t compute_stream, projection_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&projection_stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, projection_stream));

    auto run_mode = [&](int run_mode) {
        CUDA_CHECK(cudaMemset(d_O_final, 0, total_O_bytes));

        if (run_mode == SERIALIZED) {
        float *d_O_full;
        CUDA_CHECK(cudaMalloc(&d_O_full, static_cast<size_t>(p.S) * p.d_h * p.H_local * sizeof(float)));
        
        // Compute all attention sequentially on GPU
        for (int i = 0; i < T_M; ++i) {
            float* d_O_tile_ptr = d_O_full + (i * p.B_M * p.d_h * p.H_local);
            flash_attn_forward_tile(d_Q, d_K, d_V, d_O_tile_ptr, i, p, compute_stream);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // Global Matrix Multiplication Output Projection: O_full * W_O = O_final
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            d_model, p.S, p.d_h * p.H_local,
            &alpha, d_W_O, d_model,
            d_O_full, p.d_h * p.H_local,
            &beta, d_O_final, d_model
        ));
        CUDA_CHECK(cudaStreamSynchronize(projection_stream));

        // Synchronous Global Communication Barrier
        // If your MPI build is not CUDA-aware, stage this through host memory first
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, d_O_final, total_O_bytes / sizeof(float), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
        CUDA_CHECK(cudaFree(d_O_full));
        }
        else if (run_mode == TILED_UNOVERLAPPED) {
        // Tiled execution loop without interleaving MPI calls
        for (int i = 0; i < T_M; ++i) {
            flash_attn_forward_tile(d_Q, d_K, d_V, d_O_tile, i, p, compute_stream);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));

            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_model, p.B_M, p.d_h * p.H_local,
                &alpha, d_W_O, d_model,
                d_O_tile, p.d_h * p.H_local,
                &beta, d_P_tile_array[i], d_model
            ));
            CUDA_CHECK(cudaStreamSynchronize(projection_stream));
        }

        // Sequential block execution of communication at the end
        for (int i = 0; i < T_M; ++i) {
            MPI_CHECK(MPI_Allreduce(d_P_tile_array[i], d_P_reduced_array[i], tile_P_bytes / sizeof(float), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
            CUDA_CHECK(cudaMemcpyAsync(d_O_final + (i * p.B_M * d_model), d_P_reduced_array[i], tile_P_bytes, cudaMemcpyDeviceToDevice, projection_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(projection_stream));
        }
        else if (run_mode == OVERLAPPED) {
        std::vector<MPI_Request> mpi_requests(T_M);
        std::vector<MPI_Status> mpi_statuses(T_M);
        std::vector<cudaEvent_t> attention_done(T_M);
        std::vector<cudaEvent_t> projection_done(T_M);

        for (int i = 0; i < T_M; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&attention_done[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&projection_done[i], cudaEventDisableTiming));
            mpi_requests[i] = MPI_REQUEST_NULL;
        }

        for (int i = 0; i < T_M; ++i) {
            // Step A: Launch Attention Tile i
            flash_attn_forward_tile(d_Q, d_K, d_V, d_O_tile_array[i], i, p, compute_stream);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(attention_done[i], compute_stream));

            // Step B: Linear Layer Projection waits for attention without blocking the host
            CUDA_CHECK(cudaStreamWaitEvent(projection_stream, attention_done[i], 0));
            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasSgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d_model, p.B_M, p.d_h * p.H_local,
                &alpha, d_W_O, d_model,
                d_O_tile_array[i], p.d_h * p.H_local,
                &beta, d_P_tile_array[i], d_model
            ));
            CUDA_CHECK(cudaEventRecord(projection_done[i], projection_stream));

            if (i > 0) {
                CUDA_CHECK(cudaEventSynchronize(projection_done[i - 1]));

                // Step C: Trigger Non-Blocking Tile-Level Communication Window
                // Communication for tile i-1 can progress while tile i is still queued/running.
                int count = tile_P_bytes / sizeof(float);
                MPI_CHECK(MPI_Iallreduce(d_P_tile_array[i - 1], d_P_reduced_array[i - 1], count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &mpi_requests[i - 1]));
            }
        }

        CUDA_CHECK(cudaEventSynchronize(projection_done[T_M - 1]));
        int count = tile_P_bytes / sizeof(float);
        MPI_CHECK(MPI_Iallreduce(d_P_tile_array[T_M - 1], d_P_reduced_array[T_M - 1], count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &mpi_requests[T_M - 1]));

        // Global Synchronization Boundary across asynchronous communication streams
        MPI_CHECK(MPI_Waitall(T_M, mpi_requests.data(), mpi_statuses.data()));

        // Gather final completed tiles into unified contiguous matrix layout
        for (int i = 0; i < T_M; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(d_O_final + (i * p.B_M * d_model), d_P_reduced_array[i], tile_P_bytes, cudaMemcpyDeviceToDevice, projection_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(projection_stream));

        for (int i = 0; i < T_M; ++i) {
            CUDA_CHECK(cudaEventDestroy(attention_done[i]));
            CUDA_CHECK(cudaEventDestroy(projection_done[i]));
        }
        }
    };

    // Timing Instrumentation Setup
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start_time = MPI_Wtime();
    CUDA_CHECK(cudaEventRecord(start_event, compute_stream));

    run_mode(mode);

    // End Instrumentation Tracking
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_event, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double end_time = MPI_Wtime();

    float gpu_elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_elapsed_ms, start_event, stop_event));
    double wall_clock_time = end_time - start_time;
    double wall_ms = wall_clock_time * 1000.0;
    double exposed_ms = std::max(wall_ms - static_cast<double>(gpu_elapsed_ms), 0.0);

    std::string validation_status = validate ? "fail" : "";
    double max_abs_err = std::numeric_limits<double>::quiet_NaN();
    double max_rel_err = std::numeric_limits<double>::quiet_NaN();

    if (validate) {
        std::vector<float> candidate(total_O_bytes / sizeof(float));
        CUDA_CHECK(cudaMemcpy(candidate.data(), d_O_final, total_O_bytes, cudaMemcpyDeviceToHost));
        std::vector<float> reference = compute_cpu_reference();

        max_abs_err = 0.0;
        max_rel_err = 0.0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            double diff = std::abs(static_cast<double>(candidate[i]) - static_cast<double>(reference[i]));
            double denom = std::max(std::abs(static_cast<double>(reference[i])), 1.0e-12);
            max_abs_err = std::max(max_abs_err, diff);
            max_rel_err = std::max(max_rel_err, diff / denom);
        }
        validation_status = (max_abs_err <= 1.0e-4 || max_rel_err <= 1.0e-4) ? "pass" : "fail";
    }

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        if (validate) {
            std::cout << "VALIDATION status=" << validation_status
                      << " mode=" << mode
                      << " max_abs_err=" << max_abs_err
                      << " max_rel_err=" << max_rel_err << std::endl;
        }

        std::cout << "\n================ PERFORMANCE DATA ================" << std::endl;
        std::cout << "Execution Mode: " << mode << " (1:Serial, 2:Tiled, 3:Overlapped)" << std::endl;
        std::cout << "Sequence Length (S): " << p.S << " | Tile Sizes: B_M=" << p.B_M << ", B_N=" << p.B_N << std::endl;
        std::cout << "Wall Clock Execution Time (MPI_Wtime): " << wall_ms << " ms" << std::endl;
        std::cout << "CUDA Event Window Duration:           " << gpu_elapsed_ms << " ms" << std::endl;
        std::cout << "==================================================\n" << std::endl;

        std::cout << "METRIC"
                  << " mode=" << mode
                  << " ranks=" << num_ranks
                  << " S=" << p.S
                  << " d_h=" << p.d_h
                  << " H_local=" << p.H_local
                  << " B_M=" << p.B_M
                  << " B_N=" << p.B_N
                  << " tile_count=" << T_M
                  << " d_model=" << d_model
                  << " message_bytes=" << tile_P_bytes
                  << " total_reduced_bytes=" << (tile_P_bytes * static_cast<size_t>(T_M))
                  << " wall_ms=" << wall_ms
                  << " cuda_event_ms=" << gpu_elapsed_ms
                  << " exposed_ms=" << exposed_ms;
        if (validate) {
            std::cout << " validation_status=" << validation_status
                      << " max_abs_err=" << max_abs_err
                      << " max_rel_err=" << max_rel_err;
        }
        std::cout << std::endl;
    }

    // Clean Up Resources
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(projection_stream);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_W_O); cudaFree(d_O_tile); cudaFree(d_O_final);
    for(int i = 0; i < T_M; ++i) {
        cudaFree(d_O_tile_array[i]);
        cudaFree(d_P_tile_array[i]);
        cudaFree(d_P_reduced_array[i]);
    }
    delete[] d_O_tile_array;
    delete[] d_P_tile_array;
    delete[] d_P_reduced_array;

    MPI_Finalize();
    return 0;
}