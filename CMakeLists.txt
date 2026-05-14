cmake_minimum_required(VERSION 3.18)
project(fused_attn LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

find_library(NCCL_LIB nccl
    HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64 /opt/nvidia/nccl/lib)
find_path(NCCL_INC nccl.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/include /opt/nvidia/nccl/include)
if(NOT NCCL_LIB OR NOT NCCL_INC)
    message(FATAL_ERROR "NCCL not found. Load the nccl module or set NCCL_LIB and NCCL_INC.")
endif()

set(COMMON_SOURCES
    attn_main.cu
    run_drivers.cu
    flash_attn.cu
    baseline_attention.cu
)

add_executable(attn_tests attn_tests.cu ${COMMON_SOURCES})
add_executable(attn_bench attn_bench.cu ${COMMON_SOURCES})

foreach(target attn_tests attn_bench)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES 80
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xcompiler=-O3
        >
    )
    target_include_directories(${target} PRIVATE
        /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include
        ${NCCL_INC}
    )
    target_link_directories(${target} PRIVATE
        /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib
        /opt/cray/pe/mpich/9.0.1/gtl/lib
    )
    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        CUDA::cublas
        mpi_gnu_123
        mpi_gtl_cuda
        ${NCCL_LIB}
    )
    target_link_options(${target} PRIVATE
        -Wl,--no-as-needed
    )
endforeach()
