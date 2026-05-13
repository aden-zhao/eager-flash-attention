cmake_minimum_required(VERSION 3.18)
project(fused_attn LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(SOURCES
    attn_tests.cu
    attn_main.cu
    run_drivers.cu
    flash_attn.cu
    baseline_attention.cu
)

add_executable(attn_tests ${SOURCES})

set_target_properties(attn_tests PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 80
)

target_compile_options(attn_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xcompiler=-O3
    >
)

target_include_directories(attn_tests PRIVATE
    /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include
)

target_link_directories(attn_tests PRIVATE
    /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib
    /opt/cray/pe/mpich/9.0.1/gtl/lib
)

target_link_libraries(attn_tests PRIVATE
    CUDA::cudart
    CUDA::cublas
    mpi_gnu_123
    mpi_gtl_cuda
)

target_link_options(attn_tests PRIVATE
    -Wl,--no-as-needed
)
