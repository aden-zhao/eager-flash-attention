cmake_minimum_required(VERSION 3.18)
project(eager_flash_attn LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# --- NCCL: `module load nccl` on Perlmutter exports NCCL_HOME ---
find_library(NCCL_LIB nccl
    HINTS $ENV{NCCL_HOME}/lib $ENV{NCCL_DIR}/lib
          ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64 /opt/nvidia/nccl/lib)
find_path(NCCL_INC nccl.h
    HINTS $ENV{NCCL_HOME}/include $ENV{NCCL_DIR}/include
          ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/include /opt/nvidia/nccl/include)
if(NOT NCCL_LIB OR NOT NCCL_INC)
    message(FATAL_ERROR "NCCL not found. Load the nccl module or set NCCL_LIB and NCCL_INC.")
endif()

# --- MPI: prefer Cray MPICH via CRAY_MPICH_DIR (set by `module load cray-mpich`),
#     linking the CUDA GTL so device buffers can be passed to MPI. Fall back to FindMPI.
if(DEFINED ENV{CRAY_MPICH_DIR})
    set(MPI_INC "$ENV{CRAY_MPICH_DIR}/include")
    get_filename_component(CRAY_MPICH_ROOT "$ENV{CRAY_MPICH_DIR}/../../.." ABSOLUTE)
    find_library(MPI_LIB NAMES mpi_gnu mpi_gnu_123 mpi_gnu_91
        HINTS "$ENV{CRAY_MPICH_DIR}/lib" NO_DEFAULT_PATH)
    find_library(MPI_GTL_LIB NAMES mpi_gtl_cuda
        HINTS "${CRAY_MPICH_ROOT}/gtl/lib" NO_DEFAULT_PATH)
    if(NOT MPI_LIB OR NOT MPI_GTL_LIB)
        message(FATAL_ERROR "Cray MPICH libs not found under $ENV{CRAY_MPICH_DIR} / ${CRAY_MPICH_ROOT}/gtl")
    endif()
    set(MPI_LIBS ${MPI_LIB} ${MPI_GTL_LIB})
    message(STATUS "Using Cray MPICH: ${MPI_LIB} + ${MPI_GTL_LIB}")
else()
    find_package(MPI REQUIRED)
    set(MPI_INC ${MPI_CXX_INCLUDE_DIRS})
    set(MPI_LIBS ${MPI_CXX_LIBRARIES})
endif()

set(COMMON_SOURCES
    attn_main.cu
    run_drivers.cu
    flash_attn.cu
    flash_attn_tc.cu
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
            -lineinfo
            -Xcompiler=-O3
        >
    )
    target_include_directories(${target} PRIVATE ${MPI_INC} ${NCCL_INC})
    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        ${MPI_LIBS}
        ${NCCL_LIB}
    )
    target_link_options(${target} PRIVATE -Wl,--no-as-needed)
endforeach()
