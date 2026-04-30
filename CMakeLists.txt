cmake_minimum_required(VERSION 3.18)
project(tile_comm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(tile_comm tile_comm.cu baseline_attention.cu)

set_target_properties(tile_comm PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 80
)

# Perlmutter Cray MPICH include path.
target_include_directories(tile_comm PRIVATE
    /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/include
)

# Perlmutter Cray MPICH + CUDA GTL library paths.
target_link_directories(tile_comm PRIVATE
    /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib
    /opt/cray/pe/mpich/9.0.1/gtl/lib
)

target_link_libraries(tile_comm PRIVATE
    CUDA::cudart
    CUDA::cublas
    mpi_gnu_123
    mpi_gtl_cuda
)

# Helps prevent the linker from dropping GTL-related libraries.
target_link_options(tile_comm PRIVATE
    -Wl,--no-as-needed
)
